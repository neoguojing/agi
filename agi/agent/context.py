from dataclasses import dataclass
from langchain.agents.middleware import dynamic_prompt
from langchain.agents.middleware import wrap_model_call
from langchain.tools import tool, ToolRuntime
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
import asyncio
from abc import ABC, abstractmethod
from typing import List, Dict, Optional
from pydantic import BaseModel, Field

@dataclass
class Context:
    user_id: str
    conversation_id: str

class ContextProvider(ABC):
    @abstractmethod
    async def load(self, runtime, state) -> dict:
        pass

class UserProfileProvider(ContextProvider):
    async def load(self, runtime, state):
        # 从 Store 中根据 NAMESPACE 获取数据
        user_id = runtime.context.user_id
        profile_data = runtime.store.get(NAMESPACE, user_id)
        
        return {
            "user_profile": profile_data.value if profile_data else {}
        }
    
class KnowledgeProvider(ContextProvider):
    def __init__(self, retriever):
        self.retriever = retriever

    async def load(self, runtime, state):
        query = state["messages"][-1].content

        docs = await self.retriever.aretrieve(query)

        return {
            "knowledge": docs
        }
    
class ToolContextProvider(ContextProvider):
    def load(self, runtime, state):
        return {
            "tools": runtime.store.get(("tools",), "all")
        }
    


class AsyncContextBuilder:
    def __init__(self, providers, timeout=2.0):
        self.providers = providers
        self.timeout = timeout

    async def build(self, runtime, state):
        # 增加超时控制，确保 LLM 响应速度
        tasks = [
            asyncio.wait_for(p.load(runtime, state), timeout=self.timeout) 
            for p in self.providers
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)

        context = {}
        for r in results:
            if isinstance(r, asyncio.TimeoutError):
                # 记录超时日志，但不中断流程
                continue
            if isinstance(r, Exception):
                continue
            context.update(r)

        return context

class ContextRenderer:
    def render(self, ctx: dict) -> list:
        messages = []
        
        # Inject RAG / Knowledge
        if ctx.get("knowledge"):
            messages.append(SystemMessage(
                content=f"### Relevant Knowledge\n{ctx['knowledge']}"
            ))

        # Inject User Persona / Profile
        if ctx.get("user_profile"):
            profile = ctx["user_profile"]
            # Formatting the dict for better LLM readability
            profile_str = "\n".join([f"- {k}: {v}" for k, v in profile.items() if v])
            messages.append(SystemMessage(
                content=f"### User Persona & Preferences\n{profile_str}"
            ))
            
        return messages



NAMESPACE = ("user_profile",)

class UserPersona(BaseModel):
    """
    Represents a persistent profile of the user to tailor AI interactions.
    This schema is used by the LLM to extract and update user-specific context.
    """
    
    # --- Static / Demographic Information ---
    full_name: Optional[str] = Field(
        None, 
        description="The user's legal or preferred full name."
    )
    job_role: Optional[str] = Field(
        None, 
        description="The user's current professional title or primary occupation (e.g., 'Senior DevOps Engineer')."
    )
    
    # --- Dynamic Preferences ---
    interests: List[str] = Field(
        default_factory=list, 
        description="A list of topics, industries, or technologies the user shows consistent interest in."
    )
    communication_style: str = Field(
        default="professional", 
        description="The user's preferred tone and language style (e.g., 'humorous', 'concise', 'academic', 'layman-friendly')."
    )
    
    # --- Competency & Depth Control ---
    technical_level: Dict[str, str] = Field(
        default_factory=dict, 
        description=(
            "A mapping of specific domains to the user's proficiency level. "
            "Example: {'Python': 'expert', 'Quantum Physics': 'beginner'}. "
            "Used to calibrate the technical depth of AI responses."
        )
    )

    # --- Temporal Context ---
    current_goals: List[str] = Field(
        default_factory=list, 
        description="Active projects, immediate objectives, or long-term milestones the user is currently working toward."
    )

class ProfileUpdater:
    def __init__(self, extractor_model):
        # 建议使用支持 structured_output 的模型
        self.extractor_model = extractor_model

    async def update(self, runtime, messages, ai_response):
        user_id = runtime.context.user_id
        store = runtime.store

        last_user_msg = messages[-1].content
        if len(last_user_msg) < 5:
            return

        existing_profile = store.get(NAMESPACE, user_id)

        if existing_profile and existing_profile.value:
            current_val = UserPersona(**existing_profile.value).model_dump()
        else:
            current_val = UserPersona().model_dump()

        extraction_prompt = f"""
        Analyze conversation and extract ONLY explicitly stated user info.

        Current Profile:
        {current_val}

        Last User Message:
        {last_user_msg}

        AI Response:
        {ai_response}

        Return ONLY JSON patch.
        """

        structured_extractor = self.extractor_model.with_structured_output(UserPersona)

        try:
            patch = await structured_extractor.ainvoke(extraction_prompt)
        except Exception:
            return

        updated_val = self._smart_merge(current_val, patch.model_dump(exclude_none=True))

        store.put(NAMESPACE, user_id, updated_val)

    def _smart_merge(self, old: dict, new: dict) -> dict:
        # 1. 标量字段（直接覆盖）
        for key in ["full_name", "job_role", "communication_style"]:
            if new.get(key):
                old[key] = new[key]

        # 2. list 去重合并
        for key in ["interests", "current_goals"]:
            if new.get(key):
                old[key] = list(set(old.get(key, []) + new[key]))

        # 3. dict merge（technical_level）
        if new.get("technical_level"):
            old.setdefault("technical_level", {})
            old["technical_level"].update(new["technical_level"])

        return old