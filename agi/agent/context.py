from dataclasses import dataclass
from langchain.agents.middleware import dynamic_prompt
from langchain.agents.middleware import wrap_model_call
from langchain.tools import tool, ToolRuntime
import asyncio
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from abc import ABC, abstractmethod

@dataclass
class Context:
    user_id: str
    role: str


@tool
def get_user_profile(runtime: ToolRuntime[Context]) -> str:
    """Get user profile info"""

    user_id = runtime.context.user_id
    store = runtime.store

    profile = store.get(("users",), user_id)

    if not profile:
        return "No profile found"

    return f"User name: {profile.value['name']}"


@dynamic_prompt
def role_prompt(request):
    role = request.runtime.context.role

    if role == "admin":
        return "You are an admin assistant."
    else:
        return "You are a normal assistant."
    

@wrap_model_call
def inject_user_memory(request, handler):

    user_id = request.runtime.context.user_id
    store = request.runtime.store

    memory = store.get(("memory",), user_id)

    if memory:
        messages = [
            *request.messages,
            {
                "role": "system",
                "content": f"User preference: {memory.value}"
            }
        ]
        request = request.override(messages=messages)

    return handler(request)


class ContextProvider(ABC):
    @abstractmethod
    async def load(self, runtime, state) -> dict:
        pass

class UserProfileProvider(ContextProvider):
    async def load(self, runtime, state):
        return {
            "role": runtime.context.role,
            "user_id": runtime.context.user_id
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

        if ctx.get("knowledge"):
            messages.append({
                "role": "system",
                "content": f"Relevant Knowledge:\n{ctx['knowledge']}"
            })

        if ctx.get("profile"):
            messages.append({
                "role": "system",
                "content": f"Relevant Knowledge:\n{ctx['knowledge']}"
            })

        return messages



NAMESPACE = ("user_profile",)

class UserPersona(BaseModel):
    # 静态信息
    full_name: Optional[str] = None
    job_role: Optional[str] = None
    
    # 动态偏好
    interests: List[str] = Field(default_factory=list, description="用户感兴趣的领域")
    communication_style: str = Field(default="professional", description="用户偏好的交流风格，如：幽默、严谨、简洁")
    
    # 技能等级 (决定回复深度)
    technical_level: Dict[str, str] = Field(
        default_factory=dict, 
        description="用户在特定领域的等级，例如 {'Python': 'beginner'}"
    )

    # 长期目标/任务
    current_goals: List[str] = Field(default_factory=list, description="用户当前正在关注的任务或目标")

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