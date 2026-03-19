import asyncio
import datetime
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Optional, Any, Callable,Tuple

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from pydantic import BaseModel, Field
from .context import USER_PROFILE_CONTEXT,get_session_entity_id,get_session_context_id

# --- Constants & Key Management ---

class ContextKeys:
    """Centralized keys to ensure consistency between Providers and Renderer."""
    ENV = "env"
    USER = "user_profile"
    SESSION = "session"
    ENTITY = "entities"
    KNOWLEDGE = "knowledge"

# --- Context Providers ---

class ContextProvider(ABC):
    @abstractmethod
    async def load(self, runtime: Any, state: Dict) -> Dict:
        """Asynchronously load specific context data."""
        pass

class EnvironmentProvider(ContextProvider):
    async def load(self, runtime, state) -> dict:
        now = datetime.datetime.now()
        return {
            ContextKeys.ENV: {
                "local_time": now.strftime("%Y-%m-%d %H:%M:%S"),
                "weekday": now.strftime("%A"),
                "platform": getattr(runtime.context, "platform", "web"),
                "location": getattr(runtime.context, "location", "unknown")
            }
        }

class UserProfileProvider(ContextProvider):
    async def load(self, runtime, state):
        user_id = runtime.context.user_id
        # Assuming USER_PROFILE_CONTEXT is defined in your constants
        profile_data = runtime.store.get(user_id, USER_PROFILE_CONTEXT)
        return {
            ContextKeys.USER: profile_data.value if profile_data else {}
        }

class SessionContextProvider(ContextProvider):
    async def load(self, runtime, state) -> dict:
        user_id = runtime.context.user_id
        session_id = runtime.context.conversation_id
        # get_session_context_id is your helper for namespacing
        session_meta = runtime.store.get(user_id, get_session_context_id(session_id))
        
        messages = state.get("messages", [])
        return {
            ContextKeys.SESSION: {
                "turn_count": len(messages) // 2,
                "metadata": session_meta.value if session_meta else {}
            }
        }

class EntityContextProvider(ContextProvider):
    async def load(self, runtime, state) -> dict:
        user_id = runtime.context.user_id
        session_id = runtime.context.conversation_id
        active_entities = runtime.store.get(user_id, get_session_entity_id(session_id))
        return {
            ContextKeys.ENTITY: active_entities.value if active_entities else []
        }

class KnowledgeProvider(ContextProvider):
    def __init__(self, retriever):
        self.retriever = retriever

    async def load(self, runtime, state):
        last_msg = state["messages"][-1].content if state["messages"] else ""
        docs = await self.retriever.aretrieve(last_msg) if last_msg else ""
        return {ContextKeys.KNOWLEDGE: docs}

# --- Context Builder & Renderer ---

FormatterConfig = Tuple[str, Callable[[Any], str]]

class ContextRenderer:
    """Handles the transformation of raw context data into a structured SystemMessage."""
    
    def __init__(self):
        # 明确指定类型为 Dict[str, FormatterConfig]
        self._formatters: Dict[str, FormatterConfig] = {
            ContextKeys.ENV: ("Current Environment", self._fmt_env),
            ContextKeys.USER: ("User Persona & Preferences", self._fmt_user),
            ContextKeys.SESSION: ("Session State", self._fmt_session),
            ContextKeys.ENTITY: ("Active Entities", self._fmt_entities),
            ContextKeys.KNOWLEDGE: ("Relevant Knowledge", lambda x: str(x))
        }

    def render(self, ctx: Dict[str, Any]) -> List[SystemMessage]:
        core_instruction = "You are a professional AI Assistant. Use the context below to tailor your response."
        sections = [core_instruction, "---"]

        for key, config in self._formatters.items():
            # 显式解包，提高代码可读性
            header, formatter = config
            
            if data := ctx.get(key):
                # 此时 Lint 会识别到 formatter 是 Callable，不再报错
                formatted_content = formatter(data)
                if formatted_content:
                    sections.append(f"### {header}\n{formatted_content}")

        sections.append("---\nRespond to the user's latest query based on the information above.")
        return [SystemMessage(content="\n\n".join(sections))]

    # 确保辅助函数遵循 (Any) -> str 的签名
    def _fmt_env(self, d: Any) -> str:
        # 增加一些健壮性检查
        if not isinstance(d, dict): return str(d)
        return f"- Time: {d.get('local_time')} ({d.get('weekday')})\n- Platform: {d.get('platform')}"

    def _fmt_user(self, d: Any) -> str:
        if not isinstance(d, dict): return str(d)
        return "\n".join([f"- {k.replace('_', ' ').title()}: {v}" for k, v in d.items() if v])

    def _fmt_session(self, d: Any) -> str:
        if not isinstance(d, dict): return str(d)
        meta = d.get("metadata", {})
        return f"- Intent: {meta.get('current_intent', 'General')}\n- Turn: {d.get('turn_count', 0)}"

    def _fmt_entities(self, d: Any) -> str:
        if not isinstance(d, list): return str(d)
        return "\n".join([f"- {e['name']} ({e['type']}): {e.get('attributes', {})}" for e in d])

class AsyncContextManager:
    """Unified Pipeline to Build and Render Context."""
    
    def __init__(self, providers: List[ContextProvider], timeout: float = 2.0):
        self.providers = providers
        self.timeout = timeout
        self.renderer = ContextRenderer()

    async def get_context_message(self, runtime, state) -> List[SystemMessage]:
        tasks = [
            asyncio.wait_for(p.load(runtime, state), timeout=self.timeout) 
            for p in self.providers
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        merged_context = {}
        for r in results:
            if isinstance(r, dict):
                merged_context.update(r)
            # Log exceptions/timeouts here in production
            
        return self.renderer.render(merged_context)

# --- Default Implementation Factory ---

def create_default_context_manager(retriever=None) -> AsyncContextManager:
    providers = [
        EnvironmentProvider(),
        UserProfileProvider(),
        SessionContextProvider(),
        EntityContextProvider()
    ]
    if retriever:
        providers.append(KnowledgeProvider(retriever))
    
    return AsyncContextManager(providers=providers)