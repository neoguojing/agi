import asyncio
import datetime
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any, Callable, Tuple

from deepagents.backends.protocol import BackendProtocol
from .context import (
    USER_PROFILE_CONTEXT,
    get_session_entity_id,
    get_session_context_id,
    BackendMixin,
)

# =========================
# Constants
# =========================

class ContextKeys:
    ENV = "env"
    USER = "user_profile"
    SESSION = "session"
    ENTITY = "entities"
    KNOWLEDGE = "knowledge"

# =========================
# Providers
# =========================

class ContextProvider(ABC):
    @abstractmethod
    async def load(self, runtime: Any, state: Dict) -> Dict:
        pass


class EnvironmentProvider(ContextProvider):
    async def load(self, runtime, state) -> dict:
        now = datetime.datetime.now()
        return {
            ContextKeys.ENV: {
                "local_time": now.strftime("%Y-%m-%d %H:%M:%S"),
                "weekday": now.strftime("%A"),
                "platform": getattr(runtime.context, "platform", "web"),
                "location": getattr(runtime.context, "location", "unknown"),
            }
        }


# =========================
# User Profile
# =========================

class UserProfileProvider(ContextProvider, BackendMixin):
    def __init__(self, backend: Optional[Any] = None):
        self.backend = backend

    def _get_backend(self, runtime) -> Optional[BackendProtocol]:
        return self.backend(runtime) if callable(self.backend) else self.backend

    async def load(self, runtime, state):
        user_id = runtime.context.user_id
        path = "/profiles/profile.json"  # ✅ 修复路径

        backend = self._get_backend(runtime)

        if backend is not None:
            raw = await backend.aread(path)
            print(f"********{path}**********{raw}")

            data = self._load_json(raw, {})
            print(f"******************{data}")
            return {ContextKeys.USER: data}

        # fallback
        profile_data = await runtime.store.aget(user_id, USER_PROFILE_CONTEXT)
        return {ContextKeys.USER: profile_data.value if profile_data else {}}


# =========================
# Session
# =========================

class SessionContextProvider(ContextProvider, BackendMixin):
    def __init__(self, backend: Optional[Any] = None):
        self.backend = backend

    def _get_backend(self, runtime) -> Optional[BackendProtocol]:
        return self.backend(runtime) if callable(self.backend) else self.backend

    async def load(self, runtime, state) -> dict:
        user_id = runtime.context.user_id
        session_id = runtime.context.conversation_id
        path = "/sessions/session.json"  # ✅ 修复路径

        backend = self._get_backend(runtime)

        if backend is not None:
            raw = await backend.aread(path)
            session_meta = self._load_json(raw, {})
        else:
            existing = await runtime.store.aget(
                user_id, get_session_context_id(session_id)
            )
            session_meta = existing.value if existing else {}

        messages = state.get("messages", []) or []

        return {
            ContextKeys.SESSION: {
                "turn_count": len(messages) // 2,
                "metadata": session_meta,
            }
        }


# =========================
# Entities
# =========================

class EntityContextProvider(ContextProvider, BackendMixin):
    def __init__(self, backend: Optional[Any] = None):
        self.backend = backend

    def _get_backend(self, runtime) -> Optional[BackendProtocol]:
        return self.backend(runtime) if callable(self.backend) else self.backend

    async def load(self, runtime, state) -> dict:
        user_id = runtime.context.user_id
        session_id = runtime.context.conversation_id
        path = "/entities/entities.json"  # ✅ 修复路径

        backend = self._get_backend(runtime)

        if backend is not None:
            raw = await backend.aread(path)
            active_entities = self._load_json(raw, [])  # ✅ 修复类型
        else:
            existing = await runtime.store.aget(
                user_id, get_session_entity_id(session_id)
            )
            active_entities = existing.value if existing else []

        return {ContextKeys.ENTITY: active_entities}


# =========================
# Knowledge
# =========================

class KnowledgeProvider(ContextProvider):
    def __init__(self, retriever):
        self.retriever = retriever

    async def load(self, runtime, state):
        messages = state.get("messages", [])
        last_msg = messages[-1].content if messages else ""
        docs = await self.retriever.aretrieve(last_msg) if last_msg else ""
        return {ContextKeys.KNOWLEDGE: docs}


# =========================
# Renderer
# =========================

FormatterConfig = Tuple[str, Callable[[Any], str]]


class ContextRenderer:
    def __init__(self):
        self._formatters: Dict[str, FormatterConfig] = {
            ContextKeys.ENV: ("Current Environment", self._fmt_env),
            ContextKeys.USER: ("User Persona & Preferences", self._fmt_user),
            ContextKeys.SESSION: ("Session State", self._fmt_session),
            ContextKeys.ENTITY: ("Active Entities", self._fmt_entities),
            ContextKeys.KNOWLEDGE: ("Relevant Knowledge", lambda x: str(x)),
        }

    def render(self, ctx: Dict[str, Any]) -> str:
        sections = ["Use the context below to tailor your response.", "---"]

        for key, (header, formatter) in self._formatters.items():
            data = ctx.get(key)
            if data:
                content = formatter(data)
                if content:
                    sections.append(f"## {header}\n{content}")

        sections.append("---\nRespond to the user's latest query.")
        return "\n".join(sections)

    def _fmt_env(self, d):
        if not isinstance(d, dict): return str(d)
        return f"- Time: {d.get('local_time')} ({d.get('weekday')})\n- Platform: {d.get('platform')}"

    def _fmt_user(self, d):
        if not isinstance(d, dict): return str(d)
        return "\n".join([f"- {k}: {v}" for k, v in d.items() if v])

    def _fmt_session(self, d):
        if not isinstance(d, dict): return str(d)
        meta = d.get("metadata", {})
        return f"- Intent: {meta.get('current_intent', 'General')}\n- Turn: {d.get('turn_count', 0)}"

    def _fmt_entities(self, d):
        if not isinstance(d, list): return str(d)

        safe_lines = []
        for e in d:
            if isinstance(e, dict):
                name = e.get("name", "unknown")
                typ = e.get("type", "unknown")
                attr = e.get("attributes", {})
                safe_lines.append(f"- {name} ({typ}): {attr}")

        return "\n".join(safe_lines)


# =========================
# Manager
# =========================

class AsyncContextManager:
    def __init__(self, providers: List[ContextProvider], timeout: float = 2.0):
        self.providers = providers
        self.timeout = timeout
        self.renderer = ContextRenderer()

    async def get_context_str(self, runtime, state) -> str:
        tasks = [
            asyncio.wait_for(p.load(runtime, state), timeout=self.timeout)
            for p in self.providers
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        merged = {}
        for r in results:
            if isinstance(r, dict):
                merged.update(r)

        return self.renderer.render(merged)


# =========================
# Factory
# =========================

def create_default_context_manager(retriever=None, backend=None):
    providers = [
        EnvironmentProvider(),
        UserProfileProvider(backend),
        SessionContextProvider(backend),
        EntityContextProvider(backend),
    ]

    if retriever:
        providers.append(KnowledgeProvider(retriever))

    return AsyncContextManager(providers)