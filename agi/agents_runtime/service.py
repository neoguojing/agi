from __future__ import annotations

from .context_engine import get_context_engine
from .deepagent_builder import build_deep_agent
from .memory_engine import MemorySearchManager, create_default_memory_manager
from .types import AgentRuntimeConfig


class AgentRuntimeService:
    """面向业务调用的统一入口，串联 context/memory/deepagent。"""

    def __init__(
        self,
        config: AgentRuntimeConfig,
        *,
        context_engine_id: str = "legacy",
        memory_manager: MemorySearchManager | None = None,
    ) -> None:
        self.config = config
        self.context_engine = get_context_engine(context_engine_id)
        self.memory_manager = memory_manager or create_default_memory_manager()
        self.agent = build_deep_agent(config)

    def run(self, session_id: str, user_text: str, *, token_budget: int = 40) -> dict:
        self.context_engine.ingest(session_id, [{"role": "user", "content": user_text}])
        self.memory_manager.index(session_id, user_text, metadata={"role": "user"})

        assembled = self.context_engine.assemble(session_id, token_budget=token_budget)
        memory_hits = self.memory_manager.search(user_text, limit=3)

        response = self.agent.invoke({"messages": assembled.messages})
        content = str(response)

        self.context_engine.ingest(session_id, [{"role": "assistant", "content": content}])
        self.memory_manager.index(session_id, content, metadata={"role": "assistant"})

        compact_result = self.context_engine.compact(session_id, reason="turn_end")
        return {
            "response": response,
            "assembled": assembled,
            "memory_hits": memory_hits,
            "compact": compact_result,
            "memory_status": self.memory_manager.status,
        }
