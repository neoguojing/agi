from __future__ import annotations

from typing import Any

from .context_engine import get_context_engine
from .deepagent_builder import build_deep_agent
from .knowledge import KnowledgeFusionService
from .legacy_adapters import LegacyKnowledgeAdapter, LegacyTaskAdapter
from .memory_engine import MemorySearchManager, create_default_memory_manager
from .multimodal import Modality, MultiModalExecutor, MultiModalRequest, MultiModalRouter
from .types import AgentRuntimeConfig


class AgentRuntimeService:
    """统一入口：融合多模态、知识库、上下文记忆与 deepagent。"""

    def __init__(
        self,
        config: AgentRuntimeConfig,
        *,
        context_engine_id: str = "legacy",
        memory_manager: MemorySearchManager | None = None,
        legacy_task_adapter: LegacyTaskAdapter | None = None,
        knowledge_adapter: LegacyKnowledgeAdapter | None = None,
    ) -> None:
        self.config = config
        self.context_engine = get_context_engine(context_engine_id)
        self.memory_manager = memory_manager or create_default_memory_manager()
        self.agent = build_deep_agent(config)

        self.legacy_task_adapter = legacy_task_adapter or LegacyTaskAdapter()
        kb_adapter = knowledge_adapter or LegacyKnowledgeAdapter()
        self.knowledge_service = KnowledgeFusionService(kb_adapter.manager)

        self.router = MultiModalRouter()
        self.executor = MultiModalExecutor(
            {
                Modality.TEXT: lambda req: None,
                Modality.IMAGE_GENERATE: lambda req: self.legacy_task_adapter.invoke(Modality.IMAGE_GENERATE, req),
                Modality.IMAGE_EDIT: lambda req: self.legacy_task_adapter.invoke(Modality.IMAGE_EDIT, req),
                Modality.IMAGE_UNDERSTAND: lambda req: self.legacy_task_adapter.invoke(Modality.IMAGE_UNDERSTAND, req),
                Modality.AUDIO_TRANSCRIBE: lambda req: self.legacy_task_adapter.invoke(Modality.AUDIO_TRANSCRIBE, req),
                Modality.AUDIO_GENERATE: lambda req: self.legacy_task_adapter.invoke(Modality.AUDIO_GENERATE, req),
                Modality.MULTIMODAL: lambda req: self.legacy_task_adapter.invoke(Modality.MULTIMODAL, req),
            }
        )

    async def run_auto(
        self,
        session_id: str,
        request: MultiModalRequest,
        *,
        token_budget: int = 40,
        collection: str | list[str] | None = None,
        tenant_id: str | None = None,
    ) -> dict[str, Any]:
        route, legacy_result = self.executor.invoke(request, self.router)

        content = request.text or ""
        self.context_engine.ingest(session_id, [{"role": "user", "content": content}])
        self.memory_manager.index(session_id, content, metadata={"role": "user", "modality": route.modality.value})

        assembled = self.context_engine.assemble(session_id, token_budget=token_budget)

        kb_hits = []
        if collection and content:
            kb_hits = await self.knowledge_service.search(collection, content, tenant_id=tenant_id, top_k=4)
            assembled.messages = self.knowledge_service.inject_to_messages(assembled.messages, kb_hits)

        if route.modality == Modality.TEXT:
            response = self.agent.invoke({"messages": assembled.messages})
        else:
            response = {
                "modality": route.modality.value,
                "reason": route.reason,
                "legacy_result": legacy_result,
            }

        response_text = str(response)
        self.context_engine.ingest(session_id, [{"role": "assistant", "content": response_text}])
        self.memory_manager.index(
            session_id,
            response_text,
            metadata={"role": "assistant", "modality": route.modality.value},
        )

        compact_result = self.context_engine.compact(session_id, reason="turn_end")
        return {
            "route": route,
            "response": response,
            "assembled": assembled,
            "memory_hits": self.memory_manager.search(content, limit=3) if content else [],
            "knowledge_hits": kb_hits,
            "compact": compact_result,
            "memory_status": self.memory_manager.status,
        }
