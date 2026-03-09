from __future__ import annotations

from typing import Any

from ..engines.context_engine import get_context_engine
from ..integration.deepagent_builder import build_deep_agent
from .harness import TodoManager
from ..engines.knowledge import KnowledgeFusionService
from ..integration.legacy_adapters import LegacyKnowledgeAdapter, LegacyTaskAdapter
from ..engines.memory_engine import MemorySearchManager, create_default_memory_manager
from ..core.messages import MediaInput, create_multimodal_human_message, message_to_payload
from ..engines.multimodal import Modality, MultiModalExecutor, MultiModalRequest, MultiModalRouter
from .session_context import SessionContextManager
from .hitl import build_resume_payload, extract_interrupt_actions
from ..core.types import AgentRuntimeConfig


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
        session_manager: SessionContextManager | None = None,
        todo_manager: TodoManager | None = None,
    ) -> None:
        self.config = config
        self.context_engine = get_context_engine(context_engine_id)
        self.memory_manager = memory_manager or create_default_memory_manager()
        self.agent = build_deep_agent(config)
        self.session_manager = session_manager or SessionContextManager()
        self.todo_manager = todo_manager or TodoManager()

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

    @staticmethod
    def _to_human_payload(request: MultiModalRequest) -> dict[str, Any]:
        msg = create_multimodal_human_message(
            text=request.text,
            image=MediaInput(url=request.image, base64=request.image_base64, mime_type=request.image_mime_type),
            audio=MediaInput(url=request.audio, base64=request.audio_base64, mime_type=request.audio_mime_type),
        )
        return message_to_payload(msg)

    def _build_memory_layout_prompt(self) -> dict[str, str] | None:
        if not self.config.enable_long_term_memory:
            return None
        prefix = self.config.long_term_memory_prefix
        content = (
            "持久化记忆目录约定:\n"
            f"- 长期记忆路径前缀: {prefix}\n"
            f"- 用户偏好: {prefix}preferences.txt\n"
            f"- 研究资料: {prefix}research/notes.txt\n"
            f"- 项目知识: {prefix}project/knowledge.md\n"
            "当用户要求长期保存时，优先写入上述长期记忆目录。"
        )
        return {"role": "system", "content": content}

    def invoke_text_turn(
        self,
        session_id: str,
        messages: list[dict[str, Any]],
        *,
        thread_id: str | None = None,
    ) -> dict[str, Any]:
        """用于 HITL 场景：直接触发文本轮次，并返回可能的 __interrupt__。"""
        state = self.session_manager.get_or_create(session_id, thread_id=thread_id)
        run_config = self.session_manager.to_run_config(state)
        return self.agent.invoke({"messages": messages}, config=run_config)

    def resume_interrupt(
        self,
        session_id: str,
        decisions: list[dict[str, Any]],
        *,
        thread_id: str | None = None,
    ) -> dict[str, Any]:
        """按同一 thread_id 恢复中断执行。"""
        state = self.session_manager.get_or_create(session_id, thread_id=thread_id)
        run_config = self.session_manager.to_run_config(state)
        from langgraph.types import Command

        return self.agent.invoke(Command(resume=build_resume_payload(decisions)), config=run_config)

    @staticmethod
    def inspect_interrupts(result: dict[str, Any]):
        return extract_interrupt_actions(result)

    async def run_auto(
        self,
        session_id: str,
        request: MultiModalRequest,
        *,
        token_budget: int = 40,
        collection: str | list[str] | None = None,
        tenant_id: str | None = None,
        thread_id: str | None = None,
    ) -> dict[str, Any]:
        route, legacy_result = self.executor.invoke(request, self.router)

        state = self.session_manager.get_or_create(session_id, thread_id=thread_id)

        content = request.text or ""
        human_payload = self._to_human_payload(request)
        self.context_engine.ingest(session_id, [human_payload])
        self.memory_manager.index(session_id, content, metadata={"role": "user", "modality": route.modality.value})

        assembled = self.context_engine.assemble(session_id, token_budget=token_budget)
        memory_layout_prompt = self._build_memory_layout_prompt()
        if memory_layout_prompt:
            assembled.messages = [memory_layout_prompt] + assembled.messages

        kb_hits = []
        if collection and content:
            kb_hits = await self.knowledge_service.search(collection, content, tenant_id=tenant_id, top_k=4)
            assembled.messages = self.knowledge_service.inject_to_messages(assembled.messages, kb_hits)

        run_config = self.session_manager.to_run_config(state)

        # harness planning capability: keep a lightweight todo list in session state
        if content:
            self.todo_manager.write_todos(
                session_id,
                [{"id": "analyze", "content": "分析用户请求", "status": "completed"}, {"id": "answer", "content": "生成并返回结果", "status": "in_progress"}],
            )

        if route.modality == Modality.TEXT:
            if kb_hits:
                assembled.messages = [{
                    "role": "system",
                    "content": "若任务复杂请优先委派给 knowledge-researcher 子代理做检索与总结，再返回精简答案。",
                }] + assembled.messages
            response = self.agent.invoke({"messages": assembled.messages}, config=run_config)
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
        self.todo_manager.update_status(session_id, "answer", "completed")
        return {
            "session": state,
            "route": route,
            "response": response,
            "assembled": assembled,
            "memory_hits": self.memory_manager.search(content, limit=3) if content else [],
            "knowledge_hits": kb_hits,
            "compact": compact_result,
            "memory_status": self.memory_manager.status,
            "todos": self.todo_manager.list_todos(session_id),
        }
