from __future__ import annotations

from typing import Any

from ..engines.multimodal import Modality, MultiModalRequest


class LegacyTaskAdapter:
    """把 agi/tasks 下老任务包装成新框架可调用处理器。"""

    def __init__(self) -> None:
        from agi.tasks.task_factory import (
            TASK_IMAGE_GEN,
            TASK_LLM,
            TASK_MULTI_MODEL,
            TASK_SPEECH_TEXT,
            TASK_TTS,
            TaskFactory,
        )

        self._factory = TaskFactory
        self._task_ids = {
            Modality.TEXT: TASK_LLM,
            Modality.IMAGE_GENERATE: TASK_IMAGE_GEN,
            Modality.IMAGE_EDIT: TASK_IMAGE_GEN,
            Modality.IMAGE_UNDERSTAND: TASK_MULTI_MODEL,
            Modality.MULTIMODAL: TASK_MULTI_MODEL,
            Modality.AUDIO_TRANSCRIBE: TASK_SPEECH_TEXT,
            Modality.AUDIO_GENERATE: TASK_TTS,
        }

    def invoke(self, modality: Modality, request: MultiModalRequest) -> Any:
        task_id = self._task_ids[modality]
        task = self._factory.create_task(task_id)
        payload = self._to_payload(modality, request)
        if hasattr(task, "invoke"):
            return task.invoke(payload)
        return task

    @staticmethod
    def _to_payload(modality: Modality, request: MultiModalRequest) -> Any:
        if modality in {Modality.TEXT, Modality.IMAGE_GENERATE, Modality.IMAGE_EDIT}:
            return request.text or ""
        if modality == Modality.AUDIO_TRANSCRIBE:
            return {
                "audio": request.audio,
                "audio_base64": request.audio_base64,
                "audio_mime_type": request.audio_mime_type,
                "text": request.text,
            }
        if modality == Modality.AUDIO_GENERATE:
            return request.text or ""
        return {
            "text": request.text,
            "audio": request.audio,
            "audio_base64": request.audio_base64,
            "audio_mime_type": request.audio_mime_type,
            "image": request.image,
            "image_base64": request.image_base64,
            "image_mime_type": request.image_mime_type,
            "metadata": request.metadata or {},
        }


class LegacyKnowledgeAdapter:
    """封装旧 KnowledgeManager，统一知识库构建/检索接口。"""

    def __init__(self) -> None:
        from agi.tasks.task_factory import TaskFactory

        self.manager = TaskFactory.get_knowledge_manager()

    def build(self, file_path: str, *, tenant_id: str = "default") -> Any:
        if hasattr(self.manager, "load_knowledge"):
            return self.manager.load_knowledge(file_path, tenant_id=tenant_id)
        return None

    def search(self, query: str, *, tenant_id: str = "default", top_k: int = 4) -> list[Any]:
        if hasattr(self.manager, "search"):
            result = self.manager.search(query, tenant_id=tenant_id, top_k=top_k)
            return result if isinstance(result, list) else [result]
        return []
