from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable


class Modality(str, Enum):
    TEXT = "text"
    IMAGE_GENERATE = "image_generate"
    IMAGE_UNDERSTAND = "image_understand"
    IMAGE_EDIT = "image_edit"
    AUDIO_TRANSCRIBE = "audio_transcribe"
    AUDIO_GENERATE = "audio_generate"
    MULTIMODAL = "multimodal"


@dataclass(slots=True)
class MultiModalRequest:
    text: str | None = None
    image: str | None = None
    audio: str | None = None
    target: str | None = None
    metadata: dict[str, Any] | None = None


@dataclass(slots=True)
class RouteResult:
    modality: Modality
    reason: str


class MultiModalRouter:
    """轻量规则路由器：自动识别文本/图像/语音任务并选择执行链。"""

    def route(self, request: MultiModalRequest) -> RouteResult:
        text = (request.text or "").lower()
        has_image = bool(request.image)
        has_audio = bool(request.audio)

        if has_audio and request.target == "text":
            return RouteResult(Modality.AUDIO_TRANSCRIBE, "audio file provided and target=text")
        if has_audio and request.target == "audio":
            return RouteResult(Modality.AUDIO_GENERATE, "audio continuation/generation requested")
        if has_audio and request.text:
            return RouteResult(Modality.MULTIMODAL, "audio+text provided")

        if has_image:
            if any(token in text for token in ["edit", "修改", "重绘", "换背景", "擦除"]):
                return RouteResult(Modality.IMAGE_EDIT, "image exists and edit intent detected")
            if any(token in text for token in ["describe", "识别", "看图", "图里", "what is in"]):
                return RouteResult(Modality.IMAGE_UNDERSTAND, "image exists and understanding intent detected")
            return RouteResult(Modality.MULTIMODAL, "image with generic prompt")

        if any(token in text for token in ["draw", "generate image", "生成图片", "画一张", "海报"]):
            return RouteResult(Modality.IMAGE_GENERATE, "image generation intent from text")
        if any(token in text for token in ["朗读", "语音", "tts", "speak"]):
            return RouteResult(Modality.AUDIO_GENERATE, "speech generation intent from text")

        return RouteResult(Modality.TEXT, "default text pathway")


class MultiModalExecutor:
    """按 modality 分发到具体处理器，处理器可以是旧 TaskFactory Runnable 或新链路。"""

    def __init__(self, handlers: dict[Modality, Callable[[MultiModalRequest], Any]]) -> None:
        self.handlers = handlers

    def invoke(self, request: MultiModalRequest, router: MultiModalRouter | None = None) -> tuple[RouteResult, Any]:
        active_router = router or MultiModalRouter()
        route = active_router.route(request)
        handler = self.handlers.get(route.modality)
        if handler is None:
            raise KeyError(f"No handler configured for modality={route.modality}")
        return route, handler(request)
