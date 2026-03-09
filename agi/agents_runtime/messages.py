from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class MediaInput:
    url: str | None = None
    base64: str | None = None
    mime_type: str | None = None


def _create_text_block(text: str) -> dict[str, Any]:
    from langchain_core.messages.content import create_text_block

    return create_text_block(text)


def _create_image_block(image: MediaInput) -> dict[str, Any]:
    from langchain_core.messages.content import create_image_block

    return create_image_block(url=image.url, base64=image.base64, mime_type=image.mime_type)


def _create_audio_block(audio: MediaInput) -> dict[str, Any]:
    # 当前 langchain content 工厂未统一提供 create_audio_block，这里按 ContentBlock 协议构造
    block: dict[str, Any] = {"type": "audio"}
    if audio.url:
        block["url"] = audio.url
    if audio.base64:
        block["base64"] = audio.base64
    if audio.mime_type:
        block["mime_type"] = audio.mime_type
    return block


def create_multimodal_human_message(
    *,
    text: str | None = None,
    image: MediaInput | None = None,
    audio: MediaInput | None = None,
    extras: list[dict[str, Any]] | None = None,
):
    """基于 LangChain HumanMessage + ContentBlock 构造多模态输入。"""
    from langchain_core.messages import HumanMessage

    content_blocks: list[dict[str, Any]] = []
    if text:
        content_blocks.append(_create_text_block(text))
    if image and (image.url or image.base64):
        content_blocks.append(_create_image_block(image))
    if audio and (audio.url or audio.base64):
        content_blocks.append(_create_audio_block(audio))
    if extras:
        content_blocks.extend(extras)

    return HumanMessage(content=content_blocks if content_blocks else (text or ""))


def create_knowledge_system_message(knowledge_text: str):
    from langchain_core.messages import SystemMessage

    return SystemMessage(content=knowledge_text)


def message_to_payload(message: Any) -> dict[str, Any]:
    """把 LangChain Message 归一为 deepagents 可消费的 role/content 结构。"""
    role = "system"
    msg_type = getattr(message, "type", "")
    if msg_type in {"human", "user"}:
        role = "user"
    elif msg_type in {"ai", "assistant"}:
        role = "assistant"
    elif msg_type == "system":
        role = "system"

    content = getattr(message, "content", message)
    return {"role": role, "content": content}
