from __future__ import annotations

from typing import Any, Literal, Union

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableSerializable

from agi.config import log

InputType = Literal["text", "image", "audio", "video"]


def _extract_media_from_block(content: dict[str, Any]) -> tuple[str | None, InputType]:
    """Extract media payload from one message block.

    Supports both legacy internal format and OpenAI-compatible formats:
    - {"type": "image", "image": "..."}
    - {"type": "audio", "audio": "..."}
    - {"type": "video", "video": "..."}
    - {"type": "image_url", "image_url": {"url": "..."}}
    - {"type": "input_audio", "input_audio": {"data": "..."}}
    """
    block_type = content.get("type")

    if block_type == "image":
        media = content.get("image")
        return media, "image"
    if block_type == "audio":
        media = content.get("audio")
        return media, "audio"
    if block_type == "video":
        media = content.get("video")
        return media, "video"

    # OpenAI-compatible blocks
    if block_type == "image_url":
        media = (content.get("image_url") or {}).get("url")
        return media, "image"
    if block_type == "input_audio":
        media = (content.get("input_audio") or {}).get("data")
        return media, "audio"

    return None, "text"


# 从用户消息中抽取content的内容，转换为模型可处理的格式
def parse_input_messages(
    input: Union[HumanMessage, list[HumanMessage]],
) -> tuple[str | None, str | None, InputType]:
    """Parse message content and extract (media, prompt, input_type)."""
    media: str | None = None
    prompt: str | None = None
    input_type: InputType = "text"

    if isinstance(input, list):
        input = input[-1]

    if isinstance(input.content, list):
        for content in input.content:
            if not isinstance(content, dict):
                continue

            extracted_media, extracted_type = _extract_media_from_block(content)
            if extracted_media:
                media = extracted_media
                input_type = extracted_type

            if content.get("type") == "text":
                prompt = content.get("text")
    elif isinstance(input.content, str):
        prompt = input.content

    return media, prompt, input_type


# Custom LLM class for integration with runnable modules
class CustomerLLM(RunnableSerializable[HumanMessage, AIMessage]):
    model: Any = None
    tokenizer: Any = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = None

    def destroy(self):
        if self.model is not None:
            del self.model
        log.info(f"Model {self.model_name} destroyed successfully")

    def encode(self, input):
        if self.tokenizer is not None:
            return self.tokenizer.encode(input)
        return None

    def decode(self, ids):
        if self.tokenizer is not None:
            return self.tokenizer.decode(ids)
        return ""

    @property
    def model_name(self) -> str:
        return ""  # Model name placeholder (to be customized)
