import base64
import httpx
import mimetypes
import os
from typing import Callable, Awaitable, List, Dict, Any
from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.messages.content import (
    create_text_block,
    create_image_block,
    create_audio_block,
)

import logging
logger = logging.getLogger(__name__)

class MultimodalBase64Middleware(AgentMiddleware):
    """
    中间件：仅当最后一条消息是 HumanMessage 时，将消息中的 image/audio 转换为 Base64。
    """

    async def _get_base64_content(self, source: str) -> tuple[str, str]:
        if source.startswith(("http://", "https://")):
            async with httpx.AsyncClient() as client:
                response = await client.get(source)
                response.raise_for_status()
                content = response.content
                mime_type = response.headers.get("Content-Type") or mimetypes.guess_type(source)[0]
        else:
            if not os.path.exists(source):
                raise FileNotFoundError(f"Missing file at path: {source}")
            with open(source, "rb") as f:
                content = f.read()
            mime_type = mimetypes.guess_type(source)[0]

        b64_str = base64.b64encode(content).decode("utf-8")
        return b64_str, mime_type or "application/octet-stream"

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        try:
            if not request.messages:
                return await handler(request)

            last_is_human = isinstance(request.messages[-1], HumanMessage)
            new_messages = []

            for idx, msg in enumerate(request.messages):
                if not isinstance(msg, HumanMessage):
                    new_messages.append(msg)
                    continue

                is_last_msg = idx == len(request.messages) - 1
                should_encode = last_is_human and is_last_msg
                content_blocks = []

                if isinstance(msg.content, list):
                    for item in msg.content:
                        if not isinstance(item, dict):
                            continue
                        c_type = item.get("type")
                        if c_type == "text":
                            content_blocks.append(create_text_block(text=item.get("text", "")))
                        elif c_type == "image":
                            mime = item.get("mime_type", "image/png")
                            if should_encode:
                                base64_data = item.get("base64")
                                if not base64_data:
                                    source = item.get("file_id") or item.get("url")
                                    if source:
                                        base64_data, mime = await self._get_base64_content(source)
                                if base64_data:
                                    content_blocks.append(create_image_block(base64=base64_data, mime_type=mime))
                            else:
                                content_blocks.append(
                                    create_text_block(text=f"[image omitted: {item.get('file_id') or item.get('url')}]")
                                )
                        elif c_type == "audio":
                            mime = item.get("mime_type", "audio/wav")
                            if should_encode:
                                base64_data = item.get("base64")
                                if not base64_data:
                                    source = item.get("file_id") or item.get("url")
                                    if source:
                                        base64_data, mime = await self._get_base64_content(source)
                                if base64_data:
                                    content_blocks.append(create_audio_block(base64=base64_data, mime_type=mime))
                            else:
                                content_blocks.append(
                                    create_text_block(text=f"[audio omitted: {item.get('file_id') or item.get('url')}]")
                                )

                new_messages.append(
                    HumanMessage(
                        content=msg.content if isinstance(msg.content, str) else content_blocks,
                    )
                )

            return await handler(request.override(messages=new_messages))
        except Exception as e:
            logger.exception("Multimodal middleware 发生未预期异常")
            return ModelResponse(
                result=[AIMessage(content=f"Multimodal middleware failed: {type(e).__name__}: {e}")]
            )
