import base64
import httpx
import mimetypes
import os
from typing import Callable, Awaitable, List, Dict, Any
from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain_core.messages import BaseMessage,HumanMessage


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

        # 检查最后一条消息是否 HumanMessage
        if not request.messages or request.messages[-1].type != "human":
            # 不是 HumanMessage，直接调用 handler
            return await handler(request)

        # 仅处理最后一条消息
        last_msg = request.messages[-1]
        new_content = []
        for item in last_msg.content:
            if not isinstance(item, dict):
                new_content.append(item)
                continue

            c_type = item.get("type")
            # 仅处理 image 和 audio
            if c_type in ["image", "audio"]:
                source = item.get("file_id") or item.get("url")
                if source and not item.get("base64"):
                    try:
                        b64_data, mime = await self._get_base64_content(source)
                        encoded_item = item.copy()
                        encoded_item["base64"] = b64_data
                        encoded_item["mime_type"] = item.get("mime_type") or mime
                        encoded_item.pop("url", None)
                        new_content.append(encoded_item)
                        continue
                    except Exception as e:
                        print(f"❌ [Base64 Middleware Error]: {str(e)}")

            # 其他类型原样保留
            new_content.append(item)

        msg_dict = last_msg.model_dump()
        msg_dict["content"] = new_content
        new_last_msg = type(last_msg)(**msg_dict)

        # 替换最后一条消息，其它消息保持不变
        new_messages = request.messages[:-1] + [new_last_msg]
        print(f"****************{new_messages}")
        return await handler(request.override(messages=new_messages))