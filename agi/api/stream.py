import json
import uuid
import time
import traceback
from typing import AsyncGenerator
from agi.agent.agent import stream_agent
from langchain_core.messages import BaseMessage
from agi.config import log
from typing import Dict,Any

async def stream_response(state: Dict[str, Any]):
    index = 0

    async for part in stream_agent(state):

        chunk = {
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": "agi-model",
            "choices": [{
                "index": index,
                "delta": {},
                "finish_reason": None,
            }],
        }

        # =========================
        # LLM token
        # =========================
        if part["type"] == "messages":
            msg, meta = part["data"]

            if msg.content:
                chunk["choices"][0]["delta"] = {
                    "role": "assistant",
                    "content": msg.content
                }

                finish = meta.get("finish_reason")
                if finish:
                    chunk["choices"][0]["finish_reason"] = finish

        # =========================
        # Tool / 状态更新
        # =========================
        elif part["type"] == "updates":
            for node, data in part["data"].items():
                interrupt = data.get("__interrupt__")
                if interrupt:
                    chunk["choices"][0]["delta"] = {
                        "role": "assistant",
                        "content": interrupt[0].value
                    }
                    chunk["choices"][0]["finish_reason"] = "stop"

        # =========================
        # 自定义（citation / 进度）
        # =========================
        elif part["type"] == "custom":
            data = part["data"]
            if data.get("citations"):
                chunk["choices"][0]["delta"] = {
                    "role": "assistant",
                    "content": [{"citations": data["citations"]}]
                }
            else:
                continue

        else:
            continue

        yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
        index += 1

    yield "data: [DONE]\n\n"