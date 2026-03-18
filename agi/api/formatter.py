import uuid
import time
from typing import Dict, Any
from langchain_core.messages import BaseMessage

def format_response(resp: Dict[str, Any]) -> Dict[str, Any]:
    msg: BaseMessage = resp.get("messages", [])[-1]
    content = getattr(msg, "content", "")
    metadata = getattr(msg, "response_metadata", {}) or {}
    usage = metadata.get("token_usage", {})

    return {
        "id": f"chatcmpl-{uuid.uuid4()}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "agi-model",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": metadata.get("finish_reason", "stop"),
        }],
        "usage": {
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
        },
    }