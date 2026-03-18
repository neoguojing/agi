from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from agi.apps.common import ChatCompletionRequest, verify_api_key
from agi.agent.agent import invoke_agent
from .stream import stream_response
from .formatter import format_response
from .media import process_multimodal_content
from langchain_core.messages import HumanMessage
from typing import Dict, Any

router = APIRouter()

def build_state(request: ChatCompletionRequest) -> Dict[str, Any]:
    msg = request.messages[-1]

    if msg.role != "user":
        raise HTTPException(400, "Last message must be user")

    # 文本
    if isinstance(msg.content, str):
        messages = [HumanMessage(content=msg.content)]
        input_type = "text"

    # 多模态
    else:
        content, input_type = process_multimodal_content(msg.content)
        messages = [HumanMessage(content=content)]

    return {
        "messages": messages,
        "input_type": input_type,
        "need_speech": request.need_speech,
        "user_id": request.user or "default",
        "conversation_id": request.conversation_id,
        "feature": request.feature,
        "collection_names": request.db_ids,
    }


@router.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest, api_key: str = Depends(verify_api_key)):
    state = build_state(request)
    if request.stream:
        return StreamingResponse(stream_response(state), media_type="text/event-stream")

    resp = await invoke_agent(state)
    return format_response(resp)