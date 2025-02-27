import uuid
from fastapi import FastAPI, Depends, HTTPException, Request,Query
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import StreamingResponse
from typing import AsyncGenerator
from typing import List, Union, Dict, Any
from pydantic import BaseModel, Field
import json
import time
from langchain_core.messages import HumanMessage,BaseMessage

# 假设的 AgiGraph 模块（需要根据实际情况调整）
from agi.tasks.graph import AgiGraph, State

# 初始化 FastAPI 应用
app = FastAPI(
    title="AGI API",
    description="兼容 OpenAI API 的 AGI 接口",
    version="1.0.0",
)

# 实例化 AgiGraph（假设的外部模块）
graph = AgiGraph()

# 认证配置
security = HTTPBearer()
API_KEY = "123"  # 请替换为实际的 API 密钥

# 认证函数
async def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return credentials.credentials

# 兼容 OpenAI 的消息格式
class ChatMessage(BaseModel):
    role: str = Field(description="消息角色，例如 'user' 或 'assistant'")
    content: Union[str, List[Dict[str, Any]]] = Field(description="消息内容，可以是文本或多模态数据")

# 兼容 OpenAI 的请求格式
class ChatCompletionRequest(BaseModel):
    model: str = Field(default="agi-model", description="模型名称")
    messages: List[ChatMessage] = Field(description="对话历史")
    stream: bool = Field(default=False, description="是否使用流式响应")
    max_tokens: int = Field(default=1024, ge=1, description="最大生成 token 数", optional=True)
    user: str = Field(default="", description="用户名")


@app.post("/v1/chat/completions", summary="兼容 OpenAI 的聊天完成接口")
async def chat_completions(
    request: ChatCompletionRequest,
    # http_request: Request,  # 添加 Request 对象
    need_speech: bool = Query(default=False, description="是否需要语音输出"),
    conversation_id: str = Query(default="", description="会话id"),
    api_key: str = Depends(verify_api_key)
):
    """
    处理聊天完成请求，支持流式和非流式响应，兼容 OpenAI API。
    """
    # 获取额外参数
    # extra_query = dict(http_request.query_params)
    # print("extra_query:", extra_query)
    # need_speech = extra_query.get("need_speech",False)
    # if isinstance(need_speech, str):
    #     need_speech = need_speech.lower() == "true"  # 转换为布尔值
    
    print("need_speech:", need_speech)
        
    if need_speech and request.stream:
        raise HTTPException(status_code=400, detail="语音输出不支持流式响应")


    internal_messages: List[Union[HumanMessage, Dict[str, Union[str, List[Dict[str, str]]]]]] = []
    input_type = "text"  # 默认输入类型
    for msg in request.messages:
        if msg.role == "user":
            if isinstance(msg.content, str):
                internal_messages.append(HumanMessage(content=msg.content))
            else:
                content: List[Dict[str, str]] = []
                for item in msg.content:
                    if item["type"] == "image":
                        # 假设 item["image"] 是图像数据的某种表示（例如，文件路径或 base64 编码）
                        content.append({"type": "image", "image": item["image"]})
                        input_type = "image"
                    elif item["type"] == "audio":
                        # 假设 item["audio"] 是音频数据的某种表示
                        content.append({"type": "audio", "audio": item["audio"]})
                        input_type = "audio"
                    elif item["type"] == "text": 
                        content.append({"type":"text","text":item["text"]})
                    else:
                        # 处理不支持的类型
                        raise ValueError(f"不支持的多模态类型: {item['type']}")
                internal_messages.append(HumanMessage(content=content))
        elif msg.role == "assistant":
            internal_messages.append({"role": "assistant", "content": msg.content})
            
        
    state_data = State(
        messages=internal_messages,
        input_type=input_type,
        need_speech=need_speech,
        user_id=request.user,
        conversation_id=conversation_id
    )

    if request.stream:
        return StreamingResponse(generate_stream_response(state_data), media_type="text/event-stream")
    else:
        resp = graph.invoke(state_data)
        return format_non_stream_response(resp)

# 格式化非流式响应
def format_non_stream_response(resp: Dict[str, Any]) -> Dict[str, Any]:
    """
    将内部响应格式化为 OpenAI 兼容的非流式响应。
    """
    last_message = resp.get("messages")
    assistant_content = "No response"
    finish_reason = "stop"
    completion_tokens = 0
    prompt_tokens = 0
    total_tokens = 0
    
    if last_message is not None:
        last_message = resp["messages"][-1]
        assistant_content = last_message.content
        
        response_metadata = last_message.response_metadata
        if response_metadata is not None and "finish_reason" in response_metadata:
            finish_reason = response_metadata["finish_reason"]
        
        # 从 resp 中获取 token_usage
        token_usage = response_metadata.get("token_usage")  # 假设在顶级响应中
        if token_usage:
            completion_tokens = token_usage["completion_tokens"]  # 字典访问
            prompt_tokens = token_usage["prompt_tokens"]
            total_tokens = token_usage["total_tokens"]
    
    return {
        "id": f"chatcmpl-{uuid.uuid4()}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "agi-model",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": assistant_content
                },
                "finish_reason": finish_reason
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,    # 可根据实际情况计算
            "completion_tokens": completion_tokens, # 可根据实际情况计算
            "total_tokens": total_tokens      # 可根据实际情况计算
        }
    }
        
async def generate_stream_response(state_data: State) -> AsyncGenerator[str, None]:
    """
    生成 OpenAI 兼容的流式响应，使用 SSE 格式，调用 stream 方法。
    
    Args:
        state_data (Dict[str, Any]): 输入的状态数据，用于生成事件流。
    
    Yields:
        str: SSE 格式的流式响应块，符合 OpenAI API 规范。
    """
    events = graph.stream(state_data)
    index = 0  # 初始化 index
    try:
        for event in events:
            chunk = {
                "id": f"chatcmpl-{uuid.uuid4()}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": "agi-model",
                "choices": [{"index": index, "delta": {}, "finish_reason": None}] # 使用递增的index
            }

            if isinstance(event, BaseMessage):
                role = "user" if event.__class__.__name__ == "HumanMessage" else "assistant"
                # 跳过用户消息
                if role == "user":
                    continue
                
                chunk["choices"][0]["delta"] = {"role": role, "content": event.content}
                finish_reason = getattr(event, "response_metadata", {}).get("finish_reason")
                if finish_reason:
                    chunk["choices"][0]["finish_reason"] = finish_reason
            elif isinstance(event, dict):
                if "error" in event:
                    chunk["choices"] = []
                    chunk["error"] = {"message": event["error"], "type": "server_error"}
                else:
                    content = event.get("content", str(event))
                    chunk["choices"][0]["delta"] = {"content": content}

            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
            index += 1 #index递增

        final_chunk = {
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": "agi-model",
            "choices": [{"index": index, "delta": {}, "finish_reason": "stop"}]
        }
        yield f"data: {json.dumps(final_chunk, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    except Exception as e:
        error_chunk = {
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": "agi-model",
            "choices": [],
            "error": {"message": str(e), "type": "server_error"}
        }
        yield f"data: {json.dumps(error_chunk, ensure_ascii=False)}\n\n"
        # 记录错误日志
        print(f"Error in generate_stream_response: {e}")
# 启动服务
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)