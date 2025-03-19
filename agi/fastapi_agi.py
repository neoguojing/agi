import uuid
import base64
from fastapi import FastAPI, Depends, HTTPException, Request,Query,UploadFile,File
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import StreamingResponse,FileResponse
from typing import AsyncGenerator
from typing import List, Union, Dict, Any,Optional
from pydantic import BaseModel, Field
import openai
import json
import time
import os
from langchain_core.messages import HumanMessage,BaseMessage
from agi.tasks.task_factory import TaskFactory, TASK_EMBEDDING
from fastapi.middleware.cors import CORSMiddleware
# 假设的 AgiGraph 模块（需要根据实际情况调整）
from agi.tasks.graph import AgiGraph, State
from agi.fast_api_file import router_file
from agi.config import CACHE_DIR
from pydub import AudioSegment
import logging

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

# 初始化 FastAPI 应用
app = FastAPI(
    title="AGI API",
    description="兼容 OpenAI API 的 AGI 接口",
    version="1.0.0",
)

app.include_router(router_file)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
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
    db_ids: List[str] = Field(default=None, description="知识库列表")
    need_speech: bool = Field(default=False, description="是否需要语音输出")
    feature: str = Field(default="agent", description="支持的特性：agent,web,rag")
    conversation_id: str = Field(default="", description="会话id")


@app.post("/v1/chat/completions", summary="兼容 OpenAI 的聊天完成接口")
async def chat_completions(
    request: ChatCompletionRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    处理聊天完成请求，支持流式和非流式响应，兼容 OpenAI API。
    """
    
    if request.need_speech and request.stream:
        raise HTTPException(status_code=400, detail="语音输出不支持流式响应")


    internal_messages: List[Union[HumanMessage, Dict[str, Union[str, List[Dict[str, str]]]]]] = []
    input_type = "text"  # 默认输入类型
    # 处理知识库参数
    additional_kwargs = {}
    if request.db_ids is not None:
        additional_kwargs["collection_names"] = json.dumps(request.db_ids)
        
    # 只处理最后一条消息
    # for msg in request.messages:
    msg = request.messages[-1]
    if msg.role == "user":
        if isinstance(msg.content, str):
            internal_messages.append(HumanMessage(content=msg.content,additional_kwargs=additional_kwargs))
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
            internal_messages.append(HumanMessage(content=content,additional_kwargs=additional_kwargs))
        # elif msg.role == "assistant":
        #     internal_messages.append({"role": "assistant", "content": msg.content})
    if request.user is None or request.user == "":
        request.user = "default_tenant"
    state_data = State(
        messages=internal_messages,
        input_type=input_type,
        need_speech=request.need_speech,
        user_id=request.user,
        conversation_id=request.conversation_id,
        feature=request.feature
    )

    # TODO 文章的引用信息如何处理
    if request.stream:
        log.debug(f"request: {request}")
        return StreamingResponse(generate_stream_response(state_data), media_type="text/event-stream")
    else:
        resp = graph.invoke(state_data)
        log.debug(f"request:{request}")
        log.debug(f"response:{resp}")
        return format_non_stream_response(resp)

image_style = 'style="width: 100%; max-height: 100vh;"'
audio_style = "width: 300px; height: 50px;"  # 添加样式
def handle_response_content_as_string(content: Union[str,List]) -> str:
    if isinstance(content,str):
        return content
    elif isinstance(content,List):
        ret = ""
        for item in content:
            if item.get("type") == "text":
                ret =  item.get("text")
            elif item.get("type") == "image":
                image_source =  item.get("image")
                ret = f'<img src="{image_source}" {image_style}>\n'
                
            elif item.get("type") == "audio":
                audio_source_base64=  item.get("audio")
                ret = f'<audio src="{audio_source_base64}" {audio_style} controls></audio>\n'

        return ret
    return ""

# 格式化非流式响应
# web参数，控制返回值为string，适配openwebui
def format_non_stream_response(resp: Dict[str, Any],web: bool = False) -> Dict[str, Any]:
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
        if web:
            assistant_content = handle_response_content_as_string(assistant_content)
        
        # 处理additional_kwargs信息
        additional_kwargs = last_message.additional_kwargs
        if additional_kwargs is not None and additional_kwargs.get("citations"):
            assistant_content = {"type":"text","text":assistant_content,"citations":additional_kwargs.get("citations")}

        # 处理metadata信息
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
# web参数，控制返回值为string，适配openwebui 废弃
async def generate_stream_response(state_data: State,web: bool= False) -> AsyncGenerator[str, None]:
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
                
                if web:
                    event.content = handle_response_content_as_string(event.content)
                
                # 处理additional_kwargs信息
                additional_kwargs = event.additional_kwargs
                if additional_kwargs is not None and additional_kwargs.get("citations"):
                    if isinstance(event.content,str):
                        event.content = {"type":"text","text":event.content,"citations":additional_kwargs.get("citations")}
                    else:
                        event.content = {"type":"text","text":event.content.get("text"),"citations":additional_kwargs.get("citations")}
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
        log.debug(f"Error in generate_stream_response: {e}")

class Model(BaseModel):
    id: str                # 模型的唯一标识符，例如 "gpt-3.5-turbo"
    object: str            # 固定为 "model"
    created: int           # 模型创建时间的 UNIX 时间戳
    owned_by: str          # 模型的所有者，例如 "openai"

# 定义模型列表的响应结构
class ModelListResponse(BaseModel):
    object: str            # 固定为 "list"
    data: List[Model]      # 模型对象数组


@app.get("/v1/models", response_model=ModelListResponse, summary="模型列表接口")
async def list_models(api_key: str = Depends(verify_api_key)):
    # 构造固定的模型列表
    fixed_models = [
        Model(
            id="agi",
            object="model",
            created=1677654321,  # 示例时间戳，可根据需要调整
            owned_by="agi"       # 示例所有者，可根据需要调整
        )
    ]
    # 返回符合 ModelListResponse 结构的响应
    return ModelListResponse(
        object="list",
        data=fixed_models
    )

class TranscriptionResponse(BaseModel):
    text: str
   

async def convert_to_base64(file: UploadFile = File(...)):
    try:
        # 异步读取文件内容为字节数据
        audio_bytes = await file.read()
        
        # 将字节数据编码为 Base64
        base64_encoded = base64.b64encode(audio_bytes)
        
        # 将 Base64 字节转换为字符串（UTF-8 解码）
        base64_string = base64_encoded.decode("utf-8")
        
        # 返回 Base64 编码结果
        return base64_string
    except Exception as e:
        log.debug(e)
        return ""

MAX_FILE_SIZE_MB = 25
MAX_FILE_SIZE = MAX_FILE_SIZE_MB * 1024 * 1024
def compress_audio(file_path):
    if os.path.getsize(file_path) > MAX_FILE_SIZE:
        file_dir = os.path.dirname(file_path)
        audio = AudioSegment.from_file(file_path)
        audio = audio.set_frame_rate(16000).set_channels(1)  # Compress audio
        compressed_path = f"{file_dir}/{id}_compressed.opus"
        audio.export(compressed_path, format="opus", bitrate="32k")
        log.debug(f"Compressed audio to {compressed_path}")

        if (
            os.path.getsize(compressed_path) > MAX_FILE_SIZE
        ):  # Still larger than MAX_FILE_SIZE after compression
            raise Exception(f"file size greater than {MAX_FILE_SIZE_MB}MB")
        return compressed_path
    else:
        return file_path
    
@app.post("/v1/audio/transcriptions", summary="语音转文本")
async def create_transcription(file: UploadFile, api_key: str = Depends(verify_api_key)):
    ext = file.filename.split(".")[-1]
    id = uuid.uuid4()

    filename = f"{id}.{ext}"
    contents = file.file.read()

    file_dir = f"{CACHE_DIR}/upload/audio"
    os.makedirs(file_dir, exist_ok=True)
    file_path = f"{file_dir}/{filename}"

    with open(file_path, "wb") as f:
        f.write(contents)

    
    try:
        file_path = compress_audio(file_path)
    except Exception as e:
        log.exception(e)

        raise HTTPException(
            status_code=400,
            detail=str(e),
        )

    try:
        internal_messages: List[Union[HumanMessage, Dict[str, Union[str, List[Dict[str, str]]]]]] = []
        input_type = "audio"  # 默认输入类型
        content: List[Dict[str, str]] = []
        # base64_audio = await convert_to_base64(file)
        # content.append({"type": "audio", "audio": base64_audio})
        content.append({"type": "audio", "audio": file_path})
        internal_messages.append(HumanMessage(content=content))

            
        state_data = State(
            messages=internal_messages,
            input_type=input_type,
            need_speech=False,
            user_id="transcriptions",
            conversation_id="",
            feature="speech"
        )

        resp = graph.invoke(state_data)
        last_message = resp.get("messages")
        assistant_content = ""
        if last_message is not None:
            last_message = resp["messages"][-1]
            assistant_content = last_message.content
        return {"text": assistant_content}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class SpeechRequest(BaseModel):
    model: Optional[str] = "whisper-1"
    input: Optional[str] = None
    voice: Optional[str] = None
    response_format: Optional[str] = "wav"
    speed: Optional[float] = 0.0
    
@app.post("/v1/audio/speech",summary="文本转语音")
async def generate_speech(request: SpeechRequest, api_key: str = Depends(verify_api_key)):
    """
    接收文本并生成语音文件。
    """
    try:
        internal_messages: List[Union[HumanMessage, Dict[str, Union[str, List[Dict[str, str]]]]]] = []
        input_type = "text"  # 默认输入类型
        internal_messages.append(HumanMessage(content=request.input))

            
        state_data = State(
            messages=internal_messages,
            input_type=input_type,
            need_speech=True,
            user_id="speech",
            conversation_id="",
            feature="tts"
        )

        resp = graph.invoke(state_data)
        
        last_message = resp.get("messages")
        file_path = ""
        if last_message is not None:
            last_message = last_message[-1]
            if isinstance(last_message.content[0],dict):
                file_path = last_message.content[0].get("file_path","")
        if not file_path or not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")
        
        return FileResponse(file_path, media_type=f"audio/{request.response_format}", filename=file_path)
    
    except Exception as e:
        log.error(e)
        raise HTTPException(status_code=500, detail=str(e))

# 定义请求体模型
class EmbeddingRequest(BaseModel):
    input: str

@app.post("/v1/embeddings",summary="文本向量")
async def get_embedding(request: EmbeddingRequest):
    if not request.input:
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
    # 生成嵌入向量
    llm_task = TaskFactory.create_task(TASK_EMBEDDING)
    embedding = llm_task.embed_query(request.input)
    
    return {
        "object": "list",
        "data": [
            {
            "object": "embedding",
            "embedding": embedding,
            "index": 0
            }
        ],
        "model": "bge-m3",
        "usage": {
            "prompt_tokens": 0,
            "total_tokens": 0
        }
    }

# 启动服务
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)