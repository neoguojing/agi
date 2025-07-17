from agi.config import API_KEY,BASE_URL,CACHE_DIR
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi import Depends, HTTPException
from pydantic import BaseModel,Field
from typing import List, Union, Literal, Optional,Dict,Any
# 认证配置
security = HTTPBearer()

# 认证函数
async def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return credentials.credentials


class SpeechRequest(BaseModel):
    model: Optional[str] = "tts"
    input: Optional[str] = None
    voice: Optional[str] = None
    response_format: Optional[str] = "wav"
    speed: Optional[float] = 0.0
    user: str = Field(default="", description="用户名" , optional=True)
    stream: bool = Field(default=False, description="是否使用流式响应" , optional=True)



class ImageURL(BaseModel):
    url: str
    detail: Optional[Literal["low", "auto", "high"]] = "auto"

class MessageContent(BaseModel):
    type: Literal["text", "image_url","image","video","audio"]
    text: Optional[str] = None
    image_url: Optional[ImageURL] = None
    image: Optional[str] = None
    video: Optional[str] = None
    audio: Optional[str] = None


# 兼容 OpenAI 的消息格式
class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"] = Field(description="消息角色，例如 'user' 或 'assistant'")
    content: Union[str, List[MessageContent]] = Field(description="消息内容，可以是文本或多模态数据")

# 兼容 OpenAI 的请求格式
class ChatCompletionRequest(BaseModel):
    model: str = Field(default="agi", description="模型名称" , optional=True)
    messages: List[ChatMessage] = Field(description="对话历史")
    stream: Optional[bool] = Field(default=False, description="是否使用流式响应" , optional=True)
    max_tokens: Optional[int] = Field(default=1024, ge=1, description="最大生成 token 数", optional=True)
    user: Optional[str] = Field(default="", description="用户名" , optional=True)
    db_ids: Optional[List[str]] = Field(default=None, description="知识库列表", optional=True)
    need_speech: Optional[bool] = Field(default=False, description="是否需要语音输出", optional=True)
    feature: Optional[str] = Field(default="", description="支持的特性：agent,web,rag", optional=True)
    conversation_id: Optional[str] = Field(default="", description="会话id" , optional=True)

# ======== 响应格式 ========

class ChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    choices: List[dict]