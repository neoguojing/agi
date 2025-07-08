import os
import urllib.parse

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


def path_to_preview_url(file_path: str, base_url: str = BASE_URL) -> str:
        """
        将文件路径转换为图片预览 URL。
        
        Args:
            file_path (str): 服务器上的文件路径，例如 "uploads/picture.jpg"
            base_url (str): 服务器基地址，默认 "http://localhost:8000"
        
        Returns:
            str: 可用于预览的 URL，例如 "http://localhost:8000/files/picture.jpg"
        
        Raises:
            ValueError: 如果文件路径不在上传目录内
        """
        # 确保文件路径在 CACHE_DIR 内，防止目录遍历
        if not os.path.realpath(file_path).startswith(os.path.realpath(CACHE_DIR)):
            raise ValueError("File path is outside the upload directory")
        
        # 获取相对于 UPLOAD_DIR 的文件名
        file_name = os.path.basename(file_path)
        
        # 构建预览 URL
        preview_url = f"{base_url}/v1/files/{urllib.parse.quote(file_name)}"
        return preview_url


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
    stream: bool = Field(default=False, description="是否使用流式响应" , optional=True)
    max_tokens: int = Field(default=1024, ge=1, description="最大生成 token 数", optional=True)
    user: str = Field(default="", description="用户名" , optional=True)
    db_ids: List[str] = Field(default=None, description="知识库列表", optional=True)
    need_speech: bool = Field(default=False, description="是否需要语音输出", optional=True)
    feature: str = Field(default="", description="支持的特性：agent,web,rag", optional=True)
    conversation_id: str = Field(default="", description="会话id" , optional=True)

# ======== 响应格式 ========

class ChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    choices: List[dict]