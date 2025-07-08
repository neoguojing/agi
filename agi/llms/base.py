from pydantic import Field
from typing import Any, Union, Literal, List, Dict
from langchain_core.runnables import Runnable, RunnableSerializable, RunnableConfig
from langchain_core.messages import AIMessage, HumanMessage
import os
from agi.config import BASE_URL,CACHE_DIR
import urllib.parse
from agi.config import log

# 从用户消息中抽取content的内容，转换为模型可处理的格式
def parse_input_messages(input: Union[HumanMessage,list[HumanMessage]]):
    """
    Parse the content of the HumanMessage to extract the image and prompt.
    """
    media = None
    prompt = None
    if isinstance(input, list):
        input = input[-1]
        
    if isinstance(input.content, list):
        for content in input.content:
            media_type = content.get("type")
            if media_type == "image":
                # Create Image instance based on media type
                media_data = content.get("image")
                if media_data is not None and media_data != "":
                    media = media_data
            if media_type == "audio":
                # Create Image instance based on media type
                media_data = content.get("audio")
                if media_data is not None and media_data != "":
                    media = media_data
            elif media_type == "text":
                prompt = content.get("text")
    elif isinstance(input.content, str):
        prompt = input.content
    
    return media, prompt

# Custom LLM class for integration with runnable modules
class CustomerLLM(RunnableSerializable[HumanMessage, AIMessage]):
    device: str = Field(default_factory=None)
    model: Any = None
    tokenizer: Any = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.device = None
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