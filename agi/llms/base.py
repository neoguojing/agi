import torch
from langchain.llms.base import LLM
from pydantic import  Field
from typing import Any,Union,Literal,List,Dict
from langchain_core.runnables import Runnable, RunnableSerializable,RunnableConfig
from pydantic import BaseModel, HttpUrl, constr,ConfigDict
from langchain_core.messages import AIMessage, HumanMessage
import base64
import requests
from typing import Optional
from PIL import Image as PILImage
from io import BytesIO
from typing import Optional, List
import requests
from diffusers.utils import load_image
from enum import Enum


class ImageType(Enum):
    URL = "URL"  # 表示 URL 类型
    FILE_PATH = "FILE_PATH"  # 表示文件路径类型
    BASE64 = "BASE64"  # 表示 Base64 编码
    PIL_IMAGE = "PIL_IMAGE"  # 表示 PIL 图片对象
    
class AudioType(Enum):
    URL = "URL"  # 
    FILE_PATH = "FILE_PATH" 
    BYTE_IO = "BYTE_IO" 
    NUMPY = "NUMPY" 
    
class Image(BaseModel):
    url: Optional[str] = None  # 图片的 URL
    pil_image: Optional[PILImage.Image] = None  # 使用 PIL 图像对象
    filename: Optional[str] = None  # 文件名
    filetype: Optional[str] = None  # 文件类型 (如 'image/jpeg', 'image/png')
    size: Optional[int] = None  # 文件大小（字节）
    file_path: Optional[str] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)
    @classmethod
    def new(cls, input,type: ImageType):
        """从本地文件创建 Image 实例"""
        pil_image = None
        if type == ImageType.URL or type == ImageType.FILE_PATH:
            pil_image = load_image(input)
            filename = input.split('/')[-1]
        elif type == ImageType.BASE64:
            # 步骤1：解码 Base64 字符串为字节数据
            image_data = base64.b64decode(input)
            # 步骤2：将字节数据转换为 BytesIO 对象
            image_bytes = BytesIO(image_data)
            # 步骤3：使用 PIL.Image.open() 打开 BytesIO 对象
            pil_image = PILImage.Image.open(image_bytes)
        elif type == ImageType.PIL_IMAGE:
            pil_image = input
            
        filetype = pil_image.format  # 使用 PIL 提取文件格式
        size = pil_image.tobytes().__sizeof__()  # 计算字节大小

        instance = cls()
        instance.pil_image = pil_image
        instance.filename = filename
        instance.filetype = filetype
        instance.size = size
        return instance

    def save_image(self, output_path: str):
        """保存 PIL 图像对象为图片文件"""
        if self.pil_image:
            self.pil_image.save(output_path)

import os
import numpy as np
import requests
from pydantic import BaseModel, HttpUrl
from typing import Optional, List
from io import BytesIO

class Audio(BaseModel):
    url: Optional[HttpUrl] = None  # 音频的 URL
    file_path: Optional[str] = None
    samples: Optional[BytesIO] = None  # 音频的样本数据
    filename: Optional[str] = None  # 文件名
    filetype: Optional[str] = None  # 文件类型 (如 'audio/mpeg', 'audio/wav')
    size: Optional[int] = None  # 文件大小（字节）

    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    @classmethod
    def from_local(cls, audio_path: str):
        """从本地文件创建 Audio 实例"""
        with open(audio_path, "rb") as audio_file:
            binary_data = audio_file.read()
            samples = BytesIO(binary_data)  # 将二进制数据存储在 BytesIO 中
            filename = os.path.basename(audio_path)
            filetype = filename.split('.')[-1]  # 简单提取文件扩展名
            size = len(binary_data)
        
        return cls(samples=samples, file_path=audio_path, filename=filename, filetype=filetype, size=size)

    @classmethod
    def from_url(cls, url: HttpUrl):
        """从 URL 下载音频并创建 Audio 实例"""
        response = requests.get(url)
        if response.status_code == 200:
            binary_data = response.content
            samples = BytesIO(binary_data)  # 将二进制数据存储在 BytesIO 中
            filename = os.path.basename(url)
            filetype = filename.split('.')[-1]  # 简单提取文件扩展名
            size = len(binary_data)
            return cls(url=url, samples=samples, filename=filename, filetype=filetype, size=size)
        else:
            raise Exception(f"Failed to download audio: {response.status_code}")

    def to_binary(self) -> bytes:
        """将样本数据转换回二进制格式"""
        return self.samples.getvalue()  # 从 BytesIO 中获取二进制数据


class CustomerLLM(RunnableSerializable[HumanMessage, AIMessage]):
    device: str = Field(default_factory=lambda: str(torch.device('cpu')))
    model: Any = None
    tokenizer: Any = None

    def __init__(self, llm, **kwargs):
        super(CustomerLLM, self).__init__(**kwargs)
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')
        self.model = llm

    def destroy(self):
        if self.model is not None:
            del self.model
            torch.cuda.empty_cache()
            print(f"model {self.model_name} destroy success")

    def encode(self,input):
        if self.tokenizer is not None:
            return self.tokenizer.encode(input)
        return None
        
    def decode(self,ids):
        if self.tokenizer is not None:
            return self.tokenizer.decode(ids)
        return ""
    
    @property
    def model_name(self) -> str:
        return ""
