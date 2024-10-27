import torch
from langchain.llms.base import LLM
from pydantic import  Field
from typing import Any,Union,Literal,List,Dict
from langchain_core.runnables import Runnable, RunnableSerializable,RunnableConfig
from langchain_core.messages.base import BaseMessage
from pydantic import BaseModel, HttpUrl, constr,ConfigDict

import base64
import requests
from typing import Optional
from PIL import Image as PILImage
from io import BytesIO
from typing import Optional, List
import requests
from diffusers.utils import load_image

class Image(BaseModel):
    url: Optional[str] = None  # 图片的 URL
    pil_image: Optional[PILImage.Image] = None  # 使用 PIL 图像对象
    filename: Optional[str] = None  # 文件名
    filetype: Optional[str] = None  # 文件类型 (如 'image/jpeg', 'image/png')
    size: Optional[int] = None  # 文件大小（字节）

    model_config = ConfigDict(arbitrary_types_allowed=True)
    @classmethod
    def new(cls, url_or_path: str):
        """从本地文件创建 Image 实例"""
        pil_image = load_image(url_or_path)
        filename = url_or_path.split('/')[-1]
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

    def pretty_repr(self) -> List[str]:
        """返回图片的美观表示"""
        lines = [
            f"URL: {self.url}" if self.url else "URL: None",
            f"Filename: {self.filename}" if self.filename else "Filename: None",
            f"Filetype: {self.filetype}" if self.filetype else "Filetype: None",
            f"Size: {self.size} bytes" if self.size is not None else "Size: None"
        ]
        return lines


from pydantic import BaseModel, HttpUrl
import requests
import numpy as np
from typing import List, Optional

class Audio(BaseModel):
    url: Optional[HttpUrl] = None                  # 音频的 URL
    samples: Optional[List[int]] = None             # 音频的样本数据
    filename: Optional[str] = None                  # 文件名
    filetype: Optional[str] = None                  # 文件类型 (如 'audio/mpeg', 'audio/wav')
    size: Optional[int] = None                      # 文件大小（字节）

    @classmethod
    def from_local(cls, audio_path: str):
        """从本地文件创建 Audio 实例"""
        with open(audio_path, "rb") as audio_file:
            binary_data = audio_file.read()
            # 假设音频是 16-bit PCM
            samples = np.frombuffer(binary_data, dtype=np.int16).tolist()
            filename = audio_path.split('/')[-1]
            filetype = filename.split('.')[-1]  # 简单提取文件扩展名
            size = len(binary_data)
        
        return cls(samples=samples, filename=filename, filetype=filetype, size=size)

    @classmethod
    def from_url(cls, url: HttpUrl):
        """从 URL 下载音频并创建 Audio 实例"""
        response = requests.get(url)
        if response.status_code == 200:
            binary_data = response.content
            # 假设音频是 16-bit PCM
            samples = np.frombuffer(binary_data, dtype=np.int16).tolist()
            filename = url.split('/')[-1]
            filetype = filename.split('.')[-1]  # 简单提取文件扩展名
            size = len(binary_data)
            return cls(url=url, samples=samples, filename=filename, filetype=filetype, size=size)
        else:
            raise Exception(f"Failed to download audio: {response.status_code}")

    def to_binary(self) -> bytes:
        """将样本数据转换回二进制格式"""
        return np.array(self.samples, dtype=np.int16).tobytes()

    def pretty_repr(self, html: bool = False) -> List[str]:
        """返回音频的美观表示。

        Args:
            html: 是否返回 HTML 格式的字符串。
                  默认值为 False。

        Returns:
            音频的美观表示。
        """
        lines = [
            f"URL: {self.url}" if self.url else "URL: None",
            f"Filename: {self.filename}" if self.filename else "Filename: None",
            f"Filetype: {self.filetype}" if self.filetype else "Filetype: None",
            f"Size: {self.size} bytes" if self.size is not None else "Size: None"
        ]

        return lines
    
class MultiModalMessage(BaseMessage):
    image: Image = None
    audio: Audio = None
    """The type of the message (used for deserialization). Defaults to "ai"."""
    
    type: Literal["agi"] = "agi"
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
        
    def __init__(
        self, content: Union[str, list[Union[str, dict]]],image: Image =None,audio: Audio = None, **kwargs: Any
    ) -> None:
        """Pass in content as positional arg.

        Args:
            content: The content of the message.
            kwargs: Additional arguments to pass to the parent class.
        """
        super().__init__(content=content, **kwargs)
        self.audio = audio
        self.image = image

    @classmethod
    def get_lc_namespace(cls) -> list[str]:
        """Get the namespace of the langchain object.

        Returns:
            The namespace of the langchain object.
            Defaults to ["langchain", "schema", "messages"].
        """
        return ["langchain", "schema", "messages"]

    @property
    def lc_attributes(self) -> dict:
        """Attrs to be serialized even if they are derived from other init args."""
        return {
            "image": self.image,
            "audio": self.audio,
        }

    def pretty_repr(self, html: bool = False) -> str:
        """Return a pretty representation of the message.

        Args:
            html: Whether to return an HTML-formatted string.
                 Defaults to False.

        Returns:
            A pretty representation of the message.
        """
        base = super().pretty_repr(html=html)
        lines = self.image.pretty_repr()
        lines.extend(self.audio.pretty_repr())
        
        return (base.strip() + "\n" + "\n".join(lines)).strip()


class CustomerLLM(RunnableSerializable[BaseMessage,BaseMessage]):
    device: str = Field(torch.device('cpu'))
    model: Any = None
    tokenizer: Any = None

    def __init__(self,llm,**kwargs):
        super(CustomerLLM, self).__init__()
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device('cpu')
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
