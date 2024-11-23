import torch
from langchain.llms.base import LLM
from pydantic import Field
from typing import Any, Union, Literal, List, Dict
from langchain_core.runnables import Runnable, RunnableSerializable, RunnableConfig
from pydantic import BaseModel, HttpUrl, constr, ConfigDict
from langchain_core.messages import AIMessage, HumanMessage
import base64
import requests
from typing import Optional
from PIL import Image as PILImage
from io import BytesIO
import numpy as np
from diffusers.utils import load_image
from enum import Enum
import os
import re
from scipy.io.wavfile import write

# Enum for Image Types
class ImageType(Enum):
    URL = "URL"  # Represents a URL type
    FILE_PATH = "FILE_PATH"  # Represents a file path type
    BASE64 = "BASE64"  # Represents a Base64 encoded type
    PIL_IMAGE = "PIL_IMAGE"  # Represents a PIL image object

# Enum for Audio Types
class AudioType(Enum):
    URL = "URL"  # Represents URL for audio
    FILE_PATH = "FILE_PATH"  # Represents file path for audio
    BYTE_IO = "BYTE_IO"  # Represents byte data for audio
    NUMPY = "NUMPY"  # Represents NumPy array for audio

# Multi-modal message type
MultiModalMessageType = Union[ImageType, AudioType]

# Image class for handling different image types
class Image(BaseModel):
    url: Optional[str] = None  # Image URL
    pil_image: Optional[PILImage.Image] = None  # PIL Image object
    filename: Optional[str] = None  # File name
    filetype: Optional[str] = None  # File type (e.g., 'image/jpeg', 'image/png')
    size: Optional[int] = None  # File size (in bytes)
    file_path: Optional[str] = None  # File path on disk
    media_type: Optional[ImageType] = None
    model_config = ConfigDict(arbitrary_types_allowed=True) 

    @classmethod
    def new(cls, input: Any, type: ImageType):
        """Creates an Image instance from local file or other formats."""
        pil_image = None
        if type == ImageType.URL or type == ImageType.FILE_PATH:
            pil_image = load_image(input)
            filename = input.split('/')[-1]
        elif type == ImageType.BASE64:
            # Decode Base64 string to bytes and load as PIL image
            image_data = base64.b64decode(input)
            image_bytes = BytesIO(image_data)
            pil_image = PILImage.Image.open(image_bytes)
        elif type == ImageType.PIL_IMAGE:
            pil_image = input

        # Extract file info
        filetype = pil_image.format  # Extract file format
        size = pil_image.tobytes().__sizeof__()  # Calculate image size

        instance = cls()
        instance.pil_image = pil_image
        instance.filename = filename
        instance.filetype = filetype
        instance.size = size
        instance.media_type = type
        return instance

    def save_image(self, output_path: str):
        """Save PIL Image object to a file."""
        if self.pil_image:
            self.pil_image.save(output_path)


def is_url(input_data):
    """判断是否是URL"""
    url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    return isinstance(input_data, str) and bool(url_pattern.match(input_data))

def is_file_path(input_data):
    """判断是否是文件路径"""
    return isinstance(input_data, str) and os.path.isfile(input_data)

def is_base64(input_data):
    """判断是否是Base64编码的字符串"""
    if isinstance(input_data, str):
        try:
            # 尝试解码Base64字符串
            base64.b64decode(input_data, validate=True)
            return True
        except (ValueError, TypeError):
            return False
    return False

def is_pil_image(input_data):
    """判断是否是PIL Image对象"""
    return isinstance(input_data, PILImage.Image)

def convert_to_pil_image(input_data) -> PILImage.Image:
    """根据输入数据判断图片类型"""
    if is_pil_image(input_data):
        return input_data
    elif is_url(input_data):
        img = Image.new(input_data,ImageType.URL)
        return img.pil_image
    elif is_file_path(input_data):
        img = Image.new(input_data,ImageType.FILE_PATH)
        return img.pil_image
    elif is_base64(input_data):
        img = Image.new(input_data,ImageType.BASE64)
        return img.pil_image
    else:
        return None



    
# Function to build a multi-modal message
def build_multi_modal_message(text_input: str, media_data: any) -> HumanMessage:
    return HumanMessage(content=[
        {"type": "text", "text": text_input},
        {"type": "media", "media": media_data},
    ])


# Audio class for handling different audio types
class Audio(BaseModel):
    url: Optional[HttpUrl] = None  # Audio URL
    file_path: Optional[str] = None  # File path to audio
    samples: Optional[BytesIO] = None  # Audio sample data in BytesIO
    filename: Optional[str] = None  # File name
    filetype: Optional[str] = None  # File type (e.g., 'audio/mpeg', 'audio/wav')
    size: Optional[int] = None  # File size (in bytes)
    media_type: Optional[AudioType] = None
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    def from_local(cls, audio_path: str):
        """Create an Audio instance from a local file."""
        with open(audio_path, "rb") as audio_file:
            binary_data = audio_file.read()
            samples = BytesIO(binary_data)  # Store binary data in BytesIO
            filename = os.path.basename(audio_path)
            filetype = filename.split('.')[-1]  # Extract file extension
            size = len(binary_data)

        return cls(samples=samples, file_path=audio_path, filename=filename, filetype=filetype, size=size,media_type = AudioType.BYTE_IO)

    @classmethod
    def from_url(cls, url: HttpUrl):
        """Create an Audio instance from a URL."""
        response = requests.get(url)
        if response.status_code == 200:
            binary_data = response.content
            samples = BytesIO(binary_data)  # Store binary data in BytesIO
            filename = os.path.basename(url)
            filetype = filename.split('.')[-1]  # Extract file extension
            size = len(binary_data)
            return cls(url=url, samples=samples, filename=filename, filetype=filetype, size=size,media_type = AudioType.BYTE_IO)
        else:
            raise Exception(f"Failed to download audio: {response.status_code}")

    @classmethod
    def from_numpy(cls, numpy_audio_data: np.ndarray,sample_rate: int):
        try:
            # 创建一个 BytesIO 流
            byte_io = BytesIO()

            # 将 NumPy 数组保存为 WAV 格式，写入到 BytesIO 对象中
            write(byte_io, sample_rate, numpy_audio_data)

            # 将指针移动到流的开头，以便后续读取
            byte_io.seek(0)

            return cls(samples=byte_io, media_type = AudioType.NUMPY)
        except Exception as e:
            print(f"转换过程发生错误：{e}")
            return None

    
    def to_binary(self) -> bytes:
        """Convert the audio sample data to binary format."""
        return self.samples.getvalue()  # Get binary data from BytesIO

def is_numpy_array(input_data):
    """判断是否是NumPy数组"""
    return isinstance(input_data, np.ndarray)

def is_bytesio(input_data):
    """判断是否是BytesIO对象"""
    return isinstance(input_data, BytesIO)

def convert_audio_to_byteio(input_data,sample_rate=None) -> BytesIO:
    """根据输入数据判断音频类型"""
    if is_url(input_data):
        audio = Audio.from_url(input_data)
        return audio.samples
    elif is_file_path(input_data):
        audio = Audio.from_local(input_data)
        return audio.samples
    elif is_numpy_array(input_data):
        audio = Audio.from_numpy(input_data)
        return audio.samples
    elif is_bytesio(input_data):
        return input_data
    else:
        return None
    

# Custom LLM class for integration with runnable modules
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
            print(f"Model {self.model_name} destroyed successfully")

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
