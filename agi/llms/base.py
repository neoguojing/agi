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
        return instance

    def save_image(self, output_path: str):
        """Save PIL Image object to a file."""
        if self.pil_image:
            self.pil_image.save(output_path)


# Function to build a multi-modal message
def build_multi_modal_message(text_input: str, media_data, msg_type: MultiModalMessageType) -> HumanMessage:
    return HumanMessage(content=[
        {"type": "text", "text": text_input},
        {"type": msg_type, msg_type: media_data},
    ])


# Audio class for handling different audio types
class Audio(BaseModel):
    url: Optional[HttpUrl] = None  # Audio URL
    file_path: Optional[str] = None  # File path to audio
    samples: Optional[BytesIO] = None  # Audio sample data in BytesIO
    filename: Optional[str] = None  # File name
    filetype: Optional[str] = None  # File type (e.g., 'audio/mpeg', 'audio/wav')
    size: Optional[int] = None  # File size (in bytes)

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

        return cls(samples=samples, file_path=audio_path, filename=filename, filetype=filetype, size=size)

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
            return cls(url=url, samples=samples, filename=filename, filetype=filetype, size=size)
        else:
            raise Exception(f"Failed to download audio: {response.status_code}")

    def to_binary(self) -> bytes:
        """Convert the audio sample data to binary format."""
        return self.samples.getvalue()  # Get binary data from BytesIO


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
