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
import PIL.ImageOps
from io import BytesIO
import numpy as np
from diffusers.utils import load_image
from enum import Enum
import os
import re
from scipy.io.wavfile import write


class Media(BaseModel):
    data: Optional[Union[BytesIO, PILImage.Image]] = None
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    def from_data(cls, input_data: Union[str,PILImage.Image,BytesIO, np.ndarray], media_type: str = "image") -> 'Media':
        """
        Unified method to handle different data sources for media (image or audio).
        Returns a Media instance with the corresponding data.
        """
        if media_type == "image":
            media_data = cls.load_image(input_data)
        elif media_type == "audio":
            media_data = cls.load_audio(input_data)
        else:
            raise ValueError(f"Unsupported media type: {media_type}")
        
        # Now we create an instance of Media with the loaded data
        return cls(data=media_data)

    @staticmethod
    def load_image(image: Union[str, PILImage.Image, bytes]) -> PILImage.Image:
        """
        Loads `image` to a PIL Image.
        
        Args:
            image (`str`, `PIL.Image.Image`, `bytes`):
                - If `str`, it could be a URL or file path.
                - If `bytes`, it is expected to be Base64 encoded image data.
                - If `PIL.Image.Image`, it is a PIL image already.
        
        Returns:
            `PIL.Image.Image`:
                A PIL Image.
        """
        if isinstance(image, str):
            if image.startswith(('http://', 'https://')):
                # Load image from URL
                image = PILImage.open(requests.get(image, stream=True).raw)
            elif os.path.isfile(image):
                # Load image from file path
                image = PILImage.open(image)
            else:
                # Decode Base64 to image
                try:
                    image_data = base64.b64decode(image)
                    image_bytes = BytesIO(image_data)
                    image = PILImage.open(image_bytes)
                except (ValueError, TypeError):
                    raise ValueError(f"Invalid URL or file path or base64: {image}")
               
                
            
        elif isinstance(image, PILImage.Image):
            # Return already loaded PIL image
            return image
        
        else:
            raise ValueError("Invalid image format. Should be a URL, file path, Base64, or PIL image.")

        # Handle EXIF data and convert to RGB
        image = PIL.ImageOps.exif_transpose(image)
        image = image.convert("RGB")
        return image


    @staticmethod
    def load_audio(audio: Union[str, np.ndarray,BytesIO]) -> BytesIO:
        """
        Loads audio from different data sources (URL, File Path, Base64, or NumPy).
        
        Args:
            audio (`str`, `np.ndarray`):
                - If `str`, it could be a URL or file path.
                - If `base64`, it is expected to be Base64 encoded audio.
                - If `np.ndarray`, it represents a NumPy array of audio data.
        
        Returns:
            `BytesIO`: A BytesIO object containing the audio data.
        """
        if isinstance(audio, str):
            if audio.startswith(('http://', 'https://')):
                # Load audio from URL
                response = requests.get(audio)
                response.raise_for_status()  # Ensure successful request
                return BytesIO(response.content)
            elif os.path.isfile(audio):
                # Load audio from file path
                with open(audio, 'rb') as file:
                    return BytesIO(file.read())
            else:
                try:
                    audio_data = base64.b64decode(audio)
                    return BytesIO(audio_data)
                except (ValueError, TypeError):
                    raise ValueError(f"Invalid URL or file path or base64: {audio}")
        
        elif isinstance(audio, np.ndarray):
            # Convert NumPy array to audio (e.g., WAV format)
            byte_io = BytesIO()
            write(byte_io, 44100, audio)  # Assuming a sample rate of 44100 for WAV format
            byte_io.seek(0)
            return byte_io
        
        elif isinstance(audio, BytesIO):
            # Return already loaded BytesIO
            return audio
        
        else:
            raise ValueError("Unsupported audio format. Should be URL, file path, Base64, or NumPy array.")


    def to_binary(self) -> bytes:
        """Convert the media data to binary format (if applicable)."""
        if isinstance(self.data, BytesIO):
            return self.data.getvalue()
        elif isinstance(self.data, PILImage.Image):
            byte_io = BytesIO()
            self.data.save(byte_io, format="PNG")  # Assume PNG format for simplicity
            byte_io.seek(0)
            return byte_io.getvalue()
        else:
            raise TypeError("Unsupported media data format")

    def save_to_file(self, file_path: str) -> None:
        """Save the media to a file (image or audio)."""
        if isinstance(self.data, PILImage.Image):
            self.data.save(file_path)
        elif isinstance(self.data, BytesIO):
            with open(file_path, "wb") as file:
                file.write(self.data.getvalue())
        else:
            raise TypeError("Unsupported media format for saving.")

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
                    media = Media.from_data(media_data,media_type)
            if media_type == "audio":
                # Create Image instance based on media type
                media_data = content.get("audio")
                if media_data is not None and media_data != "":
                    media = Media.from_data(media_data,media_type)
            elif media_type == "text":
                prompt = content.get("text")
    elif isinstance(input.content, str):
        prompt = input.content
    
    return media, prompt

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
