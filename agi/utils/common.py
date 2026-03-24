import time
from typing import Any, Union, Literal, List, Dict
from pydantic import BaseModel, HttpUrl, constr, ConfigDict
import base64
import requests
from typing import Optional
from PIL import Image as PILImage
import PIL.ImageOps
from io import BytesIO
import numpy as np
import os
import re
from scipy.io.wavfile import write
import mimetypes
import uuid
from urllib.parse import urlparse
from tempfile import gettempdir
from pathlib import Path
import urllib.parse
from agi.config import BASE_URL,CACHE_DIR,FILE_STORAGE_PATH


class Timer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.end = time.perf_counter()
        self.interval = self.end - self.start
        print(f"执行时间: {self.interval:.4f} 秒")

def remove_data_uri_prefix(data_uri):
    """
    通用地移除 Data URI 前缀。

    Args:
        data_uri (str): Data URI 字符串。

    Returns:
        str: 去除前缀后的 Base64 编码数据。
    """
    match = re.match(r"data:([a-z]+/[a-z]+(?:;[a-z]+=[a-z]+)?)?(;base64)?,(.*)", data_uri)
    if match:
        return match.group(3)  # 返回 Base64 编码数据部分
    else:
        return data_uri  # 如果不是 Data URI，则返回原始字符串
    
class Media(BaseModel):
    data: Optional[Union[BytesIO, PILImage.Image]] = None
    model_config = ConfigDict(arbitrary_types_allowed=True)
    file_path: Optional[str] = None
    file_url: Optional[str] = None
    file_base64 :Optional[str] = None
    
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
                    image = remove_data_uri_prefix(image)
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
                response = requests.get(audio,timeout=10)
                response.raise_for_status()  # Ensure successful request
                return BytesIO(response.content)
            elif os.path.isfile(audio):
                # Load audio from file path
                with open(audio, 'rb') as file:
                    return BytesIO(file.read())
            else:
                try:
                    audio = remove_data_uri_prefix(audio)
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
        

def is_url(input_str):
    return re.match(r'^https?://', input_str)

def is_base64(input_str):
    try:
        base64.b64decode(input_str, validate=True)
        return True
    except Exception:
        return False

def download_file(url,target_path: str):
    resp = requests.get(url, stream=True)
    if not resp.ok:
        raise ValueError("Failed to download file from URL")

    ext = guess_extension(resp.headers.get("Content-Type", "application/octet-stream"))
    file_path = os.path.join(target_path, f"url_{uuid.uuid4().hex}{ext}")
    with open(file_path, "wb") as f:
        for chunk in resp.iter_content(1024):
            f.write(chunk)
    return file_path

def save_base64_to_file(b64_data,target_path:str):
    import magic  # pip install python-magic
    try:
        binary = base64.b64decode(b64_data)
        # guess mime
        mime = magic.from_buffer(binary, mime=True)
        ext = guess_extension(mime)
        file_path = os.path.join(target_path, f"b64_{uuid.uuid4().hex}{ext}")
        with open(file_path, "wb") as f:
            f.write(binary)
        return file_path
    except Exception as e:
        raise ValueError("Invalid base64 data") from e

def guess_extension(mime_type):
    ext = mimetypes.guess_extension(mime_type)
    return ext or ".bin"

def guess_type(filepath):
    import magic  # pip install python-magic
    return magic.from_file(filepath, mime=True)

def classify_mime(mime_type):
    if mime_type.startswith("image/"):
        return "image"
    elif mime_type.startswith("audio/"):
        return "audio"
    elif mime_type.startswith("video/"):
        return "video"
    else:
        return "unknown"

def detect_input_and_save(input_data: str,target_path: str):
    """
    判断输入类型，并保存为文件（若需要），返回：
    - 文件路径
    - 类型（image/audio/video/unknown）
    """
    file_path = None

    if is_url(input_data):
        file_path = download_file(input_data,target_path)
    elif is_base64(input_data):
        file_path = save_base64_to_file(input_data,target_path)
    elif os.path.isfile(input_data):
        file_path = input_data
    else:
        raise ValueError("输入不是合法的 URL、base64 或本地文件路径")

    mime = guess_type(file_path)
    category = classify_mime(mime)

    return file_path, category

def file_to_data_uri(file_path: str) -> str:
    """
    读取文件并转成带 MIME 类型的 Data URI 格式 Base64 字符串

    :param file_path: 文件路径
    :return: data URI格式的字符串
    """
    mime_type, _ = mimetypes.guess_type(file_path)
    if mime_type is None:
        mime_type = "application/octet-stream"  # 默认类型

    with open(file_path, "rb") as f:
        file_bytes = f.read()
    base64_str = base64.b64encode(file_bytes).decode("utf-8")

    return f"data:{mime_type};base64,{base64_str}"


def is_relative_path(path_str: str) -> bool:
    if not os.path.isfile(path_str):
        return False
    return not Path(path_str).is_absolute()

def identify_input_type(input_str: str) -> str:
    """
    判断输入字符串是文件路径、URL 还是 base64 编码。

    Returns:
        str: "path", "url", "base64", 或 "unknown"
    """

    # 判断是否为 URL
    parsed = urlparse(input_str)
    if parsed.scheme in ("http", "https") and parsed.netloc:
        return "url"

    # 判断是否为文件路径
    if os.path.exists(input_str):
        return "path"

    # 判断是否为 base64（允许带 mime 头的）
    base64_pattern = re.compile(r"^(data:[^;]+;base64,)?[A-Za-z0-9+/=\s]+$")
    try:
        # 校验是否 base64 可解码
        content = input_str.split(",")[-1].strip()  # 支持带 data: 开头
        if base64_pattern.match(input_str):
            base64.b64decode(content, validate=True)
            return "base64"
    except Exception:
        pass

    return "unknown"


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
    if not os.path.realpath(file_path).startswith(os.path.realpath(FILE_STORAGE_PATH)):
        raise ValueError(f"File path is outside the root directory.root ={FILE_STORAGE_PATH},file_path={file_path}")
    
    # 获取相对于 UPLOAD_DIR 的文件名
    file_name = os.path.basename(file_path)
    
    # 构建预览 URL
    preview_url = f"{base_url}/v1/files/{urllib.parse.quote(file_name)}"
    return preview_url


from typing import Any, List, Union, overload
from langchain_core.messages import BaseMessage, AIMessage, ToolMessage

def extract_message_content(msg: BaseMessage) -> str:
    """从 LangChain 消息中提取纯文本内容（修复 dict 访问）"""
    
    # 简单文本（向后兼容）
    if hasattr(msg, 'content') and msg.content:
        if isinstance(msg.content, str):
            return msg.content
    
    # content_blocks（dict 格式）
    if hasattr(msg, 'content_blocks') and msg.content_blocks:
        parts = []
        for block in msg.content_blocks:
            # ✅ 使用 dict 访问 block["type"]
            block_type = block.get("type")
            
            if block_type == "text":
                parts.append(block.get("text", ""))
            elif block_type == "reasoning":
                parts.append(f"[推理] {block.get('reasoning', '')}")
            elif block_type in ("image", "audio", "video"):
                source = block.get("url") or block.get("base64", "[:base64:]")[:50] + "..."
                mime = block.get("mime_type", "")
                parts.append(f"[{block_type.upper()}] {mime}: {source}")
            elif block_type == "file":
                source = block.get("url") or block.get("mime_type", "?")
                parts.append(f"[文件] {source}")
            elif block_type == "tool_call":
                name = block.get("name", "unknown")
                args = str(block.get("args", {}))
                parts.append(f"[工具] {name}: {args}")
            else:
                # 未知类型，安全处理
                parts.append(f"[{block_type}] {str(block)}")
        
        return "\n".join(parts)
    
    # 其他情况
    if isinstance(msg, AIMessage) and msg.tool_calls:
        return "\n".join([str(tc["args"]) for tc in msg.tool_calls])
    if isinstance(msg, ToolMessage):
        return msg.content or ""
    
    return str(msg.content or "")

@overload
def extract_messages_content(messages: BaseMessage) -> str: ...

@overload
def extract_messages_content(messages: List[BaseMessage]) -> List[str]: ...

def extract_messages_content(messages: Union[BaseMessage, List[BaseMessage]]) -> Union[str, List[str]]:
    """从消息（单条/列表）中提取内容，全支持多模态"""
    if isinstance(messages, BaseMessage):
        return extract_message_content(messages)
    return [extract_message_content(msg) for msg in messages]


from langchain_core.messages import ContentBlock, SystemMessage


def append_to_system_message(
    system_message: SystemMessage | None,
    text: str,
) -> SystemMessage:
    """Append text to a system message.

    Args:
        system_message: Existing system message or None.
        text: Text to add to the system message.

    Returns:
        New SystemMessage with the text appended.
    """
    new_content: list[ContentBlock] = list(system_message.content_blocks) if system_message else []
    if new_content:
        text = f"\n\n{text}"
    new_content.append({"type": "text", "text": text})
    return SystemMessage(content_blocks=new_content)