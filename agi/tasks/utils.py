from langchain_core.messages import AIMessage, HumanMessage,BaseMessage,ToolMessage,SystemMessage
from typing import Any, List, Mapping, Optional, Union
from langchain_core.runnables import (
    RunnableLambda
)
from langchain_core.messages import (
    RemoveMessage,
    convert_to_messages,
    message_chunk_to_message,
)
from agi.tasks.define import AgentState
from langgraph.graph.message import Messages
from agi.config import log,AGI_DEBUG,CACHE_DIR,BASE_URL,FILE_STORAGE_PATH
from agi.utils.common import identify_input_type
import inspect
import json
import traceback
import uuid
import hashlib
import base64
import mimetypes
import shutil
import requests
import base64
from PIL import Image
import io
import os
from typing import Tuple,Callable
import re
from urllib.parse import urlparse
from bs4 import BeautifulSoup

# 处理推理模型返回
def split_think_content(content):
    think_content = None
    other_content = content
    try:
        if isinstance(content,list):
            content = content[0].get("text","")
            
        import re
        match = re.search(r"(<think>\s*.*?\s*</think>)\s*(.*)", content, re.DOTALL)

        if match:
            think_content = match.group(1).strip()  # 保留 <think> 标签，并去掉换行
            other_content = match.group(2).strip()  # 去掉换行
        if think_content:
            soup = BeautifulSoup(think_content, "html.parser")
            tag = soup.find("think")
            if tag is None:
                think_content = None
            else:
                inner = tag.get_text()
                if inner.strip() == "":
                    think_content = None

    except Exception as e:
        log.error(e)

    return think_content,other_content

def get_last_message_text(state: AgentState):
    last_message = state["messages"][-1]
    text = ""
    if isinstance(last_message,HumanMessage):
        if isinstance(last_message.content,str):
            text = last_message.content
        elif isinstance(last_message.content,list):
            for item in last_message.content:
                if item["type"] == "text":
                    text = item["text"]

    return text.removesuffix("/no_think").strip()

def refine_human_message(
    state: AgentState,
    formatter: Callable[[Any], Any] = None
) -> Union[str, None]:
    """
    处理最后一条人类消息，可应用自定义格式化函数
    
    参数:
        state: 包含消息列表的状态字典
        formatter: 可选的消息格式化函数
    
    返回:
        处理后的文本内容或None
    """
    if not state.get("messages"):
        return
        
    last_message = state["messages"][-1]
    
    if not isinstance(last_message, HumanMessage):
        return
    
    if isinstance(last_message.content, str):
        last_message.content = formatter(last_message.content)
    elif isinstance(last_message.content , list):
        for item in last_message.content :
            if item["type"] == "text":
                _,item["text"] = formatter(item["text"])

# 修复最后一条AI消息的text内容,去除特定标签内容
def refine_last_message_text(message :Union[AIMessage,ToolMessage,list[BaseMessage]]):
    last_message = message
    if isinstance(message,list):
        last_message = message[-1]

    if not isinstance(last_message,HumanMessage):
        if isinstance(last_message.content,str):
            _,last_message.content = split_think_content(last_message.content)
        elif isinstance(last_message.content,list):
            for item in last_message.content:
                if item["type"] == "text":
                    _,item["text"] = split_think_content(item["text"])
    return last_message


refine_last_message_runnable = RunnableLambda(refine_last_message_text)

def graph_response_format(message :Union[AIMessage,ToolMessage,list[BaseMessage]]):
    
    refine_last_message_text(message)
    if isinstance(message,list):
        return {"messages": message}
    
    return {"messages": [message]} 

graph_response_format_runnable = RunnableLambda(graph_response_format)

def format_state_message_to_str(messages):
    filter_messages = []
    for message in messages:
        if isinstance(message,SystemMessage):
            continue
        # 修正请求的类型，否则openapi会报错
        if not isinstance(message.content,str):
             message.content = json.dumps(message.content)
        filter_messages.append(message)
    return filter_messages

# TODO parent_name不是实际的函数
def debug_info(x : Any):
    if AGI_DEBUG:
        parent_name = ""
        stack = inspect.stack()
        if len(stack) > 2:  # stack[0] 是 get_parent_function_name，stack[1] 是调用它的函数
            parent_name = stack[2].function  # stack[2] 是再往上的函数，即父函数
        
        log.debug(f"message:{x}")

    return x

debug_tool = RunnableLambda(debug_info)

def compute_content_hash(content: any) -> str:
    """
    计算 content 的哈希值。
    如果 content 是列表或字典，则将其序列化为 JSON 字符串（sort_keys=True确保稳定顺序）。
    对于其他类型，则转换为字符串。
    """
    if isinstance(content, (dict, list)):
        # 序列化为 JSON 字符串，并保证键排序以获得一致的结果
        serialized = json.dumps(content, sort_keys=True)
    else:
        serialized = str(content)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

def add_messages(left: Messages, right: Messages) -> Messages:
    """
    合并两个消息列表，依据消息的 content 哈希值和消息类型进行更新或删除。

    如果右侧消息与左侧已有消息的 content（基于哈希）和消息类型相同，
    则用右侧消息替换左侧消息；如果右侧消息为 RemoveMessage，
    则删除所有匹配的消息。

    Args:
        left: 基础消息列表（或者单条消息）
        right: 要合并的消息列表（或者单条消息）

    Returns:
        合并后的消息列表
    """
    try:
        # 转换为列表
        if not isinstance(left, list):
            left = [left]
        if not isinstance(right, list):
            right = [right]
        log.debug(f"add_messages--begin--{left} \n {right}")
        left = [message_chunk_to_message(m) for m in convert_to_messages(left)]
        right = [message_chunk_to_message(m) for m in convert_to_messages(right)]
        log.debug(f"add_messages--after--{left} \n {right}")
        # 为缺失 id 的消息分配唯一 ID
        for m in left:
            if m.id is None:
                m.id = str(uuid.uuid4())
        for m in right:
            if m.id is None:
                m.id = str(uuid.uuid4())

        # 使用 content 的哈希值构建 key，同时结合消息类型
        def make_key(m):
            content_hash = compute_content_hash(m.content)
            return (content_hash, m.__class__.__name__)

        left_idx_by_key = {make_key(m): i for i, m in enumerate(left)}
        merged = left.copy()
        keys_to_remove = set()

        for m in right:
            key = make_key(m)
            if key in left_idx_by_key:
                if isinstance(m, RemoveMessage):
                    keys_to_remove.add(key)
                else:
                    merged[left_idx_by_key[key]] = m
            else:
                if isinstance(m, RemoveMessage):
                    raise ValueError(
                        f"Attempting to delete a message with content (hash) and type that doesn't exist: {key}"
                    )
                merged.append(m)

        merged = [m for m in merged if make_key(m) not in keys_to_remove]
        log.debug(f"--------{merged}")
        return merged
    except Exception as e:
        log.error(e)
        print(traceback.format_exc())


def save_media_content(source: str, output_dir: str = FILE_STORAGE_PATH) -> Tuple[str, str, Optional[str]]:
    """
    将 base64 编码的图片或音频内容、本地文件或远程URL保存为本地文件。

    Args:
        source (str): base64 字符串、文件路径或 URL。
        output_dir (str): 存储目录，默认 FILE_STORAGE_PATH。

    Returns:
        Tuple[str, str, Optional[str]]: 文件路径、本地服务URL、类型（image/audio/None）。

    Raises:
        ValueError: 无法识别的格式或不支持的媒体类型。
    """
    os.makedirs(output_dir, exist_ok=True)
    content_type = None
    file_path = ""
    url = ""

    def generate_filename(extension: str) -> str:
        return f"{uuid.uuid4().hex}{extension}"

    def detect_content_type(path: str) -> Optional[str]:
        mime_type, _ = mimetypes.guess_type(path)
        if mime_type:
            if mime_type.startswith("image/"):
                return "image"
            elif mime_type.startswith("audio/"):
                return "audio"
        # fallback: check by file extension
        ext = os.path.splitext(path)[1].lower()
        if ext in {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}:
            return "image"
        elif ext in {".wav", ".mp3", ".m4a", ".flac", ".ogg"}:
            return "audio"
        return None

    # === 1. base64（含或不含 data: 前缀） ===
    if source.startswith("data:"):
        header, encoded = source.split(",", 1)
        mime_type = header.split(";")[0][5:]
        ext = mimetypes.guess_extension(mime_type) or ".bin"
        content_type = "image" if mime_type.startswith("image/") else \
                       "audio" if mime_type.startswith("audio/") else None
        if content_type is None:
            raise ValueError(f"Unsupported MIME type: {mime_type}")
        filename = generate_filename(ext)
        file_path = os.path.join(output_dir, filename)
        with open(file_path, "wb") as f:
            f.write(base64.b64decode(encoded))

    elif os.path.isfile(source):
        # === 2. 本地文件路径 ===
        filename = os.path.basename(source)
        file_path = os.path.join(output_dir, filename)
        if not os.path.isfile(file_path):
            shutil.copy(source, file_path)
        content_type = detect_content_type(file_path)

    elif source.startswith(("http://", "https://")):
        # === 3. 网络URL ===
        response = requests.get(source, timeout=10)
        response.raise_for_status()
        mime_type = response.headers.get("Content-Type", "")
        ext = mimetypes.guess_extension(mime_type) or ".bin"
        filename = os.path.basename(urlparse(source).path) or generate_filename(ext)
        file_path = os.path.join(output_dir, filename)
        with open(file_path, "wb") as f:
            f.write(response.content)
        content_type = detect_content_type(file_path)

    else:
        # === 4. 纯 base64，无 data: 前缀 ===
        try:
            decoded = base64.b64decode(source)
        except Exception:
            raise ValueError("Invalid base64 encoding.")
        ext = ".bin"
        filename = generate_filename(ext)
        file_path = os.path.join(output_dir, filename)
        with open(file_path, "wb") as f:
            f.write(decoded)
        content_type = detect_content_type(file_path)

    # === 图像处理：可选转换为 JPEG ===
    if content_type == "image":
        try:
            with Image.open(file_path) as img:
                ext = os.path.splitext(file_path)[1].lower()
                if ext in {".png", ".webp", ".bmp", ".tiff"}:
                    # 转换为 JPEG
                    jpeg_path = os.path.splitext(file_path)[0] + ".jpg"
                    if img.mode != "RGB":
                        img = img.convert("RGB")
                    img.save(jpeg_path, format="JPEG", quality=95)
                    os.remove(file_path)
                    file_path = jpeg_path
        except Exception as e:
            print(f"Warning: Failed to convert image: {e}")

    # === 构造 URL ===
    if file_path.startswith(FILE_STORAGE_PATH):
        url = os.path.join(BASE_URL, "v1/files", os.path.basename(file_path))

    return file_path, url, content_type

def image_path_to_base64_uri(image_path: str) -> Optional[str]:
    input_type = identify_input_type(image_path)
    if input_type != "path":
        return image_path
    
    if not os.path.isfile(image_path):
        print("路径无效或文件不存在。")
        return None

    mime_type, _ = mimetypes.guess_type(image_path)
    if not mime_type or not mime_type.startswith("image/"):
        print("不是有效的图片类型。")
        return None

    try:
        with open(image_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
            return f"data:{mime_type};base64,{encoded}"
    except Exception as e:
        print(f"读取或编码失败: {e}")
        return None

def graph_print(graph):
    try:
            # Generate the image as a byte stream
        image_data = graph.get_graph().draw_mermaid_png()

        # Create a PIL Image object from the byte stream
        image = Image.open(io.BytesIO(image_data))

        # Save the image to a file
        image.save(f"{graph.get_name()}.png")
    except Exception:
        # This requires some extra dependencies and is optional
        pass


# 音频转换为Base64（带前缀）
def audio_to_base64(audio_path):
    with open(audio_path, "rb") as audio_file:
        encoded_audio = base64.b64encode(audio_file.read()).decode('utf-8')
    mime_type, _ = mimetypes.guess_type(audio_path)
    prefix = f"data:{mime_type};base64," if mime_type else "data:audio/wav;base64,"
    return prefix + encoded_audio