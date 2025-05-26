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
from agi.config import log,AGI_DEBUG,CACHE_DIR,BASE_URL
import inspect
import json
import traceback
import uuid
import hashlib
import base64
import mimetypes
from PIL import Image
import io
import os
from typing import Tuple
import re
from urllib.parse import urlparse

# 处理推理模型返回
def split_think_content(content):
    think_content = ""
    other_content = content
    try:
        if isinstance(content,list):
            content = content[0].get("text","")
            
        import re
        match = re.search(r"(<think>\s*.*?\s*</think>)\s*(.*)", content, re.DOTALL)

        if match:
            think_content = match.group(1).strip()  # 保留 <think> 标签，并去掉换行
            other_content = match.group(2).strip()  # 去掉换行

    except Exception as e:
        log.error(e)

    return think_content,other_content

def get_last_message_text(state: AgentState):
    last_message = state["messages"][-1]
    if isinstance(last_message,HumanMessage):
        if isinstance(last_message.content,str):
            return last_message.content
        elif isinstance(last_message.content,list):
            for item in last_message.content:
                if item["type"] == "text":
                    return item["text"]
    return ""

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
        
        log.info(f"message:{x}")

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
    base64_pattern = re.compile(r"^(data:\w+/\w+;base64,)?[A-Za-z0-9+/=\s]+$")
    try:
        # 校验是否 base64 可解码
        content = input_str.split(",")[-1].strip()  # 支持带 data: 开头
        if base64_pattern.match(input_str):
            base64.b64decode(content, validate=True)
            return "base64"
    except Exception:
        pass

    return "unknown"

def save_base64_content(base64_str: str, output_dir: str = CACHE_DIR) -> Tuple[str, str]:
    """
    将 base64 编码的图片或语音内容保存为文件。

    Args:
        base64_str (str): base64 编码的字符串，支持带有 `data:*/*;base64,` 前缀。
        output_dir (str): 存储目录，默认为 ./output

    Returns:
        Tuple[str, str]: 返回文件路径和类型（image 或 audio）

    Raises:
        ValueError: 当输入不是有效 base64 图片或语音内容时。
    """

    # 自动创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    ext = None
    content_type = None
    # 检测是否包含 mime 类型头
    if base64_str.startswith("data:"):
        header, encoded = base64_str.split(",", 1)
        mime_type = header.split(";")[0][5:]
        ext = mimetypes.guess_extension(mime_type)
        
        if mime_type.startswith("image/"):
            content_type = "image" 
            if ext is None:
                ext = "jpg"
        if mime_type.startswith("audio/"):
            content_type = "audio" 
            if ext is None:
                ext = "wav"
        
    else:
        # 如果没有头部信息，则无法判断类型，默认用 .bin 保存
        encoded = base64_str
        mime_type = None
        content_type = None
        ext = ".bin"

    if content_type not in {"image", "audio"}:
        raise ValueError("Unsupported or unknown content type")

    # 生成文件名
    filename = f"{content_type}_{int(os.times()[4] * 1000)}.{ext}"
    file_path = os.path.join(output_dir, filename)

    # 保存文件
    with open(file_path, "wb") as f:
        f.write(base64.b64decode(encoded))
    
    if ext != ".jpg" and content_type == "image":
        try:
            # 转换为JPEG
            jpeg_filename = f"{os.path.splitext(filename)[0]}.jpg"
            jpeg_path = os.path.join(output_dir, jpeg_filename)
            
            with Image.open(file_path) as img:
                # 转换为RGB模式（JPEG不支持透明通道）
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                img.save(jpeg_path, 'JPEG', quality=95)
            
            # 删除原始文件（可选）
            os.remove(file_path)
            file_path = jpeg_path
        except Exception as e:
            print(f"Failed to convert to JPEG: {e}")
            # 如果转换失败，继续使用原始文件
    url = ""
    if file_path.startswith(CACHE_DIR):
        url = os.path.join(BASE_URL, "v1/files", os.path.basename(file_path))
    return file_path,url, content_type

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