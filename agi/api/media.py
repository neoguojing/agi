from typing import List, Dict, Tuple,Union,Any
from agi.apps.common import MessageContent
from agi.config import FILE_STORAGE_PATH
import os
import base64
import uuid
import requests
from mimetypes import guess_extension
from langchain_core.messages import HumanMessage


# =========================
# 工具函数
# =========================

def is_base64_string(s: str) -> bool:
    try:
        if len(s) % 4 != 0:
            return False
        base64.b64decode(s, validate=True)
        return True
    except Exception:
        return False


def guess_mime_from_ext(filename: str) -> str:
    ext = filename.split(".")[-1].lower()
    mapping = {
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "png": "image/png",
        "webp": "image/webp",
        "mp4": "video/mp4",
        "wav": "audio/wav",
        "mp3": "audio/mpeg",
        "pdf": "application/pdf",
    }
    return mapping.get(ext, "application/octet-stream")


def identify_input_type(content: str) -> str:
    if content.startswith("data:"):
        return "base64"
    if content.startswith("http://") or content.startswith("https://"):
        return "url"
    if is_base64_string(content):
        return "base64"
    if os.path.exists(content):
        return "local"
    return "unknown"


# =========================
# 核心：保存多模态内容
# 返回：(file_path, mime_type, file_name)
# =========================

def save_media_content(
    content: str,
    save_dir: str,
    filename: str = None
) -> Tuple[str, str, str]:

    os.makedirs(save_dir, exist_ok=True)

    if not filename:
        filename = uuid.uuid4().hex

    # =========================
    # BASE64
    # =========================
    if content.startswith("data:") or is_base64_string(content):

        if content.startswith("data:"):
            header, data = content.split(",", 1)
            mime_type = header.split(";")[0].replace("data:", "")
        else:
            data = content
            mime_type = "application/octet-stream"

        ext = guess_extension(mime_type) or ""
        file_name = f"{filename}{ext}"
        file_path = os.path.join(save_dir, file_name)

        try:
            with open(file_path, "wb") as f:
                f.write(base64.b64decode(data))
        except Exception as e:
            raise ValueError(f"Invalid base64: {e}")

        return file_path, mime_type, file_name

    # =========================
    # URL（流式）
    # =========================
    elif content.startswith("http://") or content.startswith("https://"):

        try:
            resp = requests.get(content, stream=True, timeout=10)
        except Exception as e:
            raise ValueError(f"Download error: {e}")

        if resp.status_code != 200:
            raise ValueError(f"Download failed: {content}")

        mime_type = resp.headers.get("Content-Type", "").split(";")[0] \
            or "application/octet-stream"

        ext = guess_extension(mime_type) or ""
        file_name = f"{filename}{ext}"
        file_path = os.path.join(save_dir, file_name)

        try:
            with open(file_path, "wb") as f:
                for chunk in resp.iter_content(8192):
                    if chunk:
                        f.write(chunk)
        except Exception as e:
            raise ValueError(f"Write error: {e}")

        return file_path, mime_type, file_name

    # =========================
    # 本地文件
    # =========================
    elif os.path.exists(content):
        file_path = content
        file_name = os.path.basename(content)
        mime_type = guess_mime_from_ext(file_name)

        return file_path, mime_type, file_name

    else:
        raise ValueError("Unsupported content format")


def process_multimodal_content(
    content: Union[str, List[MessageContent]]
) -> Tuple[List[Dict[str, Any]], str]:
    """
    将自定义 MessageContent 转换为标准多模态 Dict 格式。
    支持: text, image(落盘), file(映射为 audio/video/file 并落盘)
    """
    if isinstance(content, str):
        return [{"type": "text", "text": content}], "text"

    result = []
    # 记录最后处理的非文本类型，用于返回 content_type 标识
    last_processed_type = "text"

    for item in content:
        # 1. 处理文本类型
        if item.type == "text":
            if item.text:
                result.append({"type": "text", "text": item.text})

        # 2. 处理图片类型 (image_url -> image)
        elif item.type == "image_url":
            if not item.image_url or not item.image_url.url:
                continue
            
            # 调用落盘逻辑：屏蔽 URL/Base64 差异
            file_path, mime, _ = save_media_content(item.image_url.url, FILE_STORAGE_PATH)
            
            result.append({
                "type": "image",
                "file_id": file_path,  # 传入落盘后的本地路径
                "detail": item.image_url.detail or "auto",
                "mime_type": mime
            })
            last_processed_type = "image"

        # 3. 处理文件类型 (file -> audio/video/file)
        elif item.type == "file":
            f_obj = item.file
            if not f_obj:
                continue

            # 确定来源优先级：file_id (可能是路径) > url > base64
            source = f_obj.file_id or f_obj.url or f_obj.base64
            if not source:
                continue

            # 执行落盘
            file_path, mime_type, _ = save_media_content(source, FILE_STORAGE_PATH)
            
            # 根据 MIME 类型自动映射到具体的 LangChain 类别
            final_type = "file"
            if mime_type:
                if "audio" in mime_type:
                    final_type = "audio"
                elif "video" in mime_type:
                    final_type = "video"
            
            # 构建标准输出
            file_data = {
                "type": final_type,
                "file_id": file_path,
                "mime_type": mime_type or f_obj.mime_type
            }
            
            # 如果是 base64 且你需要保留原始数据（可选，通常落盘后不需要了）
            # if f_obj.base64: file_data["base64"] = f_obj.base64
            
            result.append(file_data)
            last_processed_type = final_type

        else:
            # 记录未知类型但不中断流程
            print(f"⚠️ Warning: Found unsupported content type: {item.type}")

    return result, last_processed_type