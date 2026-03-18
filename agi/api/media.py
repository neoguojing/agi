from typing import List, Dict, Tuple,Union
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


def process_multimodal_content(content: Union[str, List[MessageContent]]) -> Tuple[List[Dict], str]:
    if isinstance(content, str):
        return [{"type": "text", "text": content}], "text"

    result = []
    last_input_type = "text"

    for item in content:
        t = item.type

        # =========================
        # 1. TEXT
        # =========================
        if t == "text":
            result.append({"type": "text", "text": item.text or ""})

        # =========================
        # 2. IMAGE (内网环境：强制下载并传路径)
        # =========================
        elif t == "image_url":
            img_obj = item.image_url
            if not img_obj: continue
            
            # 无论输入是 http 还是 base64，统一调用你的函数下载/保存到本地
            # file_path 会是如 "/data/storage/cache/img_abc.jpg" 的本地路径
            file_path, mime_type, _ = save_media_content(img_obj.url, FILE_STORAGE_PATH)
            
            # 按照你提供的 LangChain 图片输出规范
            result.append({
                "type": "image",
                "file_id": file_path,   # 传本地路径
                "detail": img_obj.detail or "auto"
            })
            last_input_type = "image"

        # =========================
        # 3. FILE (Audio/Video/Doc - 统一落盘)
        # =========================
        elif t == "file":
            f_obj = item.file
            if not f_obj: continue

            # 优先级：已有的路径 > url > base64
            source = f_obj.file_id or f_obj.url or f_obj.base64
            
            # 同样调用 save_media_content 屏蔽来源差异
            file_path, mime_type, _ = save_media_content(source, FILE_STORAGE_PATH)
            
            # 根据 MIME 映射 LangChain 的具体 type
            lc_type = "file"
            if mime_type:
                if "audio" in mime_type: lc_type = "audio"
                elif "video" in mime_type: lc_type = "video"
            
            result.append({
                "type": lc_type,
                "file_id": file_path,  # 统一传本地路径
                "mime_type": mime_type or f_obj.mime_type
            })
            last_input_type = lc_type

        else:
            raise ValueError(f"Unsupported type: {t}")

    return result, last_input_type
