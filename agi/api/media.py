from typing import List, Dict, Tuple,Union,Any
from agi.apps.common import MessageContent
from agi.config import FILE_STORAGE_PATH
import os
import base64
import uuid
import requests
from mimetypes import guess_extension
from langchain_core.messages import HumanMessage
from langchain_core.messages.content import (
    create_text_block,
    create_image_block,
    create_audio_block,
)

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
) -> HumanMessage:
    """
    将自定义 MessageContent 转换为 HumanMessage（包含 content_blocks）
    """

    # ---------- 纯文本 ----------
    if isinstance(content, str):
        return HumanMessage(content=content)

    blocks = []
    text_parts = []  # 用于 fallback content

    for item in content:
        # ---------- TEXT ----------
        if item.type == "text":
            if item.text:
                blocks.append(create_text_block(text=item.text))
        # ---------- IMAGE ----------
        elif item.type == "image_url":
            if not item.image_url or not item.image_url.url:
                continue

            file_path, mime, _ = save_media_content(
                item.image_url.url, FILE_STORAGE_PATH
            )

            blocks.append(
                create_image_block(
                    file_id=file_path,
                    mime_type=mime or "image/png",
                )
            )

        # ---------- FILE ----------
        elif item.type == "file":
            f_obj = item.file
            if not f_obj:
                continue

            source = f_obj.file_id or f_obj.url or f_obj.base64
            if not source:
                continue

            file_path, mime_type, _ = save_media_content(
                source, FILE_STORAGE_PATH
            )

            if mime_type and "audio" in mime_type:
                blocks.append(
                    create_audio_block(
                        file_id=file_path,
                        mime_type=mime_type,
                    )
                )
            else:
                # fallback
                text_parts.append(f"[Unsupported file: {mime_type}]")

        else:
            print(f"⚠️ Unsupported content type: {item.type}")

    # ⚠️ 必须有 content（很多模型依赖）
    final_text = "\n".join(text_parts) if text_parts else ""

    return HumanMessage(
        content=final_text,
        content_blocks=blocks,
    )