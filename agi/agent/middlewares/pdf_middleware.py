"""PDF 解析工具中间件 - 为 Agent 提供 PDF 解析工具。

核心功能：
1. 指定 PDF 文件地址或目录，批量解析 PDF
2. 提取每页 PDF 的内容
3. LLM 提炼 PDF 内容进行输出（两种模式）
   - 模式 1: 直接提取文本
   - 模式 2: PDF 转换为图片，由 LLM 识别图片内容

参考：filesystem_middleware.py 的中间件实现风格

使用方法：
    from agi.agent.middlewares.pdf_middleware import PdfMiddleware
    from langgraph import create_graph
    from deepagents import StoreBackend

    graph = create_graph(
        middleware=[
            PdfMiddleware(
                extract_mode="image",  # "text" 或 "image"
                page_range=None,
            )
        ],
        backend=StoreBackend()
    )
"""

import asyncio
import base64
import os
import re
import tempfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Dict, List, Literal, Callable, Awaitable
from urllib.parse import urlparse

# 尝试导入 PyMuPDF
try:
    import fitz
except ImportError:
    fitz = None

# 尝试导入 PIL
try:
    from PIL import Image
except ImportError:
    Image = None


# ========================
# 常量定义
# ========================
DEFAULT_TIMEOUT = 60  # 秒
DEFAULT_IMG_QUALITY = 85
DEFAULT_IMG_WIDTH = 1200
DEFAULT_WORKERS = 4
DEFAULT_PAGE_LIMIT = 100

# PDF MIME 类型
PDMIME = "application/pdf"

# PDF 模式
PDF_MODE_TEXT = "text"
PDF_MODE_IMAGE = "image"

# PDF 图片格式
PDF_IMG_FORMAT_PNG = "PNG"
PDF_IMG_FORMAT_JPEG = "JPEG"

# PDF 文件扩展名
PATTERN_PDF = re.compile(r"\.pdf$", re.IGNORECASE)

# 支持的图片扩展名
IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".gif", ".webp"})


# ========================
# 工具函数
# ========================


def is_pdf(file_path: str) -> bool:
    """检查文件是否为 PDF"""
    return PATTERN_PDF.search(file_path) is not None


def validate_pdf_path(file_path: str) -> str:
    """验证 PDF 路径"""
    file_path = str(file_path).strip()
    if not file_path:
        raise ValueError("文件路径不能为空")
    if not is_pdf(file_path):
        raise ValueError(f"文件不是 PDF 格式：{file_path}")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在：{file_path}")
    return os.path.abspath(file_path)


def validate_pdf_path_or_directory(file_path: str) -> str:
    """验证 PDF 文件或目录路径"""
    file_path = str(file_path).strip()
    if not file_path:
        raise ValueError("文件/目录路径不能为空")
    if is_pdf(file_path):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在：{file_path}")
        return os.path.abspath(file_path)
    if os.path.isdir(file_path):
        if not os.path.exists(file_path):
            raise NotADirectoryError(f"目录不存在：{file_path}")
        return os.path.abspath(file_path)
    raise ValueError(f"无效的路径：{file_path}")


def scan_pdf_directory(directory: str) -> List[str]:
    """扫描 PDF 目录"""
    dir_path = Path(directory)
    if not dir_path.is_dir():
        raise ValueError(f"目录不存在：{directory}")
    return sorted([str(p) for p in dir_path.glob("*.pdf")])


def extract_metadata_from_pdf(file_path: str) -> Dict[str, Any]:
    """从 PDF 提取元数据"""
    if fitz is None:
        return {}
    try:
        doc = fitz.open(file_path)
        meta = doc.metadata or {}
        doc.close()
        return {
            "title": meta.get("Title"),
            "author": meta.get("Author"),
            "creator": meta.get("Creator"),
            "producer": meta.get("Producer"),
            "creation_date": meta.get("CreationDate"),
            "modification_date": meta.get("ModDate"),
            "subject": meta.get("Subject"),
            "keywords": meta.get("Keywords"),
            "page_count": len(doc),
        }
    except Exception as e:
        return {"error": str(e)}


# ========================
# PDF 解析工具
# ========================


def parse_pdf_tool(
    file_path: str,
    extract_mode: str = PDF_MODE_IMAGE,
    page_range: Optional[range] = None,
    timeout: int = DEFAULT_TIMEOUT,
) -> Dict[str, Any]:
    """
    解析单个 PDF 文件的工具函数。

    Args:
        file_path: PDF 文件路径
        extract_mode: 提取模式 - "text" 或 "image"
        page_range: 页码范围，None=全解析
        timeout: 超时时间（秒）

    Returns:
        解析结果字典

    Raises:
        FileNotFoundError: 文件不存在
        ValueError: 文件格式错误或配置错误
        ImportError: PyMuPDF 未安装
    """
    if fitz is None:
        raise ImportError("请安装 PyMuPDF: pip install pymupdf")

    try:
        file_path = validate_pdf_path(file_path)
    except Exception as e:
        return {"error": f"验证路径失败：{e}"}

    try:
        doc = fitz.open(file_path)
    except Exception as e:
        return {"error": f"无法打开 PDF: {e}"}

    try:
        page_count = len(doc)
        all_text = []

        # 确定页码范围
        if page_range:
            start = page_range.start if page_range.start is not None else 0
            end = page_range.stop if page_range.stop is not None else page_count
            pages = range(start, end)
        else:
            pages = range(page_count)

        # 提取每页内容
        for idx in pages:
            page = doc.load_page(idx)

            # 提取文本
            text = page.get_text("text", flags=fitz.TEXT_PRESERVE_WHITESPACE)
            if text:
                all_text.append(f"--- Page {idx + 1} ---\n{text}\n")

            # 提取图片（仅在图片模式下）
            if extract_mode == PDF_MODE_IMAGE:
                try:
                    img_list = page.get_images(full=True)
                    if img_list:
                        for img_index in img_list:
                            try:
                                base_image = page.get_image(img_index[0], insertion_point=0)
                                if base_image:
                                    bitmap = base_image.get_pixmap()
                                    img_data = {
                                        "index": img_index[0],
                                        "width": bitmap.width,
                                        "height": bitmap.height,
                                        "rotation": bitmap.rotation,
                                    }
                                    # 将图片转为 base64（可选）
                                    img_data["base64"] = base64.b64encode(bitmap.tobytes()).decode("utf-8")
                                    all_text.append(f"[Image {img_index[0]}: {bitmap.width}x{bitmap.height}]\n")
                            except Exception:
                                pass
                except Exception:
                    pass

        # 获取文档元数据
        metadata = extract_metadata_from_pdf(file_path)

        # 构建结果
        result = {
            "file_path": file_path,
            "page_count": page_count,
            "extract_mode": extract_mode,
            "pages": page_count,
            "text": "".join(all_text),
            "metadata": metadata,
        }

        return result

    finally:
        doc.close()


async def aparse_pdf_tool(
    file_path: str,
    extract_mode: str = PDF_MODE_IMAGE,
    page_range: Optional[range] = None,
    timeout: int = DEFAULT_TIMEOUT,
) -> Dict[str, Any]:
    """异步解析单个 PDF 文件"""
    return await asyncio.to_thread(parse_pdf_tool, file_path, extract_mode, page_range, timeout)


def batch_parse_pdfs_tool(
    file_paths: List[str],
    extract_mode: str = PDF_MODE_IMAGE,
    page_range: Optional[range] = None,
    max_workers: int = DEFAULT_WORKERS,
) -> List[Dict[str, Any]]:
    """
    批量解析 PDF 文件列表。

    Args:
        file_paths: PDF 文件路径列表
        extract_mode: 提取模式 - "text" 或 "image"
        page_range: 页码范围，None=全解析
        max_workers: 并发工作线程数

    Returns:
        解析结果列表

    Raises:
        FileNotFoundError: 文件不存在
        ValueError: 文件格式错误或配置错误
    """
    if fitz is None:
        raise ImportError("请安装 PyMuPDF: pip install pymupdf")

    executor = ThreadPoolExecutor(max_workers=max_workers)
    futures = [
        executor.submit(
            parse_pdf_tool,
            fp,
            extract_mode,
            page_range,
            DEFAULT_TIMEOUT,
        )
        for fp in file_paths
    ]

    results = []
    for future in futures:
        try:
            results.append(future.result(timeout=timeout))
        except Exception as e:
            results.append({"error": str(e), "file_path": fp})

    executor.shutdown(wait=True)
    return results


async def abatch_parse_pdfs_tool(
    file_paths: List[str],
    extract_mode: str = PDF_MODE_IMAGE,
    page_range: Optional[range] = None,
    max_workers: int = DEFAULT_WORKERS,
) -> List[Dict[str, Any]]:
    """异步批量解析 PDF 文件列表"""
    return await asyncio.to_thread(
        batch_parse_pdfs_tool,
        file_paths,
        extract_mode,
        page_range,
        max_workers,
    )


def parse_directory_tool(
    directory: str,
    extract_mode: str = PDF_MODE_IMAGE,
    page_range: Optional[range] = None,
    timeout: int = DEFAULT_TIMEOUT,
) -> Dict[str, Any]:
    """
    解析 PDF 目录。

    Args:
        directory: PDF 目录路径
        extract_mode: 提取模式 - "text" 或 "image"
        page_range: 页码范围，None=全解析
        timeout: 超时时间（秒）

    Returns:
        解析结果字典

    Raises:
        NotADirectoryError: 目录不存在
        ValueError: 目录中无 PDF 文件
    """
    try:
        file_paths = scan_pdf_directory(directory)
    except Exception as e:
        return {"error": f"扫描目录失败：{e}"}

    if not file_paths:
        return {
            "error": f"目录中无 PDF 文件：{directory}",
            "file_count": 0,
        }

    results = batch_parse_pdfs_tool(
        file_paths,
        extract_mode,
        page_range,
        DEFAULT_WORKERS,
    )

    return {
        "directory": directory,
        "file_count": len(file_paths),
        "results": results,
        "success_count": sum(1 for r in results if "error" not in r),
        "failed_count": sum(1 for r in results if "error" in r),
    }


async def aparse_directory_tool(
    directory: str,
    extract_mode: str = PDF_MODE_IMAGE,
    page_range: Optional[range] = None,
    timeout: int = DEFAULT_TIMEOUT,
) -> Dict[str, Any]:
    """异步解析 PDF 目录"""
    return await asyncio.to_thread(parse_directory_tool, directory, extract_mode, page_range, timeout)


def export_pdf_to_images(
    file_path: str,
    output_dir: Optional[str] = None,
    page_range: Optional[range] = None,
    img_width: int = DEFAULT_IMG_WIDTH,
    img_quality: int = DEFAULT_IMG_QUALITY,
) -> List[str]:
    """
    将 PDF 转换为图片。

    Args:
        file_path: PDF 文件路径
        output_dir: 输出目录，None=临时目录
        page_range: 页码范围，None=所有页面
        img_width: 图片宽度
        img_quality: 图片质量

    Returns:
        生成的图片路径列表

    Raises:
        ImportError: PyMuPDF 未安装
    """
    if fitz is None:
        raise ImportError("请安装 PyMuPDF: pip install pymupdf")

    try:
        file_path = validate_pdf_path(file_path)
    except Exception as e:
        raise ValueError(f"验证路径失败：{e}")

    try:
        doc = fitz.open(file_path)
    except Exception as e:
        raise ValueError(f"无法打开 PDF: {e}")

    images = []
    try:
        for idx, page in enumerate(doc, start=1):
            if page_range and idx not in page_range:
                continue

            # 渲染页面为图片
            mat = fitz.Matrix(1, 1)
            pix = page.get_pixmap(matrix=mat)

            # 缩放
            if img_width < pix.width:
                mat = fitz.Matrix(img_width / pix.width, img_width / pix.height)
                pix = page.get_pixmap(matrix=mat)

            # 生成文件名
            ext = "png"
            filename = f"{Path(file_path).stem}_p{idx:04d}.{ext}"

            # 保存
            if output_dir:
                filepath = os.path.join(output_dir, filename)
            else:
                with tempfile.TemporaryDirectory() as tmpdir:
                    filepath = os.path.join(tmpdir, filename)

            # 保存为图片
            with open(filepath, "wb") as f:
                f.write(pix.tobytes())

            images.append(filepath)
    finally:
        doc.close()

    return images


async def aexport_pdf_to_images(
    file_path: str,
    output_dir: Optional[str] = None,
    page_range: Optional[range] = None,
    img_width: int = DEFAULT_IMG_WIDTH,
    img_quality: int = DEFAULT_IMG_QUALITY,
) -> List[str]:
    """异步将 PDF 转换为图片"""
    return await asyncio.to_thread(
        export_pdf_to_images,
        file_path,
        output_dir,
        page_range,
        img_width,
        img_quality,
    )


# ========================
# 中间件类
# ========================


class PdfMiddleware:
    """
    PDF 解析中间件 - 为 Agent 提供 PDF 解析工具。

    参考：filesystem_middleware.py 的实现风格。

    该中间件为 Agent 提供以下工具：
    - parse_pdf: 解析单个 PDF 文件
    - batch_parse_pdfs: 批量解析 PDF 文件列表
    - parse_directory: 解析 PDF 目录
    - export_pdf_to_images: 将 PDF 转换为图片
    """

    def __init__(
        self,
        *,
        extract_mode: Literal["text", "image"] = PDF_MODE_IMAGE,
        page_range: Optional[range] = None,
        timeout: int = DEFAULT_TIMEOUT,
        img_width: int = DEFAULT_IMG_WIDTH,
        img_quality: int = DEFAULT_IMG_QUALITY,
    ):
        """
        Initialize PDF middleware.

        Args:
            extract_mode: 提取模式 - "text" 或 "image"
            page_range: 页码范围，None=全解析
            timeout: 超时时间（秒）
            img_width: 图片宽度
            img_quality: 图片质量
        """
        self.extract_mode = extract_mode
        self.page_range = page_range
        self.timeout = timeout
        self.img_width = img_width
        self.img_quality = img_quality

        # 工具函数
        self._tools = self._create_tools()

    def _create_tools(self) -> List[Dict[str, Any]]:
        """创建工具列表"""
        return [
            {
                "name": "parse_pdf",
                "description": "解析单个 PDF 文件，提取文本和图片内容",
                "function": parse_pdf_tool,
                "args_schema": {
                    "file_path": str,
                    "extract_mode": str,
                    "page_range": Optional[range],
                    "timeout": int,
                },
            },
            {
                "name": "batch_parse_pdfs",
                "description": "批量解析 PDF 文件列表",
                "function": batch_parse_pdfs_tool,
                "args_schema": {
                    "file_paths": List[str],
                    "extract_mode": str,
                    "page_range": Optional[range],
                    "max_workers": int,
                },
            },
            {
                "name": "parse_directory",
                "description": "解析 PDF 目录，提取所有 PDF 文件内容",
                "function": parse_directory_tool,
                "args_schema": {
                    "directory": str,
                    "extract_mode": str,
                    "page_range": Optional[range],
                    "timeout": int,
                },
            },
            {
                "name": "export_pdf_to_images",
                "description": "将 PDF 转换为图片",
                "function": export_pdf_to_images,
                "args_schema": {
                    "file_path": str,
                    "output_dir": Optional[str],
                    "page_range": Optional[range],
                    "img_width": int,
                    "img_quality": int,
                },
            },
            {
                "name": "extract_metadata",
                "description": "提取 PDF 元数据",
                "function": extract_metadata_from_pdf,
                "args_schema": {
                    "file_path": str,
                },
            },
        ]

    def get_tools(self) -> List[Dict[str, Any]]:
        """获取工具列表"""
        return self._tools

    async def get_async_tools(self) -> List[Dict[str, Any]]:
        """获取异步工具列表"""
        return self._tools


# ========================
# 导出
# ========================


__all__ = [
    # 工具函数
    "parse_pdf_tool",
    "aparse_pdf_tool",
    "batch_parse_pdfs_tool",
    "abatch_parse_pdfs_tool",
    "parse_directory_tool",
    "aparse_directory_tool",
    "export_pdf_to_images",
    "aexport_pdf_to_images",
    "extract_metadata_from_pdf",
    # 中间件类
    "PdfMiddleware",
    # 常量
    "PDMIME",
    "PDF_MODE_TEXT",
    "PDF_MODE_IMAGE",
    "DEFAULT_TIMEOUT",
    "DEFAULT_IMG_QUALITY",
    "DEFAULT_IMG_WIDTH",
    "DEFAULT_WORKERS",
    "DEFAULT_PAGE_LIMIT",
    # 工具函数
    "is_pdf",
    "validate_pdf_path",
    "validate_pdf_path_or_directory",
    "scan_pdf_directory",
]
