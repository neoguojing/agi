import requests
from pathlib import Path
from typing import Union, Optional,Dict,List,Iterator
from langchain_community.document_loaders.base import BaseLoader
from langchain_core.documents import Document
import os

class TikaExtractor(BaseLoader):
    """
    Tika 文件提取工具
    支持文本、HTML 或 JSON 元数据输出
    """
    COMMON_FIELDS = [
        "Content-Type", "Content-Encoding", "Content-Length",
        "ResourceName", "title", "Author", "creator",
        "Created", "dcterms:created", "Modified", "dcterms:modified",
        "language", "Keywords", "subject", "Producer", "Page-Count"
    ]

    def __init__(self,file_path: Union[str, Path], tika_url: str = os.getenv("TIKA_URL","http://localhost:9998")):
        self.tika_url = tika_url.rstrip("/")
        self.file_path = file_path

    def lazy_load(self) -> Iterator[Document]:
        """Load and return documents from the JSON file."""
        meta = self.extract_metadata(self.file_path)
        content = self.extract_text(self.file_path)
        yield Document(page_content=content,metadata=meta)
    # -----------------------------
    # 文本提取
    # -----------------------------
    def extract_text(
        self,
        file_path: Union[str, Path],
        output: str = "html",  # text / main / html
        accept: Optional[str] = None,
        html_to_text: bool = True,  # 是否把 html 转成纯文本
    ) -> str:
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"{file_path} not found")
        import chardet
        # 自动设置 Accept header
        if not accept:
            if output in ("text", "main"):
                accept = "text/plain; charset=UTF-8"
            elif output == "html":
                accept = "text/html; charset=UTF-8"
            else:
                accept = "text/plain; charset=UTF-8"

        url = f"{self.tika_url}/tika/{output}" if output not in ("text", "html") else f"{self.tika_url}/tika"

        with open(file_path, "rb") as f:
            headers = {
                "Accept": accept,
                "Content-Type": self._guess_content_type(file_path)
            }
            resp = requests.put(url, data=f, headers=headers, timeout=60)
        resp.raise_for_status()

        # 尝试 UTF-8 解码
        try:
            content = resp.content.decode("utf-8")
        except UnicodeDecodeError:
            detected = chardet.detect(resp.content)
            encoding = detected.get("encoding") or "utf-8"
            confidence = detected.get("confidence", 0)
            print(f"[WARN] UTF-8 解码失败，尝试 {encoding} (confidence={confidence:.2f})")
            content = resp.content.decode(encoding, errors="replace")

        # 如果是 html 输出且需要转换文本
        if output == "html" and html_to_text:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(content, "html.parser")
            # 获取纯文本，保留换行
            text = "\n".join([line.strip() for line in soup.stripped_strings])
            return text
        else:
            # 返回原始 html
            return content

    # -----------------------------
    # Metadata 获取
    # -----------------------------
    def extract_metadata(self, file_path: str, accept: str = "application/json") -> Dict[str, Optional[str]]:
        """
        获取文档 metadata 并抽取常用字段

        :param file_path: 文件路径
        :param accept: Accept header, 默认 JSON
        :return: dict，常用字段 -> 值，如果字段不存在返回 None
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"{file_path} not found")

        url = f"{self.tika_url}/meta"
        with open(file_path, "rb") as f:
            headers = {"Accept": accept, "Content-Type": self._guess_content_type(file_path)}
            resp = requests.put(url, data=f, headers=headers, timeout=60)
        resp.raise_for_status()

        if accept == "application/json":
            data = resp.json()
        else:
            # 其他格式先作为纯文本返回
            data = {}
            lines = resp.text.splitlines()
            for line in lines:
                if "," in line:
                    key, val = line.split(",", 1)
                    key = key.strip().strip('"')
                    val = val.strip().strip('"')
                    data[key] = val

        # 抽取常用字段
        result = {field: data.get(field) for field in self.COMMON_FIELDS}
        return result

    # -----------------------------
    # Content-Type 猜测
    # -----------------------------
    def _guess_content_type(self, file_path: Path) -> str:
        ext = file_path.suffix.lower()
        if ext in [".pdf"]:
            return "application/pdf"
        elif ext in [".doc", ".docx"]:
            return "application/msword"
        elif ext in [".xls", ".xlsx"]:
            return "application/vnd.ms-excel"
        elif ext in [".ppt", ".pptx"]:
            return "application/vnd.ms-powerpoint"
        else:
            return "application/octet-stream"

