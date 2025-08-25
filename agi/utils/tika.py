import requests
from pathlib import Path
from typing import Union, Optional,Dict

class TikaExtractor:
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

    def __init__(self, tika_url: str = "http://localhost:9998"):
        self.tika_url = tika_url.rstrip("/")

    # -----------------------------
    # 文本提取
    # -----------------------------
    def extract_text(
        self,
        file_path: Union[str, Path],
        output: str = "text",  # text / main / html
        accept: Optional[str] = None
    ) -> str:
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"{file_path} not found")

        # 自动设置 Accept header
        if not accept:
            if output in ("text", "main"):
                accept = "text/plain"
            elif output == "html":
                accept = "text/html"
            else:
                accept = "text/plain"

        url = f"{self.tika_url}/tika/{output}" if output != "html" and output != "text" else f"{self.tika_url}/tika"
        with open(file_path, "rb") as f:
            headers = {"Accept": accept, "Content-Type": self._guess_content_type(file_path)}
            resp = requests.put(url, data=f, headers=headers, timeout=60)
        resp.raise_for_status()
        return resp.text

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

