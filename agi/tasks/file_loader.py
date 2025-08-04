from langchain_community.document_loaders import (
    BSHTMLLoader,
    CSVLoader,
    Docx2txtLoader,
    OutlookMessageLoader,
    PyPDFLoader,
    PDFPlumberLoader,
    TextLoader,
    UnstructuredEPubLoader,
    UnstructuredExcelLoader,
    UnstructuredMarkdownLoader,
    UnstructuredPowerPointLoader,
    UnstructuredRSTLoader,
    UnstructuredXMLLoader,
    JSONLoader,
    WebBaseLoader,
    YoutubeLoader,
)

from typing import Any,List,Dict,Iterator, Optional, Sequence, Union, Tuple, Set
import validators
import socket
import urllib.parse

def get_file_loader(file_path: str, file_content_type: str = None):
    from pathlib import Path

    file_ext = Path(file_path).suffix.lower().lstrip(".")
    known_type = True

    known_source_ext = {
        "go", "py", "java", "sh", "bat", "ps1", "cmd", "js", "ts", "css", "cpp", "hpp", "h", "c", "cs",
        "sql", "log", "ini", "pl", "pm", "r", "dart", "dockerfile", "env", "php", "hs", "hsc", "lua",
        "nginxconf", "conf", "m", "mm", "plsql", "perl", "rb", "rs", "db2", "scala", "bash", "swift",
        "vue", "svelte", "msg", "ex", "exs", "erl", "tsx", "jsx", "lhs"
    }

    # Extension-based loader map
    ext_loader_map = {
        "pdf": lambda: PDFPlumberLoader(file_path,dedupe=True, extract_images=False),
        "csv": lambda: CSVLoader(file_path),
        "rst": lambda: UnstructuredRSTLoader(file_path, mode="elements"),
        "xml": lambda: UnstructuredXMLLoader(file_path),
        "html": lambda: BSHTMLLoader(file_path, open_encoding="unicode_escape"),
        "htm": lambda: BSHTMLLoader(file_path, open_encoding="unicode_escape"),
        "md": lambda: UnstructuredMarkdownLoader(file_path),
        "doc": lambda: Docx2txtLoader(file_path),
        "docx": lambda: Docx2txtLoader(file_path),
        "xls": lambda: UnstructuredExcelLoader(file_path),
        "xlsx": lambda: UnstructuredExcelLoader(file_path),
        "ppt": lambda: UnstructuredPowerPointLoader(file_path),
        "pptx": lambda: UnstructuredPowerPointLoader(file_path),
        "msg": lambda: OutlookMessageLoader(file_path),
        "json": lambda: JSONLoader(file_path),
    }

    # MIME-type based loader override (for ambiguous extensions)
    mime_loader_map = {
        "application/epub+zip": lambda: UnstructuredEPubLoader(file_path),
        "application/vnd.ms-excel": lambda: UnstructuredExcelLoader(file_path),
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": lambda: UnstructuredExcelLoader(file_path),
        "application/vnd.ms-powerpoint": lambda: UnstructuredPowerPointLoader(file_path),
        "application/vnd.openxmlformats-officedocument.presentationml.presentation": lambda: UnstructuredPowerPointLoader(file_path),
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document": lambda: Docx2txtLoader(file_path),
    }

    # Loader selection
    loader = None

    if file_content_type in mime_loader_map:
        loader = mime_loader_map[file_content_type]()
    elif file_ext in ext_loader_map:
        loader = ext_loader_map[file_ext]()
    elif file_ext in known_source_ext or (file_content_type and file_content_type.startswith("text/")):
        loader = TextLoader(file_path, autodetect_encoding=True)
    else:
        loader = TextLoader(file_path, autodetect_encoding=True)
        known_type = False

    return loader, known_type

# 自定义异常
class InvalidURLException(ValueError):
    pass

def resolve_hostname(hostname: str) -> Tuple[list, list]:
    addr_info = socket.getaddrinfo(hostname, None)
    ipv4_addresses = [info[4][0] for info in addr_info if info[0] == socket.AF_INET]
    ipv6_addresses = [info[4][0] for info in addr_info if info[0] == socket.AF_INET6]
    return ipv4_addresses, ipv6_addresses

def validate_url(url: Union[str, Sequence[str]]) -> bool:
    if isinstance(url, str):
        if not validators.url(url):
            raise InvalidURLException("Invalid URL format.")
        parsed_url = urllib.parse.urlparse(url)
        ipv4_addresses, ipv6_addresses = resolve_hostname(parsed_url.hostname)
        for ip in ipv4_addresses:
            if validators.ipv4(ip, private=True):
                raise InvalidURLException("Private IP detected.")
        for ip in ipv6_addresses:
            if validators.ipv6(ip, private=True):
                raise InvalidURLException("Private IP detected.")
        return True
    elif isinstance(url, Sequence):
        return all(validate_url(u) for u in url)
    return False

def get_web_loader(url: Union[str, Sequence[str]], verify_ssl: bool = True):
    if not validate_url(url):
        raise InvalidURLException("URL is not valid.")
    from agi.utils.scrape import WebScraper
    return WebScraper(web_paths=url)

def get_youtube_loader(url: str):
    return YoutubeLoader.from_youtube_url(
        url,
        add_video_info=True,
        language='en',
        translation=None
    )

