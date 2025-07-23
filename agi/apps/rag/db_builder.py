from langchain_community.document_loaders import (
    BSHTMLLoader,
    CSVLoader,
    Docx2txtLoader,
    OutlookMessageLoader,
    PyPDFLoader,
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
import hdbscan
from sklearn.preprocessing import StandardScaler

from agi.tasks.define import State,InputType,Feature
from agi.tasks.task_factory import (
    TaskFactory,
    TASK_DOC_CHAT,
    TASK_RAG,
    TASK_WEB_SEARCH,
    TASK_LLM_WITH_HISTORY
)
from langchain_core.runnables import (
    RunnableLambda,
    RunnableConfig
)
from langchain_core.messages import SystemMessage
from langchain.prompts import ChatPromptTemplate
from langgraph.types import StreamWriter
from langgraph.graph import END, StateGraph, START
from langgraph.checkpoint.memory import MemorySaver
from agi.tasks.retriever import FilterType,SourceType
from agi.tasks.llm_app import build_citations
from agi.tasks.utils import get_last_message_text,split_think_content,graph_print
from agi.config import log
import asyncio
import json

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
        "pdf": lambda: PyPDFLoader(file_path, extract_images=False),
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
# 🚀 统一入口：异步加载节点
async def file_loader_node(state: State, config: RunnableConfig):
    loader = None

    if "url" in state:
        url = state["url"]
        if "youtube.com" in url or "youtu.be" in url:
            loader = get_youtube_loader(url)
        else:
            loader = get_web_loader(url)
    elif "file_path" in state and "filename" in state:
        file_path = state["file_path"]
        content_type = state.get("file_content_type")
        loader = get_file_loader(file_path, content_type)

    if loader:
        documents = loader.load()
        state["docs"] = documents

    return state

def cluster_texts(texts: List[str], 
                  min_cluster_size: int = 5,
                  min_samples: int = 1) -> Dict[int, List[str]]:
    """
    基于HDBSCAN对文本列表进行聚类。

    Args:
        texts: 需要聚类的文本列表
        model_name: SentenceTransformer模型名
        min_cluster_size: HDBSCAN参数，最小聚类大小
        min_samples: HDBSCAN参数，密度阈值

    Returns:
        clusters: dict，key为聚类标签，value为该类文本列表。label=-1代表噪声点。
    """

    # 1. 文本向量化
    model = TaskFactory.get_embedding()
    embeddings = model.aembed_documents(texts)

    # 2. 标准化
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)

    # 3. HDBSCAN 聚类
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
    labels = clusterer.fit_predict(embeddings_scaled)

    # 4. 聚类结果整理
    clusters = {}
    for label, text in zip(labels, texts):
        clusters.setdefault(label, []).append(text)

    return clusters
   
async def route(state: State):
    # 状态初始化
    state["context"] = None
    state["docs"] = None
    state["citations"] = None

    feature = state.get("feature","")
    if feature == Feature.RAG:
        return await rag_auto_route(state)
    elif feature == Feature.WEB:
        return "web"
    elif state.get("collection_names"):
        return await rag_auto_route(state)


# graph
checkpointer = MemorySaver()

doc_graph_builder = StateGraph(State)

doc_graph_builder.add_node("load", file_loader_node)
doc_graph_builder.add_node("split", doc_rerank_node)
doc_graph_builder.add_node("embding", TaskFactory.get_embedding)
doc_graph_builder.add_node("cluster", doc_list_node)
doc_graph_builder.add_node("summary", TaskFactory.create_task(TASK_LLM_WITH_HISTORY))
doc_graph_builder.add_node("store", TaskFactory.create_task(TASK_WEB_SEARCH))


doc_graph_builder.add_edge(START, "load")

doc_graph_builder.add_edge("load","split")
doc_graph_builder.add_edge("split", "embding")
doc_graph_builder.add_edge("embding", "cluster")
doc_graph_builder.add_edge("cluster", "summary")

doc_graph_builder.add_edge("store", END)

db_graph = doc_graph_builder.compile(checkpointer=checkpointer,name="doc_db")
rag_as_subgraph = doc_graph_builder.compile(name="doc_db")
graph_print(db_graph)
