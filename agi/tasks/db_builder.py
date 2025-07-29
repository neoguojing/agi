from agi.tasks.define import State,InputType,Feature
from agi.tasks.task_factory import (
    TaskFactory
)
from agi.tasks.cluster import TextClusterer
from agi.utils.nlp import TextProcessor
from agi.tasks.vectore_store import CollectionManager
from langchain_core.runnables import (
    RunnableConfig,
    RunnableLambda
)
from agi.config import log,CACHE_DIR
from agi.tasks.utils import graph_print
from langgraph.graph import END, StateGraph, START
from langgraph.checkpoint.memory import MemorySaver
from agi.tasks.file_loader import get_file_loader,get_web_loader,get_youtube_loader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import re
import unicodedata
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor

nlp = TextProcessor()
cluster = TextClusterer()
collection_manager = CollectionManager(data_path=CACHE_DIR,embedding=TaskFactory.get_embedding())

# 🚀 统一入口：异步加载节点
async def file_loader_node(state: State, config: RunnableConfig):
    loader = None
    if "url" in state:
        url = state["url"]
        if "youtube.com" in url or "youtu.be" in url:
            loader = get_youtube_loader(url)
        else:
            loader = get_web_loader(url)
    elif "file_path" in state:
        file_path = state["file_path"]
        loader,_ = get_file_loader(file_path)

    if loader:
        documents = loader.load()
        log.info(f"load {len(documents)} pages,{documents[0]}")
        return {"db_documents": documents}

    return {}

async def doc_split_node(state: State, config: RunnableConfig):
    text_splitter = RecursiveCharacterTextSplitter(separators=[
                                                    "\n\n",
                                                    "\n",
                                                    " ",
                                                    ".",
                                                    ",",
                                                    "\u200b",  # Zero-width space
                                                    "\uff0c",  # Fullwidth comma
                                                    "\u3001",  # Ideographic comma
                                                    "\uff0e",  # Fullwidth full stop
                                                    "\u3002",  # Ideographic full stop
                                                ],
                                                chunk_size=3000, chunk_overlap=300,add_start_index=True)

    documents = await text_splitter.atransform_documents(state["db_documents"])
    log.info(f"split {len(documents)} docs,{documents[0]}")

    return {"db_documents": documents}

async def doc_clean_node(state: State, config: RunnableConfig):
    def _clean_text(doc: Document) -> Document:
        """文本清洗主流程：
        1. 去除 HTML 标签
        2. unicode 标准化
        3. 全角转半角
        4. 去除特殊符号
        5. 合并多空白字符
        6. 去除首尾空格
        """

        # 去除 HTML 标签（如 <p>, <div>）
        text = BeautifulSoup(doc.page_content, "html.parser").get_text()

        # Unicode 标准化（兼容表情、异体字等）
        text = unicodedata.normalize("NFKC", text)

        # 全角转半角（如：中文输入法下的符号）
        def fullwidth_to_halfwidth(char):
            code = ord(char)
            if code == 0x3000:
                return ' '
            elif 0xFF01 <= code <= 0xFF5E:
                return chr(code - 0xFEE0)
            return char

        text = ''.join(fullwidth_to_halfwidth(c) for c in text)

        # 去除特殊字符（保留中英文、数字和常用标点）
        text = re.sub(r"[^\u4e00-\u9fa5a-zA-Z0-9\s.,!?;:，。！？；：]", '', text)

        # 合并多空格为一个空格
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
            
        doc.page_content = text
        # 去除首尾空格
        return doc
    with ThreadPoolExecutor() as executor:
        documents = list(executor.map(_clean_text, state["db_documents"]))

    return {"db_documents": documents}

async def doc_filter_node(state: State, config: RunnableConfig):
    def filter_doc(doc: Document):
        return nlp.remove_stopwords(doc.page_content)
    with ThreadPoolExecutor() as executor:
        filted_texts = list(executor.map(filter_doc, state["db_documents"]))
    log.info(f"filted {len(filted_texts)} texts")
    return {"filted_texts":filted_texts}

async def doc_embding_node(state: State, config: RunnableConfig):
    model = TaskFactory.get_embedding()
    def embed_doc(text: str):
        return model.embed_query(text)
    with ThreadPoolExecutor() as executor:
        embds = list(executor.map(embed_doc, state["filted_texts"]))
    log.info(f"embding {len(embds)} embding")
    return {"embds":embds}

async def doc_keywords_node(state: State, config: RunnableConfig):
    keywords_of_all = nlp.batch_process(state["filted_texts"])
    documents = state["db_documents"]
    for i, keywords in enumerate(keywords_of_all):
        documents[i].metadata["keywords"] = keywords
    log.info(f"keywords {len(documents)}")
    return {"db_documents": documents}

async def cluster_node(state: State, config: RunnableConfig):
    clusters = cluster.cluster(state["db_documents"],state["filted_texts"],state["embds"] )
    return {"clusters":clusters}

async def store_index_node(state: State, config: RunnableConfig):
    user_id = state.get("user_id")
    if not user_id and config:
        user_id = config.get("configurable").get("user_id","default")

    collection_manager.add_documents(
        documents=state["clusters"],
        collection_name="index",
        tenant=user_id
    )
    return {}

async def store_node(state: State, config: RunnableConfig):
    collection_name = state.get("collection_name")
    if not collection_name and config:
        collection_name = config.get("configurable").get("collection_name","default")

    user_id = state.get("user_id")
    if not user_id and config:
        user_id = config.get("configurable").get("user_id","default")

    collection_manager.add_documents(
        documents=state["db_documents"],
        collection_name=collection_name,
        embeddings=state["embds"],
        tenant=user_id
    )
    return {}

async def last_node(state: State, config: RunnableConfig):
    print(state["clusters"])
    print(len(state["db_documents"]))
    print(len(state["filted_texts"]))
    print(len(state["embds"]))

    state["clusters"] = []
    state["db_documents"] = []
    state["filted_texts"] = []
    state["embds"] = []
    return state

# graph
checkpointer = MemorySaver()

doc_graph_builder = StateGraph(State)

doc_graph_builder.add_node("load", file_loader_node)
doc_graph_builder.add_node("split", doc_split_node)
doc_graph_builder.add_node("clean", doc_clean_node)
doc_graph_builder.add_node("filterd", doc_filter_node)
doc_graph_builder.add_node("embding", doc_embding_node)
doc_graph_builder.add_node("cluster", cluster_node)
doc_graph_builder.add_node("keyword", doc_keywords_node)
doc_graph_builder.add_node("store_index", store_index_node)
doc_graph_builder.add_node("store_docs", store_node)
doc_graph_builder.add_node("last", last_node)



doc_graph_builder.add_edge(START, "load")

doc_graph_builder.add_edge("load","split")
doc_graph_builder.add_edge("split", "clean")
doc_graph_builder.add_edge("clean", "filterd")
# 可并行处理
doc_graph_builder.add_edge("filterd", "embding")
doc_graph_builder.add_edge("filterd", "keyword")
doc_graph_builder.add_edge("embding", "cluster")
doc_graph_builder.add_edge("keyword", "cluster")
doc_graph_builder.add_edge("cluster", "store_index")
doc_graph_builder.add_edge("cluster", "store_docs")
doc_graph_builder.add_edge("store_index", "last")
doc_graph_builder.add_edge("store_docs", "last")

doc_graph_builder.add_edge("last", END)


db_graph = doc_graph_builder.compile(checkpointer=checkpointer,name="doc_db")
doc_db_as_subgraph = doc_graph_builder.compile(name="doc_db")
graph_print(db_graph)
