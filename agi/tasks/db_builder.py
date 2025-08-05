from agi.tasks.define import State,InputType,Feature
from agi.tasks.task_factory import (
    TaskFactory
)
from agi.tasks.cluster import train
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
        log.info(f"load {len(documents)} pages")
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
    log.info(f"split {len(documents)} docs")

    return {"db_documents": documents}

async def doc_clean_node(state: State, config: RunnableConfig):
    def _clean_text(doc: Document) -> Document:
        """文本清洗主流程（修复版）：
        1. 去除 HTML 标签
        2. unicode 标准化 (NFKC)
        3. 全角转半角
        4. 温和地去除无用字符，保留重要标点和符号
        5. 合并多余空白字符
        6. 去除首尾空格
        """

        # 1. 去除 HTML 标签
        text = BeautifulSoup(doc.page_content, "html.parser").get_text(separator=' ')

        # 2. Unicode 标准化（兼容表情、异体字等）
        text = unicodedata.normalize("NFKC", text)

        # 3. 全角转半角
        def fullwidth_to_halfwidth(char: str) -> str:
            code = ord(char)
            if code == 0x3000:  # 全角空格
                return ' '
            elif 0xFF01 <= code <= 0xFF5E:  # 全角字符（除空格）
                return chr(code - 0xFEE0)
            return char
        text = ''.join(fullwidth_to_halfwidth(c) for c in text)

        # 4. 温和地去除特殊字符（修复核心）
        # 我们扩展了保留字符的范围，加入了各种括号、引号、以及常见的数学和特殊符号
        # 注意：这里将中英文标点符号统一在半角状态下处理
        # 保留：中文、英文、数字、空格
        # 保留：常用标点 .,!?;:
        # 保留：各种括号 ()[]{}
        # 保留：各种引号 "'`
        # 保留：常见数学与特殊符号 +-*/=<>@#$%&_`
        # 如果还需要保留其他字符（如日文、韩文），可以在此基础上继续添加
        # \u4e00-\u9fa5  (中文字符)
        # a-zA-Z0-9   (英文字母和数字)
        # \s           (空白字符)
        # .,!?;:'"`()\[\]{} (英文标点、引号、括号)
        # +-*/=<>@#$%&_  (数学及特殊符号)
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s.,!?;:\'"`()\[\]{}<>+\-*/=@#$%&_]', '', text, flags=re.UNICODE)


        # 5. 合并多余空白字符（包括空格、换行、制表符）
        text = re.sub(r'\s+', ' ', text)
        
        # 6. 去除首尾空格
        text = text.strip()
            
        doc.page_content = text
        return doc
    with ThreadPoolExecutor() as executor:
        documents = list(executor.map(_clean_text, state["db_documents"]))

    return {"db_documents": documents}

async def doc_filter_node(state: State, config: RunnableConfig):
    def filter_doc(doc: Document):
        return nlp.remove_stopwords(doc.page_content)
        # return doc.page_content
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

async def cluster_train_node(state: State, config: RunnableConfig):
    clusters = train(state["db_documents"],state["embds"])
    return {"clusters":clusters}

async def store_index_node(state: State, config: RunnableConfig):
    user_id = state.get("user_id")
    if not user_id and config:
        user_id = config.get("configurable").get("user_id","default")

    await collection_manager.add_documents(
        documents=state["clusters"],
        collection_name="index",
        tenant=user_id
    )
    log.info("store_index_node done")
    return {}

async def store_node(state: State, config: RunnableConfig):
    collection_name = state.get("collection_name")
    if not collection_name and config:
        collection_name = config.get("configurable").get("collection_name","default")

    user_id = state.get("user_id")
    if not user_id and config:
        user_id = config.get("configurable").get("user_id","default")

    await collection_manager.add_documents(
        documents=state["db_documents"],
        collection_name=collection_name,
        embeddings=state["embds"],
        tenant=user_id
    )
    log.info(f"store_node {collection_name} done")

    return {}

async def last_node(state: State, config: RunnableConfig):
    log.info(f"clusters={len(state['clusters'])}")
    log.info(f"db_documents={len(state['db_documents'])}")
    log.info(f"filted_texts={len(state['filted_texts'])}")
    log.info(f"embds={len(state['embds'])}")

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
doc_graph_builder.add_node("train", cluster_train_node)
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
doc_graph_builder.add_edge("embding", "train")
doc_graph_builder.add_edge("keyword", "train")
doc_graph_builder.add_edge("train", "store_index")
doc_graph_builder.add_edge("train", "store_docs")
doc_graph_builder.add_edge("store_index", "last")
doc_graph_builder.add_edge("store_docs", "last")
doc_graph_builder.add_edge("last", END)


db_graph = doc_graph_builder.compile(checkpointer=checkpointer,name="doc_db")
doc_db_as_subgraph = doc_graph_builder.compile(name="doc_db")
graph_print(db_graph)
