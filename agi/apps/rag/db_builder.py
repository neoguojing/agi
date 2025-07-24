from agi.tasks.define import State,InputType,Feature
from agi.tasks.task_factory import (
    TaskFactory
)
from agi.utils.nlp import TextProcessor
from langchain_core.runnables import (
    RunnableConfig
)
from langgraph.graph import END, StateGraph, START
from langgraph.checkpoint.memory import MemorySaver
from agi.apps.rag.file_loader import get_file_loader,get_web_loader,get_youtube_loader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import re
import unicodedata
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor

nlp = TextProcessor()

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
        loader = get_file_loader(file_path)

    if loader:
        documents = loader.lazy_load()
        state["db_documents"] = documents

    return state

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
    state["db_documents"] = await text_splitter.atransform_documents(state["db_documents"])
    return state

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
        state["db_documents"] = list(executor.map(_clean_text, state["db_documents"]))

    return state

async def doc_filter_node(state: State, config: RunnableConfig):
    def filter_doc(doc: Document):
        return nlp.remove_stopwords(doc.page_content)
    with ThreadPoolExecutor() as executor:
        state["filted_texts"] = list(executor.map(filter_doc, state["db_documents"]))
    return state

async def doc_embding_node(state: State, config: RunnableConfig):
    model = TaskFactory.get_embedding()
    def embed_doc(doc: Document):
        return model.embed_query(doc.page_content)
    with ThreadPoolExecutor() as executor:
        state["embds"] = list(executor.map(embed_doc, state["filted_texts"]))
    return state

async def doc_keywords_node(state: State, config: RunnableConfig):
    keywords_of_all = nlp.batch_process(state["filted_texts"])
    for i, keywords in enumerate(keywords_of_all):
        state["db_documents"][i].metadata["keywords"] = keywords
    return state

# graph
checkpointer = MemorySaver()

doc_graph_builder = StateGraph(State)

doc_graph_builder.add_node("load", file_loader_node)
doc_graph_builder.add_node("split", doc_split_node)
doc_graph_builder.add_node("clean", doc_clean_node)
doc_graph_builder.add_node("filterd", doc_filter_node)
doc_graph_builder.add_node("embding", doc_embding_node)
doc_graph_builder.add_node("cluster", doc_list_node)
doc_graph_builder.add_node("summary", TaskFactory.create_task(TASK_LLM_WITH_HISTORY))
doc_graph_builder.add_node("keyword", doc_keywords_node)
doc_graph_builder.add_node("store_index", TaskFactory.create_task(TASK_WEB_SEARCH))
doc_graph_builder.add_node("store_docs", TaskFactory.create_task(TASK_WEB_SEARCH))


doc_graph_builder.add_edge(START, "load")

doc_graph_builder.add_edge("load","split")
doc_graph_builder.add_edge("split", "clean")
doc_graph_builder.add_edge("clean", "filterd")
# 可并行处理
doc_graph_builder.add_edge("filterd", "embding")
doc_graph_builder.add_edge("filterd", "keyword")
doc_graph_builder.add_edge("embding", "cluster")
doc_graph_builder.add_edge("keyword", "cluster")

doc_graph_builder.add_edge("cluster", "summary")
doc_graph_builder.add_edge("summary", "store_index")
doc_graph_builder.add_edge("store_index", "store_docs")
doc_graph_builder.add_edge("store_docs", END)

db_graph = doc_graph_builder.compile(checkpointer=checkpointer,name="doc_db")
doc_db_as_subgraph = doc_graph_builder.compile(name="doc_db")
# graph_print(db_graph)
