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
from langchain.prompts import ChatPromptTemplate
from concurrent.futures import ThreadPoolExecutor
import asyncio

nlp = TextProcessor()
collection_manager = CollectionManager(data_path=CACHE_DIR,embedding=TaskFactory.get_embedding())

doc_clean_prompt = """
    You are a document cleaning and restructuring assistant.

    Your task is to take a raw fragment of a document that may be partially structured or messy (e.g., broken paragraphs, misaligned headings, noise like page numbers or repeated footers), and transform it into a clean, well-structured, and semantically coherent version.

    ## Instructions:
    - Remove any noise such as page numbers, footers, headers, URLs, or repeated titles.
    - Fix broken paragraphs: merge lines if they belong to the same paragraph.
    - Preserve and properly format section headings (e.g., convert “1. Introduction” into a Markdown-style heading like “## 1. Introduction”).
    - Normalize bullet points or numbered lists into consistent Markdown list format.
    - Retain meaningful formatting like bold text indicators (e.g., “**important**”) if visible.
    - Preserve table content as best as possible in plain text if tables are embedded.
    - Keep the semantic meaning intact. Do not hallucinate or add new information.
    - Do not summarize — return a cleaned and well-structured version of the original fragment.

    Output must be in valid, readable Markdown format only.
    Example:
    {example}
"""

intend_understand_template = ChatPromptTemplate.from_messages(
    [
        ("system",doc_clean_prompt),
        ("human", "## Input:\n {text}")
    ]
)

clean_chain = intend_understand_template | TaskFactory.get_small_llm()

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

    if not state["db_documents"]:
        return {}
    
    semaphore = asyncio.Semaphore(2)
    # 单独处理第一个文档，生成 example
    first_doc = state["db_documents"][0]
    result = await clean_chain.ainvoke({"text": first_doc.page_content + " /no_think","example":""})
    first_doc.page_content = result.content
    example = first_doc.page_content

    # 用 example 处理剩下的文档
    async def _clean_text(doc: Document):
        result = await clean_chain.ainvoke({
            "text": doc.page_content + " /no_think",
            "example": example
        })
        doc.page_content = result.content
        log.info(doc.page_content)
        return doc

    async def limited_clean_text(doc: Document):
        async with semaphore:
            return await _clean_text(doc)

    # 处理剩下的文档
    documents = await asyncio.gather(
        *(limited_clean_text(doc) for doc in state["db_documents"][1:])
    )

    # 加上第一个处理过的文档
    return {"db_documents": [first_doc] + documents}


async def doc_embding_node(state: State, config: RunnableConfig):
    model = TaskFactory.get_embedding()
    def embed_doc(doc: Document):
        return model.embed_query(doc.page_content)
    with ThreadPoolExecutor() as executor:
        embds = list(executor.map(embed_doc, state["db_documents"]))
    log.info(f"embding {len(embds)} embding")
    return {"embds":embds}

async def doc_keywords_node(state: State, config: RunnableConfig):
    documents = state["db_documents"]

    texts = [doc.page_content for doc in documents]
    keywords_of_all = nlp.batch_process(texts)

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
    log.info(f"embds={len(state['embds'])}")

    state["clusters"] = []
    state["db_documents"] = []
    state["embds"] = []
    return state

# graph
checkpointer = MemorySaver()

doc_graph_builder = StateGraph(State)

doc_graph_builder.add_node("load", file_loader_node)
doc_graph_builder.add_node("split", doc_split_node)
doc_graph_builder.add_node("clean", doc_clean_node)
doc_graph_builder.add_node("embding", doc_embding_node)
doc_graph_builder.add_node("train", cluster_train_node)
doc_graph_builder.add_node("keyword", doc_keywords_node)
doc_graph_builder.add_node("store_index", store_index_node)
doc_graph_builder.add_node("store_docs", store_node)
doc_graph_builder.add_node("last", last_node)



doc_graph_builder.add_edge(START, "load")

doc_graph_builder.add_edge("load","split")
doc_graph_builder.add_edge("split", "clean")

# 可并行处理
doc_graph_builder.add_edge("clean", "embding")
doc_graph_builder.add_edge("clean", "keyword")
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
