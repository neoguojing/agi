from agi.tasks.define import State,InputType,Feature
from agi.tasks.runtime.task_factory import (
    TaskFactory,
    TASK_DOC_CHAT,
    TASK_LLM_WITH_HISTORY
)
from agi.tasks.prompt import DEFAULT_SEARCH_PROMPT
from langchain_core.runnables import (
    RunnableLambda,
    RunnableConfig
)
from langchain_core.messages import SystemMessage
from langchain.prompts import ChatPromptTemplate
from langgraph.types import StreamWriter
from langgraph.graph import END, StateGraph, START
from langgraph.checkpoint.memory import MemorySaver
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.retrievers.web_research import QuestionListOutputParser
from agi.tasks.chat.chains import build_citations
from agi.tasks.utils import get_last_message_text,split_think_content,graph_print,refine_last_message_runnable
from agi.config import log,CACHE_DIR
from agi.tasks.vectore_store import CollectionManager
from agi.llms.rerank import rerank_with_batching
from agi.utils.search_engine import SearchEngineSelector
import json
import traceback
from datetime import datetime
from typing import Dict,List,Union
from langdetect import detect
from langchain_core.documents import Document

intend_understand_prompt = '''
    You are a router that classifies user queries for a retrieval-augmented generation (RAG) system. 

    You must decide whether the user's intent is to:
    - **summary**: Summarize existing text, document content, or provide a high-level overview without retrieving external knowledge.
    - **rag**: Retrieve knowledge from external documents (e.g., manuals, knowledge bases, articles) based on specific topics, named entities, or document structure.

    [Instructions]
    Classify the user's query strictly as either `summary` or `rag`. Use the following principles:

    1. If the query asks to “summarize something specific” (like a chapter, a policy, a person, a concept), it is `rag`.
    2. If the query is vague (e.g., “give me a summary”, “summarize this”, “概括一下”), it is `summary`.
    3. If the query contains a named entity (e.g., Elon Musk, GDPR, 联邦学习, ChatGPT), a document structure (e.g., 第三章, section 5), or is a question (e.g., “What is...”, “如何...”), it is `rag`.
    4. The query may be in English or Chinese.

    [Examples]
    Q: 简单介绍一下联邦学习是做什么的  
    A: rag

    Q: 总结一下第5章的主要内容  
    A: rag

    Q: 简要概括以下内容  
    A: summary

    Q: Give me a summary.  
    A: summary

    Q: What is the role of data encryption in secure communication?  
    A: rag

    Q: Summarize this document.  
    A: summary

    Q: 请概括下面这篇文章的核心观点  
    A: summary

    Q: Explain what Elon Musk said about AGI at the conference.  
    A: rag

    [Now classify:]
'''
intend_understand_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            intend_understand_prompt
        ),
        ("placeholder", "{messages}")
    ]
)

async def intend_understand_modify_state_messages(state: State):
    # 可能会存在重复的系统消息需要去掉
    filter_messages = []
    for message in state["messages"]:
        if isinstance(message,SystemMessage):
            continue
        # 修正请求的类型，否则openapi会报错
        if not isinstance(message.content,str):
             message.content = json.dumps(message.content)
        filter_messages.append(message)
    render_result = await intend_understand_template.ainvoke({"messages": [filter_messages[-1]]})
    log.info(f"intend_understand_modify_state_messages:{render_result}")
    return render_result.to_messages()


intend_understand__modify_state_messages_runnable = RunnableLambda(intend_understand_modify_state_messages)

intend_understand_chain = intend_understand__modify_state_messages_runnable | TaskFactory.get_llm() 

search_question_generate_chain = DEFAULT_SEARCH_PROMPT | TaskFactory.get_llm() | refine_last_message_runnable | QuestionListOutputParser()

collection_manager = CollectionManager(data_path=CACHE_DIR,embedding=TaskFactory.get_embedding())

search_engines = SearchEngineSelector()

doc_chain = TaskFactory.create_task(TASK_DOC_CHAT)

def get_clusterid_collection_pair(docs: Union[List[Document], Dict[str, List[Document]]]) -> set:
    pairs = set()
    
    if isinstance(docs, dict):
        # 处理 {question: list[Document]} 的情况
        for doc_list in docs.values():
            for doc in doc_list:
                cid = doc.metadata.get("cluster_id")
                cname = doc.metadata.get("collection_name")
                if cid is not None and cname is not None:
                    pairs.add((cid,cname))
    elif isinstance(docs, list):
        # 处理 list[Document] 的情况
        for doc in docs:
            cid = doc.metadata.get("cluster_id")
            cname = doc.metadata.get("collection_name")
            if cid is not None and cname is not None:
                pairs.add((cid,cname))
    else:
        raise TypeError(f"Unsupported type for docs: {type(docs)}")
    
    return pairs


# 产生新的问题
def refine_query(feature:str,query: str):
    language = ""
    if feature == "web":
        language = "English"
    else:
        lang = detect(query)
        if lang in ["zh","zh-cn","sw","ko","jp"]:
            language = "Chinese"
        else:
            language = "English"

    questions = search_question_generate_chain.invoke({"date":datetime.now().date(),"text":query,"results_num":3,"language":language})
    log.info(f"questions:{questions}")
    return questions

# NODE
# 文档对话
async def doc_chat_node(state: State,config: RunnableConfig,writer: StreamWriter):
    try:
        docs = state.get("docs")
        log.info(f"doc_chat_node:{len(docs)}")

        citations = build_citations(docs)
        if citations:
            writer({"citations":citations})

        log.info(f"doc_chat_node: citations={len(citations)}")
        result = await doc_chain.ainvoke(state,config=config)
        result["citations"] = citations
        result["docs"] = None
        result["docs_map"] = None
        result["index_search_result"] = None

        return result
    except Exception as e:
        log.error(f"doc_chat_node: {e}")
        print(traceback.format_exc())
        return {}

async def doc_rerank_node(state: State,config: RunnableConfig):
    try:
        docs = []
        docs_map = state.get("docs_map")
        for question,doc_list in docs_map.items():
            parts = await rerank_with_batching(question,doc_list)
            docs.extend(parts)

        # 排序（score 越大越相关）
        sorted_docs = sorted(
            docs, 
            key=lambda d: d.metadata.get("score", float("-inf")), 
            reverse=True
        )
        
        # 取前 3
        topk_docs = sorted_docs[:3]
        
        log.info(f"doc_rerank_node 3:: {topk_docs}")
        
        return {"docs": topk_docs}
    except Exception as e:
        log.error(f"doc_rerank_node: {e}")
        print(traceback.format_exc())
        return {}
    


# 获取指定文件的索引文件
async def doc_summary_node(state: State,config: RunnableConfig):
    try:
        tenant = state.get("user_id")
        source = state.get("file_path")
        state["docs"] = []
        docs = collection_manager.get_documents("index",source=source,tenant=tenant)
        log.info(f"doc_list_node:{len(docs)}")
        return {"docs":docs}
    except Exception as e:
        log.error(f"doc_summary_node: {e}")
        print(traceback.format_exc())
        return {}

async def web_search_node(state: State,config: RunnableConfig):
    try:
        feature = state.get("feature")
        question = get_last_message_text(state)
        questions = refine_query(feature,query=question)
        if not questions:
            return {}
        
        docs_map = {}
        raw_results_map = await search_engines.batch_search(questions)
        for q,raw_search_results in raw_results_map.items():
            raw_docs = []
            for source in raw_search_results:
                snippet = source.get("snippet", "")
                link = source.get("link","")
                if not snippet.strip() and not link:  # 丢弃 snippet 和 url 为空的
                    continue
                raw_docs.append(
                    Document(
                    page_content = f'{source.get("date", "")}\n{source.get("title", "")}\n{source.get("snippet")}',
                        metadata={"source": source.get("source"), "link": source.get("link"),"score":source.get("score")},
                    )
                )
            if raw_docs:
                docs_map[q] = raw_docs

        total_docs = sum(len(docs) for docs in docs_map.values())
        log.info(f"web_search_node:{total_docs}")
        return {"docs_map": docs_map}
    
    except Exception as e:
        log.error(f"web_search_node: {e}")
        print(traceback.format_exc())
        return {}

# 网页爬虫节点
async def web_scrape_node(state: State,config: RunnableConfig):
    docs_map = state.get("docs_map")
    
    try:
        from agi.utils.scrape import WebScraper
        scraper = WebScraper()
        if docs_map is None:
            # 处理直接输入url解析的场景
            import re
            text = get_last_message_text(state)
            url_pattern = r'https?://[^\s"\'<>]+'
            urls = re.findall(url_pattern, text)
            docs = await scraper.aload(urls)
            return {"docs_map": {text:docs}}
    
        query_url_map = {k: [doc.metadata["link"] for doc in v if doc.metadata.get("link")] for k, v in docs_map.items()}
        log.info(f"web_scrape_node:{len(query_url_map)}")
        if query_url_map:
            query_doc_map = await scraper.aload2(query_url_map)
            for q, new_docs in query_doc_map.items():
                if new_docs:  # 非空才追加
                    docs_map.setdefault(q, []).extend(new_docs)
        
        total_docs = sum(len(docs) for docs in docs_map.values())
        log.info(f"web_scrape_node:{total_docs}")
        return {"docs_map": docs_map}
    except Exception as e:
        log.error(f"web_scrape_node: {e}")
        print(traceback.format_exc())
        return {"docs_map": docs_map}

async def doc_split_node(state: State, config: RunnableConfig):
    try:
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
        docs_map = state["docs_map"]
        for q,docs in docs_map.items():
            documents = await text_splitter.atransform_documents(docs)
            docs_map[q] =  documents
        
        total_docs = sum(len(docs) for docs in docs_map.values())
        log.info(f"split {total_docs} docs")
        return {"docs_map": docs_map}
    except Exception as e:
        log.error(f"doc_split_node: {e}")
        print(traceback.format_exc())
        return {"docs_map": docs_map}

# 适用于web 和 rag的情况，当无法获取有效的上下文信息时，
    # 1.重置feature特性
    # 2.交给llm_with_history处理
async def context_control(state: State):
    docs_map = state.get("docs_map")
    log.info(f"context_control:{len(docs_map)}")
    if docs_map:
        return "split"
    return "llm_with_history"

# 分析用户意图，自主决策
async def rag_auto_route(state: State):
    ai = await intend_understand_chain.ainvoke(state)
    _, result = split_think_content(ai.content)
    log.info(f"rag_auto_route:{result}")

    tenant = state.get("user_id")
    if "summary" in result and state["collection_names"]:
        return "summary"
    elif "rag" in result:
        if not state.get("collection_names"):
            collection_names = collection_manager.list_collections(tenant=tenant)
            state["collection_names"] = collection_names
        return "index_search"
    log.info(f"collection_names for {tenant} are {state['collection_names']}")

    return "llm_with_history"
    
async def route(state: State):
    try:
        # 状态初始化
        state["context"] = None
        state["docs"] = None
        state["citations"] = None

        feature = state.get("feature","")
        if feature == Feature.RAG:
            return await rag_auto_route(state)
        elif feature == Feature.SCRAPE:
            return "scrape"
        else:
            state["feature"] = feature = "web"
            return "web"
    except Exception as e:
        log.error(f"route: {e}")
        print(traceback.format_exc())
        return END

async def index_search_node(state: State,config: RunnableConfig):
    try:
        tenant = state.get("user_id")
        feature = state.get("feature")
        question = get_last_message_text(state)
        questions = refine_query(feature,query=question)

        doc_map = await collection_manager.embedding_search(questions,"index",tenant=tenant)
        total_docs = sum(len(docs) for docs in doc_map.values())
        log.info(f"index_search_node:{total_docs}")
        # state["questions"] = questions
        return {"index_search_result":doc_map}
    except Exception as e:
        log.error(f"index_search_node: {e}")
        print(traceback.format_exc())
        return {}

async def search_node(state: State, config: RunnableConfig):
    try:
        tenant = state.get("user_id")
        index_docs = state.get("index_search_result")
        pairs = get_clusterid_collection_pair(index_docs)
        log.info(f"search_node: total pairs={len(pairs)}")
        questions = list(index_docs.keys())
        log.info(f"search_node:{questions}")
        docs_map: Dict[str, List[Document]] = {q: [] for q in questions}

        # 依据类和collection 检索
        if pairs:
            for cid,cname in pairs:
                parts_map = await collection_manager.embedding_search(
                    texts=questions,
                    collection_name=cname,
                    cluster_id=cid,
                    tenant=tenant
                )
                # 合并到 docs_map
                for q in questions:
                    docs_map[q].extend(parts_map.get(q, []))
        else:
            # 全量检索
            collection_names = state["collection_names"]
            for collection_name in set(collection_names):
                if collection_name == "index":
                    continue

                parts_map = await collection_manager.embedding_search(
                    texts=questions,
                    collection_name=collection_name,
                    tenant=tenant
                )
                for q in questions:
                    docs_map[q].extend(parts_map.get(q, []))

        total_docs = sum(len(docs) for docs in docs_map.values())
        log.info(f"search_node: total_docs={total_docs}")
        return {"docs_map": docs_map}
    except Exception as e:
        log.error(f"search_node: {e}")
        print(traceback.format_exc())
        return {}
# graph
checkpointer = MemorySaver()

rag_graph_builder = StateGraph(State)

rag_graph_builder.add_node("llm_with_history", TaskFactory.create_task(TASK_LLM_WITH_HISTORY))
rag_graph_builder.add_node("doc_chat", doc_chat_node)
rag_graph_builder.add_node("rerank", doc_rerank_node)

rag_graph_builder.add_node("summary", doc_summary_node)

rag_graph_builder.add_node("index_search", index_search_node)
rag_graph_builder.add_node("search", search_node)


rag_graph_builder.add_node("web", web_search_node)
rag_graph_builder.add_node("scrape", web_scrape_node)
rag_graph_builder.add_node("split", doc_split_node)
# graph
rag_graph_builder.add_conditional_edges(START, route)

rag_graph_builder.add_edge("web","scrape")
rag_graph_builder.add_conditional_edges("scrape", context_control)
rag_graph_builder.add_edge("split", "rerank")


rag_graph_builder.add_edge("index_search", "search")
rag_graph_builder.add_edge("search", "rerank")

rag_graph_builder.add_edge("rerank", "doc_chat")
rag_graph_builder.add_edge("summary", "doc_chat")

rag_graph_builder.add_edge("llm_with_history", END)
rag_graph_builder.add_edge("doc_chat", END)

rag_graph = rag_graph_builder.compile(checkpointer=checkpointer,name="rag")
rag_as_subgraph = rag_graph_builder.compile(name="rag")
graph_print(rag_graph)