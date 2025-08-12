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
from agi.config import log,CACHE_DIR
from agi.tasks.vectore_store import CollectionManager
from agi.llms.rerank import rerank_with_batching
import json
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
collection_manager = CollectionManager(data_path=CACHE_DIR,embedding=TaskFactory.get_embedding())

def get_cluster_ids(docs:list[Document]):
    cluster_ids = []
    for doc in docs:
        cluster_ids.append(doc.metadata.get("cluster_id"))
    return set(cluster_ids)

# NODE
# 文档对话
async def doc_chat_node(state: State,config: RunnableConfig,writer: StreamWriter):
    chain = TaskFactory.create_task(TASK_DOC_CHAT)
    state["citations"] = build_citations(state)
    if state.get("citations"):
        writer({"citations":state["citations"],"docs":state["docs"]})
    log.info(f"doc_chat_node:{len(state['docs'])}")
    result = await chain.ainvoke(state,config=config)
    result["citations"] = state["citations"]
    result["docs"] = None
    result["doc_list_node"] = None
    result["index_search_result"] = None

    return result

async def doc_rerank_node(state: State,config: RunnableConfig):
    question = get_last_message_text(state)
    docs = state.get("docs")
    docs = await rerank_with_batching(question,docs)
    log.info(f"doc_rerank_node:{len(docs)}")
    log.info(f"doc_rerank_node:{docs}")

    return {"docs":docs} 

# 获取指定文件的索引文件
async def doc_summary_node(state: State,config: RunnableConfig):
    tenant = state.get("user_id")
    source = state.get("file_path")
    state["docs"] = []
    docs = collection_manager.get_documents("index",source=source,tenant=tenant)
    log.info(f"doc_list_node:{len(docs)}")
    return {"docs":docs}

# 网页爬虫节点
async def web_scrape_node(state: State,config: RunnableConfig):
    km = TaskFactory.get_knowledge_manager()
    urls = state.get("urls")
    if urls and not state.get("docs"):
        _,_,docs = await km.store("web",source=urls,source_type=SourceType.WEB)
        state["docs"] = docs
        log.info(f"web_scrape_node:{len(state['docs'])}")

    return state 

# 适用于web 和 rag的情况，当无法获取有效的上下文信息时，
    # 1.重置feature特性
    # 2.交给llm_with_history处理
async def context_control(state: State):
    docs = state.get("docs")
    if docs:
        return "rerank"
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
        collection_names = collection_manager.list_collections(tenant=tenant)
        if state["collection_names"]:
            state["collection_names"].extend(collection_names)
        else:
            state["collection_names"] = collection_names
        return "index_search"
    log.info(f"collection_names for {tenant} are {state['collection_names']}")

    return "llm_with_history"
    
async def route(state: State):
    # 状态初始化
    state["context"] = None
    state["docs"] = None
    state["citations"] = None

    feature = state.get("feature","")

    if feature == Feature.RAG:
        return await rag_auto_route(state)
    else:
        return "web"

async def index_search_node(state: State,config: RunnableConfig):
    tenant = state.get("user_id")
    question = get_last_message_text(state)
    docs = await collection_manager.embedding_search([question],"index",tenant=tenant)
    # state["docs"] = docs
    log.info(f"index_search_node:{len(docs)}")
    log.info(f"index_search_node:{docs}")

    return {"index_search_result":docs} 

async def search_node(state: State,config: RunnableConfig):
    tenant = state.get("user_id")
    collection_names = state["collection_names"]
    index_docs = state.get("index_search_result")
    question = get_last_message_text(state)
    cluster_ids = get_cluster_ids(index_docs)
    docs = []

    for collection_name in set(collection_names):
        if cluster_ids:
            for id in cluster_ids:
                parts = await collection_manager.embedding_search([question],collection_name,cluster_id=id,tenant=tenant)
                docs.extend(parts)
        else: #在index未检索到时，全量检索
            parts = await collection_manager.embedding_search([question],collection_name,tenant=tenant)
            docs.extend(parts)
            
    log.info(f"search_node:{len(docs)}")
    return {"docs":docs} 
# graph
checkpointer = MemorySaver()

rag_graph_builder = StateGraph(State)

rag_graph_builder.add_node("llm_with_history", TaskFactory.create_task(TASK_LLM_WITH_HISTORY))
rag_graph_builder.add_node("doc_chat", doc_chat_node)
rag_graph_builder.add_node("rerank", doc_rerank_node)

rag_graph_builder.add_node("summary", doc_summary_node)

rag_graph_builder.add_node("index_search", index_search_node)
rag_graph_builder.add_node("search", search_node)


rag_graph_builder.add_node("web", TaskFactory.create_task(TASK_WEB_SEARCH))
rag_graph_builder.add_node("scrape", web_scrape_node)
# graph
rag_graph_builder.add_conditional_edges(START, route)

rag_graph_builder.add_edge("web","scrape")
rag_graph_builder.add_conditional_edges("scrape", context_control)

rag_graph_builder.add_edge("index_search", "search")
rag_graph_builder.add_edge("search", "rerank")
rag_graph_builder.add_edge("rerank", "doc_chat")

rag_graph_builder.add_edge("summary", "doc_chat")

rag_graph_builder.add_edge("llm_with_history", END)
rag_graph_builder.add_edge("doc_chat", END)

rag_graph = rag_graph_builder.compile(checkpointer=checkpointer,name="rag")
rag_as_subgraph = rag_graph_builder.compile(name="rag")
graph_print(rag_graph)