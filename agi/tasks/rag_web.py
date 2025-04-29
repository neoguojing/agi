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

intend_understand_prompt = '''
You are the query router for a RAG system. For each user query, reply with exactly one of these tokens:

1. summary  
   – Use this for operations on text the user has already provided in full within their query.  
   – Examples:  
     • Summarize the paragraph below: …
     • List the key entities in this text: …
     • Compare these two provided paragraphs: …

2. rag  
   – Use this when the user needs content that must be fetched or grounded in an external document, knowledge base, or corpus that they have *not* pasted in full. Also use this if they refer by name to a document (e.g. “in the Yuan master document,” “in the product spec,” “in our API guide”).  
   – Examples:  
     • What did the New York Times say about X?
     • In the docs, what is said about browser compatibility?
     • Give me details from the latest research paper on Y.
     • Plot summary of [book title].

Routing rules:
- summary
  – Query explicitly asks to summarize, extract, classify, compare, or list from text included in the user’s request.  
- rag
  – Query asks for facts, excerpts, or analysis of documents/articles/books/research **not** fully included in the request.  
  – Query names or alludes to a document, guide, spec, or corpus (“the Yuan master doc,” “our product manual,” etc.), implying you must retrieve it.  

Examples:

User: Please summarize the following paragraph: …
Router → summary

User: What are the main findings of the 2024 IPCC report?  
Router → rag

User: 在文档中，关于浏览器兼容性的描述是什么？
Router → rag


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
    render_result = await intend_understand_template.ainvoke({"messages": filter_messages})
    return render_result.to_messages()


intend_understand__modify_state_messages_runnable = RunnableLambda(intend_understand_modify_state_messages)

intend_understand_chain = intend_understand__modify_state_messages_runnable | TaskFactory.get_llm() 

# NODE
# 文档对话
async def doc_chat_node(state: State,config: RunnableConfig,writer: StreamWriter):
    chain = TaskFactory.create_task(TASK_DOC_CHAT)
    state["citations"] = build_citations(state)
    if state.get("citations"):
        writer({"citations":state["citations"],"docs":state["docs"]})
    return await chain.ainvoke(state,config=config)

async def doc_compress_node(state: State,config: RunnableConfig):
    km = TaskFactory.get_knowledge_manager()
    retriever = km.get_compress_retriever(FilterType.LLM_EXTRACT)
    question = get_last_message_text(state)
    docs = state.get("docs")
    docs = await retriever.acompress_documents(docs,question)
    if docs:
        docs = [d for d in docs if d.page_content and not d.page_content.strip().startswith("NO")]
    state["docs"] = docs
    log.info(f"doc_compress_node:{docs}")
    return state 

# 列举collection前面部分的文本页,用于总结文章
async def doc_list_node(state: State,config: RunnableConfig):
    km = TaskFactory.get_knowledge_manager()
    collection_names = state.get("collection_names",[])
    tenant = state.get("user_id")
    if len(collection_names) > 0 :
        docs = km.list_documets(collection_names[0],tenant=tenant)
        state["docs"] = docs
    return state 

# 网页爬虫节点
async def web_scrape_node(state: State,config: RunnableConfig):
    km = TaskFactory.get_knowledge_manager()
    urls = state.get("urls")
    if urls and not state.get("docs"):
        _,_,docs = await km.store("web",source=urls,source_type=SourceType.WEB)
        state["docs"] = docs
    return state 

# 适用于web 和 rag的情况，当无法获取有效的上下文信息时，
    # 1.重置feature特性
    # 2.交给llm_with_history处理
async def context_control(state: State):
    docs = state.get("docs")
    if docs:
        return "compress"
    return "llm_with_history"

# 分析用户意图，自主决策
async def rag_auto_route(state: State):
    ai = await intend_understand_chain.ainvoke(state)
    _, result = split_think_content(ai.content)
    log.info(f"rag_auto_route:{result}")

    if "summary" in result:
        return "summary"
    elif "rag" in result:
        return "rag"
    
    return "llm_with_history"
    
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

rag_graph_builder = StateGraph(State)

rag_graph_builder.add_node("doc_chat", doc_chat_node)
rag_graph_builder.add_node("compress", doc_compress_node)
rag_graph_builder.add_node("rag", TaskFactory.create_task(TASK_RAG))
rag_graph_builder.add_node("summary", doc_list_node)
rag_graph_builder.add_node("llm_with_history", TaskFactory.create_task(TASK_LLM_WITH_HISTORY))
rag_graph_builder.add_node("web", TaskFactory.create_task(TASK_WEB_SEARCH))
rag_graph_builder.add_node("scrape", web_scrape_node)



rag_graph_builder.add_conditional_edges(START, route)

rag_graph_builder.add_conditional_edges("web","scrape")
rag_graph_builder.add_conditional_edges("scrape", context_control)

rag_graph_builder.add_conditional_edges("rag", context_control)
rag_graph_builder.add_edge("summary", "doc_chat")

rag_graph_builder.add_edge("llm_with_history", END)

rag_graph_builder.add_edge("compress", "doc_chat")
rag_graph_builder.add_edge("doc_chat", END)

rag_graph = rag_graph_builder.compile(checkpointer=checkpointer,name="rag")

graph_print(rag_graph)
