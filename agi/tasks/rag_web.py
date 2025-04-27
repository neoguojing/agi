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
from agi.tasks.retriever import FilterType
from agi.tasks.utils import get_last_message_text,split_think_content
import asyncio
import json

intend_understand_prompt = '''
You are the query router for a RAG system. Your job is to inspect each user query and reply with exactly one of the following tokens, so the downstream pipeline knows which mode to run:

1. summary  
   – Use this when the user’s request is about summarizing, extracting, classifying, listing, or otherwise manipulating text they’ve already provided, without any need to fetch external documents.  
   
2. rag  
   – Use this when the user’s request likely requires retrieving or grounding in external content (e.g. “What did the New York Times say about X?”, “Give me details from the latest research paper on Y”, “What is the plot of [book title]?”).  
   
3. other  
   – Use this for any query that is neither a pure summary/extraction nor needs external retrieval. In this case, the router may either:  
     • Ask the user for clarification (“Could you clarify what you’d like?”),  
     • Route to a general-purpose chat/QA module,  
     • Or trigger another specialized pipeline (e.g. translation, code execution, scheduling).  

Routing rules:  
- If the user explicitly asks to “summarize,” “extract,” “list,” “compare,” or “classify” text they have pasted, always select **summary**.  
- If the user asks for facts, excerpts, or analysis of documents, articles, books, or research they haven’t provided in full, select **rag**.  
- Otherwise, select **other**.

Examples:

User: “Please summarize the following paragraph: …”  
Router → summary

User: “What are the main findings of the 2024 IPCC report?”  
Router → rag

User: “How do I bake a chocolate cake?”  
Router → other

---

Feel free to extend the “other” branch with additional tokens or clarifying questions as your system requires.

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

def intend_understand_modify_state_messages(state: State):
    # 可能会存在重复的系统消息需要去掉
    filter_messages = []
    for message in state["messages"]:
        if isinstance(message,SystemMessage):
            continue
        # 修正请求的类型，否则openapi会报错
        if not isinstance(message.content,str):
             message.content = json.dumps(message.content)
        filter_messages.append(message)
    return intend_understand_template.invoke({"messages": filter_messages}).to_messages()


intend_understand__modify_state_messages_runnable = RunnableLambda(intend_understand_modify_state_messages)

intend_understand_chain = intend_understand__modify_state_messages_runnable | TaskFactory.get_llm() 

# NODE
# 文档对话
def doc_chat_node(state: State,config: RunnableConfig,writer: StreamWriter):
    chain = TaskFactory.create_task(TASK_DOC_CHAT)
    if state["citations"]:
        writer({"citations":state["citations"],"docs":state["docs"]})
    return chain.invoke(state,config=config)

def doc_compress_node(state: State,config: RunnableConfig):
    km = TaskFactory.get_knowledge_manager()
    retriever = km.get_compress_retriever(FilterType.LLM_EXTRACT)
    question = get_last_message_text(state)
    docs = state.get("docs")
    state["docs"] = asyncio.run(retriever.acompress_documents(docs,question))
    return state 

# 列举collection前面部分的文本页,用于总结文章
def doc_list_node(state: State,config: RunnableConfig):
    km = TaskFactory.get_knowledge_manager()
    docs = km.list_documets()
    state["docs"] = docs
    return state 

# 适用于web 和 rag的情况，当无法获取有效的上下文信息时，
    # 1.重置feature特性
    # 2.交给llm_with_history处理
def context_control(state: State):
    docs = state.get("docs")
    if docs:
        return "compress"
    return "llm_with_history"

# 分析用户意图，自主决策
def rag_auto_route(state: State):
    ai = intend_understand_chain.invoke(state)
    _, result = split_think_content(ai.content)

    if result == "summary":
        return "summary"
    elif result == "rag":
        return "rag"
    else:
        return "llm_with_history"
    
def route_node(state: State):
    feature = state.get("feature","")

    if feature == Feature.RAG:
        return rag_auto_route(state)
    elif feature == Feature.WEB:
        return "web"
    elif state.get("collection_names"):
        return rag_auto_route(state)


# graph
checkpointer = MemorySaver()

rag_graph_builder = StateGraph(State)

rag_graph_builder.add_node("doc_chat", doc_chat_node)
rag_graph_builder.add_node("compress", doc_compress_node)
rag_graph_builder.add_node("rag", TaskFactory.create_task(TASK_RAG))
rag_graph_builder.add_node("summary", doc_list_node)
rag_graph_builder.add_node("llm_with_history", TaskFactory.create_task(TASK_LLM_WITH_HISTORY))

rag_graph_builder.add_node("web", TaskFactory.create_task(TASK_WEB_SEARCH))


rag_graph_builder.add_edge(START, route_node)
rag_graph_builder.add_conditional_edges("rag", context_control)
rag_graph_builder.add_conditional_edges("web", context_control)
rag_graph_builder.add_edge("summary", "doc_chat")
rag_graph_builder.add_edge("compress", "doc_chat")
rag_graph_builder.add_edge("doc_chat", END)
rag_graph_builder.add_edge("llm_with_history", END)

rag_graph = rag_graph_builder.compile(checkpointer=checkpointer)