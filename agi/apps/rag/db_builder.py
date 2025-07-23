from agi.tasks.define import State,InputType,Feature
from agi.tasks.task_factory import (
    TaskFactory,
    TASK_DOC_CHAT,
    TASK_RAG,
    TASK_WEB_SEARCH,
    TASK_LLM_WITH_HISTORY
)
from langchain_core.runnables import (
    RunnableConfig
)
from langgraph.graph import END, StateGraph, START
from langgraph.checkpoint.memory import MemorySaver
from agi.apps.rag.file_loader import get_file_loader,get_web_loader,get_youtube_loader

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
        documents = loader.load()
        state["docs"] = documents

    return state

   
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
