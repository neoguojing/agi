from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda

from agi.config import log
from agi.tasks.define import State
from agi.tasks.rag import rag_query
from agi.tasks.utils import get_last_message_text
from agi.utils.search_engine import SearchEngineSelector


def _ensure_messages(state: State, text: str) -> State:
    return {**state, "messages": [AIMessage(content=text)]}


async def llm_with_history_node(state: State, config: RunnableConfig | None = None) -> State:
    """Chain-free fallback chat node.

    Keep state-compatible behavior for legacy graph users while routing through
    plain model invocation (without agi.tasks.chat.chains).
    """
    from agi.tasks.runtime.task_factory import TaskFactory

    llm = TaskFactory.get_llm_with_output_format()
    output = await llm.ainvoke(state, config=config)
    if isinstance(output, dict) and output.get("messages"):
        return {**state, "messages": output["messages"]}
    return state


async def rag_retrieve_node(state: State, config: RunnableConfig | None = None) -> State:
    """RAG retrieval via tool API (replaces legacy custom chain entry)."""
    query = get_last_message_text(state)
    if not query:
        return {**state, "docs": []}

    collections = state.get("collection_names") or ["all"]
    collection_name = collections[0] if isinstance(collections, list) else "all"
    tenant = state.get("user_id") or ""
    top_k = int((config or {}).get("configurable", {}).get("top_k", 3))

    docs = await rag_query.ainvoke(
        {
            "query": query,
            "collection_name": collection_name,
            "tenant": tenant,
            "top_k": top_k,
        }
    )
    return {**state, "docs": docs}


async def web_search_node(state: State, config: RunnableConfig | None = None) -> State:
    """Web retrieval via search tool backend (without custom chain composition)."""
    query = get_last_message_text(state)
    if not query:
        return {**state, "docs": [], "urls": []}

    selector = SearchEngineSelector()
    raw_results = await selector.search(query)

    docs: list[dict[str, Any]] = []
    urls: list[str] = []
    for item in raw_results or []:
        link = item.get("link")
        snippet = item.get("snippet") or ""
        title = item.get("title") or ""
        if link:
            urls.append(link)
        if snippet.strip() or title.strip():
            docs.append({"content": f"{title}\n{snippet}".strip(), "metadata": item})

    return {**state, "docs": docs, "urls": urls}


async def doc_chat_node(state: State, config: RunnableConfig | None = None) -> State:
    """Answer from docs directly, avoiding legacy stuff_documents chain."""
    from agi.tasks.runtime.task_factory import TaskFactory

    query = get_last_message_text(state)
    docs = state.get("docs") or []

    if not docs:
        return await llm_with_history_node(state, config)

    context_lines: list[str] = []
    for idx, doc in enumerate(docs[:8], start=1):
        if isinstance(doc, dict):
            content = str(doc.get("content", "")).strip()
        else:
            content = str(getattr(doc, "page_content", "")).strip()
        if content:
            context_lines.append(f"[{idx}] {content}")

    if not context_lines:
        return await llm_with_history_node(state, config)

    prompt = (
        "You are a document-grounded assistant. Use ONLY the provided context to answer. "
        "If context is insufficient, say so briefly.\n\n"
        f"Question:\n{query}\n\nContext:\n" + "\n\n".join(context_lines)
    )

    llm = TaskFactory.get_llm()
    response = await llm.ainvoke([HumanMessage(content=prompt)])
    if not isinstance(response, AIMessage):
        log.warning("doc_chat_node received non-AI response: %s", type(response))
        return state

    return _ensure_messages(state, str(response.content))


llm_with_history_runnable = RunnableLambda(llm_with_history_node)
rag_retrieve_runnable = RunnableLambda(rag_retrieve_node)
web_search_runnable = RunnableLambda(web_search_node)
doc_chat_runnable = RunnableLambda(doc_chat_node)
