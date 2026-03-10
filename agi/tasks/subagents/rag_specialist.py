from __future__ import annotations

from typing import Any

from langchain.tools import tool
from langchain_core.messages import HumanMessage

from agi.tasks.define import State


@tool(return_direct=True)
async def rag_search(query: str) -> Any:
    """Knowledge-base retrieval and synthesis for indexed content."""
    from agi.tasks.rag_web import rag_as_subgraph

    state = State(messages=[HumanMessage(content=query)], user_id="subagent_rag", feature="rag")
    config = {"configurable": {"conversation_id": "subagent_rag", "thread_id": "subagent_rag"}}
    return await rag_as_subgraph.ainvoke(state, config=config)


rag_subagent = {
    "name": "rag_specialist",
    "description": "Use this for knowledge-base upload/retrieval and document-grounded answers.",
    "system_prompt": "You are a RAG specialist. Prefer retrieval-grounded responses with citations.",
    "tools": [rag_search],
}
