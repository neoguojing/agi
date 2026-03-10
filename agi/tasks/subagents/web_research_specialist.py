from __future__ import annotations

from typing import Any

from langchain.tools import tool
from langchain_core.messages import HumanMessage

from agi.tasks.define import State


@tool(return_direct=True)
async def web_search(query: str) -> Any:
    """Web search and synthesis for real-time or external knowledge."""
    from agi.tasks.rag_web import rag_as_subgraph

    state = State(messages=[HumanMessage(content=query)], user_id="subagent_web", feature="web")
    config = {"configurable": {"conversation_id": "subagent_web", "thread_id": "subagent_web"}}
    return await rag_as_subgraph.ainvoke(state, config=config)


@tool(return_direct=True)
async def web_scrape(query: str) -> Any:
    """Web scraping pipeline for provided URL(s)."""
    from agi.tasks.rag_web import rag_as_subgraph

    state = State(messages=[HumanMessage(content=query)], user_id="subagent_web", feature="scrape")
    config = {"configurable": {"conversation_id": "subagent_web", "thread_id": "subagent_web"}}
    return await rag_as_subgraph.ainvoke(state, config=config)


web_subagent = {
    "name": "web_research_specialist",
    "description": "Use this for web retrieval, scraping, and evidence synthesis.",
    "system_prompt": "You are a web research specialist. Gather and summarize credible evidence.",
    "tools": [web_search, web_scrape],
}
