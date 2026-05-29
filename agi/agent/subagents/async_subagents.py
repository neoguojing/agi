import contextlib
import os
from langchain_core.runnables import RunnableConfig

from agi.agent.deep_agent import create_deep_agent
from langchain.agents import create_agent
from agi.agent.middlewares.completion_notifier_middleware import build_completion_notifier
from agi.agent.models import ModelProvider
from agi.agent.subagents.general import (
    stock_analyse_subagent,
    ffmpeg_subagent,
    browser_subagent,
    pdf_parser_subagent,
)
from deepagents.middleware.async_subagents import AsyncSubAgent

# =====================================================================
# 1. 异步子智能体（AsyncSubAgent）映射配置
# =====================================================================

# --- Stock Subagent ---
stock_async_subagent: AsyncSubAgent = {
    "name": stock_analyse_subagent.get("name"),
    "description": stock_analyse_subagent.get("description"),
    "graph_id": stock_analyse_subagent.get("name")
}
if stock_async_subagent_url := os.getenv("AGI_STOCK_ASUBAGENT_URL"):
    stock_async_subagent["url"] = stock_async_subagent_url

# --- PDF Parser Subagent ---
pdf_async_subagent: AsyncSubAgent = {
    "name": pdf_parser_subagent.get("name"),
    "description": pdf_parser_subagent.get("description"),
    "graph_id": pdf_parser_subagent.get("name")
}
if pdf_async_subagent_url := os.getenv("AGI_PDF_ASUBAGENT_URL"):
    pdf_async_subagent["url"] = pdf_async_subagent_url

# --- FFmpeg Subagent ---
ffmpeg_async_subagent: AsyncSubAgent = {
    "name": ffmpeg_subagent.get("name"),
    "description": ffmpeg_subagent.get("description"),
    "graph_id": ffmpeg_subagent.get("name")
}
if ffmpeg_async_subagent_url := os.getenv("AGI_FFMPEG_ASUBAGENT_URL"):
    ffmpeg_async_subagent["url"] = ffmpeg_async_subagent_url

# --- Browser Subagent ---
browser_async_subagent: AsyncSubAgent = {
    "name": browser_subagent.get("name"),
    "description": browser_subagent.get("description"),
    "graph_id": browser_subagent.get("name")
}
if browser_async_subagent_url := os.getenv("AGI_BROWSER_ASUBAGENT_URL"):
    browser_async_subagent["url"] = browser_async_subagent_url


# =====================================================================
# 2. 独立 LangGraph 助手（Graph Context Managers）
# =====================================================================

@contextlib.asynccontextmanager
async def stock_graph(config: RunnableConfig):
    """Promote the stock subagent to a standalone LangGraph assistant."""
    configurable = (config or {}).get("configurable", {})
    notifier = build_completion_notifier(
        parent_thread_id=configurable.get("parent_thread_id"),
        parent_assistant_id=configurable.get("parent_assistant_id"),
        subagent_name=stock_analyse_subagent.get("name"),
    )
    yield create_agent(
        model=ModelProvider.get_falback_model(),  # 保持原代码中的拼写
        tools=stock_analyse_subagent.get("tools", []),
        system_prompt=stock_analyse_subagent.get("system_prompt", ""),
        middleware=[*stock_analyse_subagent.get("middleware", []), notifier],
        name=stock_analyse_subagent.get("name"),
    )


@contextlib.asynccontextmanager
async def pdf_graph(config: RunnableConfig):
    """Promote the PDF parser subagent to a standalone LangGraph assistant."""
    configurable = (config or {}).get("configurable", {})
    notifier = build_completion_notifier(
        parent_thread_id=configurable.get("parent_thread_id"),
        parent_assistant_id=configurable.get("parent_assistant_id"),
        subagent_name=pdf_parser_subagent.get("name"),
    )
    yield create_agent(
        model=ModelProvider.get_falback_model(),
        tools=pdf_parser_subagent.get("tools", []),
        system_prompt=pdf_parser_subagent.get("system_prompt", ""),
        middleware=[*pdf_parser_subagent.get("middleware", []), notifier],
        name=pdf_parser_subagent.get("name"),
    )


@contextlib.asynccontextmanager
async def ffmpeg_graph(config: RunnableConfig):
    """Promote the FFmpeg subagent to a standalone LangGraph assistant."""
    configurable = (config or {}).get("configurable", {})
    notifier = build_completion_notifier(
        parent_thread_id=configurable.get("parent_thread_id"),
        parent_assistant_id=configurable.get("parent_assistant_id"),
        subagent_name=ffmpeg_subagent.get("name"),
    )
    yield create_agent(
        model=ModelProvider.get_falback_model(),
        tools=ffmpeg_subagent.get("tools", []),
        system_prompt=ffmpeg_subagent.get("system_prompt", ""),
        middleware=[*ffmpeg_subagent.get("middleware", []), notifier],
        name=ffmpeg_subagent.get("name"),
    )


@contextlib.asynccontextmanager
async def browser_graph(config: RunnableConfig):
    """Promote the browser subagent to a standalone LangGraph assistant."""
    configurable = (config or {}).get("configurable", {})
    notifier = build_completion_notifier(
        parent_thread_id=configurable.get("parent_thread_id"),
        parent_assistant_id=configurable.get("parent_assistant_id"),
        subagent_name=browser_subagent.get("name"),
    )
    yield create_agent(
        model=ModelProvider.get_falback_model(),
        tools=browser_subagent.get("tools", []),
        system_prompt=browser_subagent.get("system_prompt", ""),
        middleware=[*browser_subagent.get("middleware", []), notifier],
        name=browser_subagent.get("name"),
    )