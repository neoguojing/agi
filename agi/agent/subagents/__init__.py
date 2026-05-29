# tools/__init__.py
import os
# 导入具体的工具实现
from .general import *
from deepagents.middleware.async_subagents import AsyncSubAgent

# 显式暴露可用工具数组
# 你可以直接放函数（如果用了 @tool 装饰器），也可以放实例化后的对象
buildin_agents = [
    # tts_subagent,
    # visual_subagent,
    # perception_subagent,
    # stt_subagent,
    browser_subagent,
    web_search_subagent,
    ffmpeg_subagent,
    pdf_parser_subagent,
    # stock_analyse_subagent,
]


def _async_subagent(name: str, description: str, graph_id: str, env_url: str | None = None) -> AsyncSubAgent:
    """Create an async subagent spec.

    If ``env_url`` is unset or the environment variable is empty, no ``url`` is
    included and LangGraph SDK uses in-process ASGI transport for graphs
    co-registered in the same ``langgraph.json``. Setting the URL switches the
    same spec to HTTP transport for direct remote-agent communication.
    """

    spec: AsyncSubAgent = {
        "name": name,
        "description": description,
        "graph_id": graph_id,
    }
    if env_url:
        url = os.getenv(env_url)
        if url:
            spec["url"] = url
    return spec


buildin_async_agents: list[AsyncSubAgent] = [
    _async_subagent(
        name="stock-analyse-asubagent",
        description="Specialized in stock market analyse task.",
        graph_id="stock-analyse-asubagent",
        env_url="AGI_STOCK_ASUBAGENT_URL",
    ),
]


# 导出清单，方便其他模块调用
__all__ = ["buildin_agents", "make_backend", "buildin_async_agents"]
