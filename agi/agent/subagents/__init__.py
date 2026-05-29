# tools/__init__.py
import os

from deepagents.middleware.async_subagents import AsyncSubAgent

# 导入具体的工具实现
from .general import *

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

stock_async_subagent: AsyncSubAgent = {
    "name": "stock-analyse-asubagent",
    "description": "Specialized in stock market analyse task.",
    "graph_id": "stock-analyse-asubagent",
}

# Empty URL means co-registered LangGraph graphs communicate in-process via ASGI.
# Set this env var only when the stock subagent runs on another LangGraph server.
if stock_async_subagent_url := os.getenv("AGI_STOCK_ASUBAGENT_URL"):
    stock_async_subagent["url"] = stock_async_subagent_url

buildin_async_agents: list[AsyncSubAgent] = [stock_async_subagent]


# 导出清单，方便其他模块调用
__all__ = ["buildin_agents", "make_backend", "buildin_async_agents"]
