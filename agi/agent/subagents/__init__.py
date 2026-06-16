# tools/__init__.py
# 导入具体的工具实现
from .general import *
from .async_subagents import *

# 显式暴露可用工具数组
# 你可以直接放函数（如果用了 @tool 装饰器），也可以放实例化后的对象
buildin_agents = [
    planner_subagent,
    # tts_subagent,
    # visual_subagent,
    # perception_subagent,
    # stt_subagent,
    web_search_subagent,
    # ffmpeg_subagent,
    # pdf_parser_subagent,
    # stock_analyse_subagent,
    # browser_subagent,
]


buildin_async_agents: list[AsyncSubAgent] = [
    stock_async_subagent,
    browser_async_subagent,
    ffmpeg_async_subagent,
    pdf_async_subagent,
]


# 导出清单，方便其他模块调用
__all__ = ["buildin_agents", "make_backend", "buildin_async_agents"]
