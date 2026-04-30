# tools/__init__.py
from typing import List
# 导入具体的工具实现
from .general import *

# 显式暴露可用工具数组
# 你可以直接放函数（如果用了 @tool 装饰器），也可以放实例化后的对象
buildin_agents = [
    tts_subagent,
    visual_subagent,
    perception_subagent,
    stt_subagent,
    browser_subagent,
    web_search_subagent,
    ffmpeg_subagent,
    pdf_parser_subagent
]
        

# 导出清单，方便其他模块调用
__all__ = ["buildin_agents","make_backend"]