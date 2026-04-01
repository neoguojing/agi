# tools/__init__.py
from typing import List
from langchain_core.tools import BaseTool

# 导入具体的工具实现
from .weather import get_weather_info
from .stock_market import get_stock
from .image_gen import RemoteImageGenTool,RemoteImageEditTool
from .multimodal import RemoteMultiModalTool
from .tts import RemoteTTSTool
from .whisper import RemoteTranscriptionTool
from .web_search import search_web

# 显式暴露可用工具数组
# 你可以直接放函数（如果用了 @tool 装饰器），也可以放实例化后的对象
buildin_tools: List[BaseTool] = [
    get_weather_info,
    get_stock,
]

# 导出清单，方便其他模块调用
__all__ = ["buildin_tools","search_web","RemoteImageGenTool",
           "RemoteImageEditTool","RemoteTTSTool","RemoteTranscriptionTool","RemoteMultiModalTool"]