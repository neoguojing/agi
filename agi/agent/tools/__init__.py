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
from agi.web.search_engine import SearchEngineSelector

# 显式暴露可用工具数组
# 你可以直接放函数（如果用了 @tool 装饰器），也可以放实例化后的对象
buildin_tools: List[BaseTool] = [
    get_weather_info,
    get_stock,
    # SearchEngineSelector()
]

# 导出清单，方便其他模块调用
__all__ = ["buildin_tools","SearchEngineSelector","RemoteImageGenTool",
           "RemoteImageEditTool","RemoteTTSTool","RemoteTranscriptionTool","RemoteMultiModalTool"]