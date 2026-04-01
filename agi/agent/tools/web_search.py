from typing import List, Dict
import json
from langchain.tools import ToolRuntime, tool
from langchain_core.messages import ToolMessage
from langgraph.types import Command

# 假设 SearchEngineSelector 类已导入
from agi.web.search_engine import SearchEngineSelector

# 建议在全局或应用启动时初始化，避免每次调用工具都重新加载引擎
_search_selector = SearchEngineSelector()

@tool
async def search_web(query: str, runtime: ToolRuntime) -> Command:
    """
    Search for real-time information using intelligent engine selection.
    Best for news, facts, and complex queries requiring up-to-date data.
    """
    
    # 1. 直接 await 异步搜索方法
    # batch_search 返回格式: { "query string": [ {doc1}, {doc2}... ] }
    results_dict = await _search_selector.batch_search([query])
    
    # 2. 提取当前查询的结果
    documents = results_dict.get(query, [])

    # 3. 格式化返回内容
    # 将搜索结果格式化为字符串，直接作为消息内容返回给模型
    if not documents:
        content = "No relevant search results found."
    else:
        # 推荐返回 JSON 字符串，保持结构化信息以便 LLM 解析
        content = json.dumps(documents, ensure_ascii=False, indent=2)

    # 4. 返回 Command
    # 仅更新 messages 列表，不修改图状态的其他字段
    return Command(
        update={
            "messages": [
                ToolMessage(
                    content=content,
                    tool_call_id=runtime.tool_call_id,
                )
            ]
        }
    )