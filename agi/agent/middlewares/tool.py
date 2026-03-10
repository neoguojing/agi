from langchain.agents.middleware.types import AgentMiddleware
from typing import Any

class DynamicToolMiddleware(AgentMiddleware):
    def __init__(self, retriever: ToolRetriever, top_k: int = 5):
        self.retriever = retriever
        self.top_k = top_k

    async def __call__(self, state: dict[str, Any], call_next):
        # 获取最后一条用户消息的内容作为检索 query
        last_message = state["messages"][-1].content
        
        # 1. 动态检索工具
        selected_tools = self.retriever.retrieve(last_message, k=self.top_k)
        
        # 2. 注入到状态中 (假设 create_deep_agent 的底层逻辑支持从 state 读取 tools)
        # 在 LangGraph 中，我们通常修改 RunnableConfig 或传递给 LLM 的 bind_tools
        state["dynamic_tools"] = selected_tools
        
        # 3. 继续执行后续中间件
        return await call_next(state)