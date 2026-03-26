import asyncio
import json
from typing import Callable, List, Optional,Awaitable
from venv import logger
from langchain_core.messages import SystemMessage, BaseMessage
from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse

from agi.agent.context import create_default_context_manager,UnifiedContextUpdater
from agi.utils.common import append_to_system_message
class ContextEngineeringMiddleware(AgentMiddleware):
    """
    上下文工程中间件：
    1. 并发加载 RAG 知识与用户画像
    2. 动态注入模型 Prompt
    3. 异步、非阻塞地更新用户画像
    """
    def __init__(
        self, 
        extractor_model, 
        retriever=None, 
        timeout: float = 2.0
    ):
        # --- 内部直接构建，减少外部参数 ---
        self.builder = create_default_context_manager(retriever=retriever)
        
        # 显式持有 updater，消除 runtime.extra 依赖
        self.updater = UnifiedContextUpdater(model=extractor_model)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]], 
    ) -> ModelResponse:
        
        # 1. 异步并行构建上下文 (RAG + Profile)
        injected_context_str = await self.builder.get_context_str(request.runtime,request.state)
        
        # 3. 智能合并消息 (确保注入内容位于首个 SystemMessage 之后)
        request = request.override(system_message=append_to_system_message(request.system_message, injected_context_str))

        # 4. 执行模型调用
        self._log_debug_info(injected_context_str, len(request.messages))
        response = await handler(request)
        async def update_profile_task():
            try:
                await self.updater.update(
                    runtime=request.runtime, 
                    messages=request.messages, 
                    ai_response=response.result
                )
            except Exception as e:
                logger.error(f"Failed to update user profile: {e}")
    
        asyncio.create_task(update_profile_task())

        return response


    def _log_debug_info(self, ctx_data: str, total_count: int):
        print(f"--- [Context Engine] 注入数据: {ctx_data} | 消息流长度: {total_count} ---")