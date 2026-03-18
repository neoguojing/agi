import asyncio
import json
from typing import Callable, List, Optional
from langchain_core.messages import SystemMessage, BaseMessage
from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse

from agi.agent.context import UserProfileProvider,KnowledgeProvider,AsyncContextBuilder,ContextRenderer,ProfileUpdater

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
        providers = [UserProfileProvider()]
        if retriever:
            providers.append(KnowledgeProvider(retriever))
            
        self.builder = AsyncContextBuilder(providers, timeout=timeout)
        self.renderer = ContextRenderer()
        
        # 显式持有 updater，消除 runtime.extra 依赖
        self.updater = ProfileUpdater(extractor_model=extractor_model)

    async def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        
        # 1. 异步并行构建上下文 (RAG + Profile)
        ctx_data = await self.builder.build(request.runtime, request.state)

        # 2. 渲染注入消息
        injected_messages = self.renderer.render(ctx_data)
        
        # 3. 智能合并消息 (确保注入内容位于首个 SystemMessage 之后)
        new_messages = self._smart_inject(request.messages, injected_messages)
        request = request.override(messages=new_messages)

        # 4. 执行模型调用
        self._log_debug_info(ctx_data, len(new_messages))
        response = await handler(request)

        # 5. 后置处理：利用 asyncio.shield 实现安全的“火后即忘”更新
        # 直接使用 self.updater，逻辑清晰
        asyncio.create_task(
            asyncio.shield(
                self.updater.update(
                    runtime=request.runtime, 
                    messages=request.messages, 
                    ai_response=response.result
                )
            )
        )

        return response

    def _smart_inject(self, original: List[BaseMessage], injected: List[BaseMessage]) -> List[BaseMessage]:
        """将背景知识注入到系统指令之后，用户对话之前"""
        if not injected:
            return original
            
        res = list(original)
        # 寻找第一个非 System 消息的位置（通常是 Human 消息）
        insert_idx = 0
        for i, msg in enumerate(res):
            if msg.type != "system":
                insert_idx = i
                break
        else:
            insert_idx = len(res)
            
        return res[:insert_idx] + injected + res[insert_idx:]

    def _log_debug_info(self, ctx_data: dict, total_count: int):
        print(f"--- [Context Engine] 注入数据: {list(ctx_data.keys())} | 消息流长度: {total_count} ---")