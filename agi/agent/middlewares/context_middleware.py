import json
import platform
import datetime
import os
import asyncio
from typing import Callable, List, Awaitable, Any
from venv import logger
from langchain_core.messages import SystemMessage, BaseMessage, AIMessage
from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from deepagents.backends.protocol import BackendProtocol
from agi.agent.prompt import get_middleware_prompt
from agi.utils.common import append_to_system_message
from agi.agent.context.memory import (
    MemoryMaintenanceManager,
    format_memory_for_llm,
    MessageProvider
)

class MiddlewareMessageProvider(MessageProvider):
    """Dynamic message provider that can be updated by the middleware."""
    def __init__(self):
        self._messages = []

    def update_messages(self, messages: List[BaseMessage]):
        self._messages = messages

    def get_messages(self) -> List[BaseMessage]:
        return self._messages

class ContextEngineeringMiddleware(AgentMiddleware):
    """
    上下文工程中间件：
    1. 动态注入模型 Prompt
    2. 后台异步执行记忆提取 (MemoryMaintenanceManager)
    3. 动态注入最新记忆 (format_memory_for_llm)
    4. 消息压缩策略（保留）
    """
    def __init__(
        self,
        backend = None,
        llm = None
    ):
        self.backend = backend
        self.llm = llm
        self.memory_manager = None
        self.message_provider = MiddlewareMessageProvider()

    def _get_backend(self, runtime) -> BackendProtocol:
        if callable(self.backend):
            return self.backend(runtime)
        return self.backend

    def _format_environment_context(self, runtime) -> str:
        try:
            now = datetime.datetime.utcnow().isoformat()

            env_info = {
                "current_time_utc": now,
                "timezone": "UTC",
                "os": platform.system(),
                "os_version": platform.version(),
                "python_version": platform.python_version()
            }

            # runtime 可选扩展
            if runtime:
                env_info.update({
                    "user_id": getattr(runtime.context, "user_id", None),
                    "conversation_id": getattr(runtime.context, "conversation_id", None),
                })

            env_str = json.dumps(env_info, indent=2, ensure_ascii=False)

            return f"""
    <environment>
    {env_str}
    </environment>
    """.strip()

        except Exception as e:
            logger.error(f"Failed to build environment context: {e}")
            return "<environment>(failed to load)</environment>"

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:

        runtime = request.runtime
        backend = self._get_backend(runtime)

        # 1. 更新动态消息 Provider
        self.message_provider.update_messages(request.messages)

        # 2. 确保后台记忆维护任务已启动
        if self.memory_manager is None and self.llm is not None:
            # 这里的 llm 假设从 runtime 获取，如果 runtime 没有则尝试从 request 获取
            self.memory_manager = MemoryMaintenanceManager(
                llm=self.llm,
                backend=backend,
                messages=self.message_provider
            )
            await self.memory_manager.start()
         
        # 3. 获取当前最新的记忆快照并格式化
        # 调用 format_memory_for_llm 默认读取所有类型的记忆 (profile, episodic, semantic)
        memory_body = format_memory_for_llm(backend)
        memory_context_str = get_middleware_prompt("context").format(agent_memory=memory_body)

        # 4. 构建环境上下文
        env_context_str = self._format_environment_context(runtime)

        injected_context_str = f"""
        {env_context_str}

        {memory_context_str}
        """.strip()

        # 5. 注入系统 Prompt
        request = request.override(
            system_message=append_to_system_message(request.system_message, injected_context_str)
        )

        # 6. 执行模型调用
        try:
            response = await handler(request)
            return response
        except Exception as e:
            logger.exception("ContextEngineeringMiddleware model call failed: %s", e)
            return ModelResponse(result=[AIMessage(content=f"Model call failed: {type(e).__name__}: {e}")])

    def _log_debug_info(self, ctx_data: str, total_count: int):
        print(f"--- [Context Engine] 注入数据: {ctx_data} | 消息流长度: {total_count} ---")
