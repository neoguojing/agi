import json
from typing import Callable, List, Awaitable
from venv import logger
from langchain_core.messages import SystemMessage, BaseMessage
from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from deepagents.backends.protocol import BackendProtocol
from agi.agent.context import UnifiedContextManager,ContextCompressor
from agi.agent.prompt import get_middleware_prompt
from agi.utils.common import append_to_system_message

class ContextEngineeringMiddleware(AgentMiddleware):
    """
    上下文工程中间件：
    1. 动态注入模型 Prompt
    2. 异步更新用户画像
    3. 消息压缩策略：
       - 保护 SystemMessage 和最新 10 条消息
       - 单条消息超过阈值则压缩
       - 文件保存为 .txt 纯文本格式
    """
    def __init__(
        self, 
        extractor_model,
        backend = None,
        memory_paths: List[str] = ["/memories/AGENT.md"]

    ):
        self.backend = backend
        self.memory_paths = memory_paths
        self.compressor = ContextCompressor()

    def _get_backend(self, runtime) -> BackendProtocol:
        if callable(self.backend):
            return self.backend(runtime)
        return self.backend
    
    def _format_agent_memory(self,runtime) -> str:
        if not self.memory_paths:
            return get_middleware_prompt("context").format(agent_memory="(No memory loaded)")
        
        backend = self._get_backend(runtime)  # 这里传 None，因为我们只需要读取文件内容
        contents = {}
        for path in self.memory_paths:
            try:
                content = backend.read(path)
                contents[path] = content
            except Exception as e:
                logger.error(f"Failed to read memory file {path}: {e}")
                contents[path] = None   

        sections = [f"{path}\n{contents[path]}" for path in self.memory_paths if contents.get(path)]
        if not sections:
            return get_middleware_prompt("context").format(agent_memory="(No memory loaded)")

        memory_body = "\n\n".join(sections)
        return get_middleware_prompt("context").format(agent_memory=memory_body)
    
    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]], 
    ) -> ModelResponse:
        import pdb; pdb.set_trace()

        request.runtime.context.set_incremental_messages(request.messages)
        # 1. 获取上下文信息
        injected_context_str = self._format_agent_memory(request.runtime)
        # 2. 执行消息压缩
        # backend = self._get_backend(request.runtime)
        # compressed_messages = await self.compressor.compress(request.messages, backend)
        
        # 3. 注入系统 Prompt
        request = request.override(
            # messages=compressed_messages,
            system_message=append_to_system_message(request.system_message, injected_context_str)
        )

        # 4. 执行模型调用
        self._log_debug_info(injected_context_str, len(request.messages))
        response = await handler(request)
        
        return response

    def _log_debug_info(self, ctx_data: str, total_count: int):
        print(f"--- [Context Engine] 注入数据: {ctx_data} | 消息流长度: {total_count} ---")