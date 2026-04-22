import json
import platform
import datetime
import os
from typing import Callable, List, Awaitable
from venv import logger
from langchain_core.messages import SystemMessage, BaseMessage
from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from deepagents.backends.protocol import BackendProtocol
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
        backend = None,
        memory_paths: List[str] = ["/memories/facts.md","/memories/preferences.md","/memories/lessons.md"]

    ):
        self.backend = backend
        self.memory_paths = memory_paths
        self.is_file_inited = False

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
    
    def create_files(self,backend):
        for path in self.memory_paths:
            file_info = backend.ls_info(path)
            if not file_info:
                backend.upload_files([(path,b"")])
        
        self.is_file_inited = True


    def _format_agent_memory(self,runtime) -> str:
        if not self.memory_paths:
            return get_middleware_prompt("context").format(agent_memory="(No memory loaded)")
        
        backend = self._get_backend(runtime)  # 这里传 None，因为我们只需要读取文件内容
        if not self.is_file_inited:
            self.create_files(backend)

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
        
        env_context_str = self._format_environment_context(request.runtime)
        # 1. 获取上下文信息
        memory_context_str = self._format_agent_memory(request.runtime)

        injected_context_str = f"""
        {env_context_str}

        {memory_context_str}
        """.strip()

        # 3. 注入系统 Prompt
        request = request.override(
            system_message=append_to_system_message(request.system_message, injected_context_str)
        )

        # 4. 执行模型调用
        self._log_debug_info(injected_context_str, len(request.messages))
        response = await handler(request)
        
        return response

    def _log_debug_info(self, ctx_data: str, total_count: int):
        print(f"--- [Context Engine] 注入数据: {ctx_data} | 消息流长度: {total_count} ---")