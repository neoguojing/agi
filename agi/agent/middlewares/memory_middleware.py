import asyncio
import os
import time
import json
from typing import Callable, List, Awaitable
import logging
from langchain_core.messages import SystemMessage, BaseMessage
from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from deepagents.backends.protocol import BackendProtocol
from agi.utils.common import append_to_system_message, extract_messages_content
from agi.agent.context.checkpoint import checkpoint_to_state

logger = logging.getLogger(__name__)


MEMORY_SYSTEM_PROMPT = """<agent_memory>
{agent_memory}
</agent_memory>
"""

class MemoryMiddleware(AgentMiddleware):
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
        checkpointer = None,
        channels = None,
        config = None,
        memory_paths: List[str] = ["/memories/AGENT.md"]
    ):
        self.backend = backend
        self.checkpointer = checkpointer
        self.channels = channels
        self.config = config
        self.memory_paths = memory_paths

    def _get_backend(self, runtime) -> BackendProtocol:
        if callable(self.backend):
            return self.backend(runtime)
        return self.backend

    async def load_state_from_outside(self):
        cp = await self.checkpointer.aget(self.config)
        return checkpoint_to_state(cp, self.channels)


    def _format_agent_memory(self,runtime) -> str:
        if not self.memory_paths:
            return "(No memory loaded)"
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
            return "(No memory loaded)"

        memory_body = "\n\n".join(sections)
        return memory_body
    
    def _format_messages(self, messages: List[BaseMessage]) -> str:
        lines = []
        for msg in messages:
            role = msg.type
            content = extract_messages_content(msg)
            if role != "system" and content:
                lines.append(f"[{role}]: {content}")
        return "\n".join(lines)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]], 
    ) -> ModelResponse:
        try:
            state = await self.load_state_from_outside()
            messages = state["messages"]
            # formatted_messages = self._format_messages(messages)
            agent_memory = self._format_agent_memory(request.runtime)

            target_memory_prompt = MEMORY_SYSTEM_PROMPT.format(agent_memory=agent_memory)
            
            # 3. 注入系统 Prompt
            request = request.override(
                system_message=append_to_system_message(request.system_message, target_memory_prompt)
            )

            request.messages = messages

            response = await handler(request)

            return response
        except Exception as e:
            logger.error(e)
        
        
    