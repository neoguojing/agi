import asyncio
import os
import time
import json
from typing import Callable, List, Awaitable
from venv import logger
from langchain_core.messages import SystemMessage, BaseMessage
from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from deepagents.backends.protocol import BackendProtocol
from agi.agent.context import UnifiedContextManager
from agi.utils.common import append_to_system_message, extract_messages_content

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
        compression_threshold: int = 100, # 单条消息字符数阈值 N
        compression_dir: str = "/compressed_messages"
    ):
        self.backend = backend
        self.manager = UnifiedContextManager(llm_model=extractor_model)
        self.last_message_count = 0
        self.compression_threshold = compression_threshold
        self.compression_dir = compression_dir

    def _get_backend(self, runtime) -> BackendProtocol:
        if callable(self.backend):
            return self.backend(runtime)
        return self.backend
    
    async def _compress_messages(self, messages: List[BaseMessage], runtime) -> List[BaseMessage]:
        """
        遍历消息列表，对符合条件的单条消息进行压缩
        """
        if not messages:
            return messages

        # 1. 分离 SystemMessage (永远保留)
        system_messages = [msg for msg in messages if isinstance(msg, SystemMessage)]
        non_system_messages = [msg for msg in messages if not isinstance(msg, SystemMessage)]

        if not non_system_messages:
            return messages

        # 2. 分离最新 10 条非系统消息 (永远保留)
        recent_non_system_messages = non_system_messages[-10:]
        old_non_system_messages = non_system_messages[:-10]

        if not old_non_system_messages:
            return system_messages + recent_non_system_messages

        # 3. 遍历“老消息”，检查单条长度并压缩
        tasks = []
        
        for msg in old_non_system_messages:
            # 检查单条消息长度
            if len(msg.content) > self.compression_threshold:
                tasks.append(self._compress_single_message(msg, runtime))

        # 并行执行所有压缩任务
        if tasks:
            await asyncio.gather(*tasks)

        # 4. 重组消息列表
        return system_messages + old_non_system_messages + recent_non_system_messages

    async def _compress_single_message(self, msg: BaseMessage, runtime):
        """
        压缩单条消息：
        1. 使用 extract_messages_content 提取内容
        2. 直接写入 .txt 文件
        3. 原地替换 msg.content 为文件路径
        """
        # 1. 提取内容 (使用工具函数)
        # extract_messages_content 返回的是字符串列表，我们需要将其合并或直接取用
        # 对于单条消息，通常提取出来就是一个包含该消息内容的列表
        content_text = extract_messages_content(msg)

        # 2. 生成文件名 (.txt)
        timestamp = int(time.time() * 1000) 
        file_name = f"{msg.type}_{msg.id}_{timestamp}.txt"
        file_path = os.path.join(self.compression_dir, file_name)

        # 3. 异步写入文件 (直接写入纯文本)
        backend = self._get_backend(runtime)
        await backend.awrite(file_path, content_text)

        # 4. 原地替换内容
        msg.content = f"消息内容过长已压缩，文件路径: {file_path}"

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]], 
    ) -> ModelResponse:

        # 1. 获取上下文信息
        injected_context = await self.manager.get_context(request.runtime)
        injected_context_str = json.dumps(injected_context, ensure_ascii=False, indent=2)
        
        # 2. 执行消息压缩
        compressed_messages = await self._compress_messages(request.messages, request.runtime)
        
        # 3. 注入系统 Prompt
        request = request.override(
            messages=compressed_messages,
            system_message=append_to_system_message(request.system_message, injected_context_str)
        )

        # 4. 执行模型调用
        self._log_debug_info(injected_context_str, len(request.messages))
        response = await handler(request)
        
        # 5. 异步更新画像
        if len(request.messages) - self.last_message_count > 10:
            async def update_profile_task():
                try:
                    await self.manager.update(
                        runtime=request.runtime,
                        messages=request.messages[-10:]
                    )
                except Exception as e:
                    logger.error(f"Failed to update user profile: {e}")
        
            asyncio.create_task(update_profile_task())
            self.last_message_count = len(request.messages)
            
        return response

    def _log_debug_info(self, ctx_data: str, total_count: int):
        print(f"--- [Context Engine] 注入数据: {ctx_data} | 消息流长度: {total_count} ---")