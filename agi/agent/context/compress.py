import os
import asyncio
import time
from typing import List, Tuple
from langchain_core.messages import BaseMessage, SystemMessage
from agi.utils.common import extract_messages_content

class ContextCompressor:
    """
    认知缓冲区压缩引擎：
    负责将活跃记忆(Active Memory)转化为温存储(Warm Storage)
    """
    def __init__(
        self, 
        threshold: int = 300, 
        storage_dir: str = "/compressed_messages",
        keep_recent: int = 10
    ):
        self.threshold = threshold
        self.storage_dir = storage_dir
        self.keep_recent = keep_recent

    async def compress(self, messages: List[BaseMessage], backend) -> List[BaseMessage]:
        """执行压缩流水线"""
        if not messages:
            return messages

        # 1. 路由分发：保护系统消息与活跃消息(Hot Tier)
        system_msgs = [m for m in messages if isinstance(m, SystemMessage)]
        other_msgs = [m for m in messages if not isinstance(m, SystemMessage)]

        if len(other_msgs) <= self.keep_recent:
            return messages

        hot_tier = other_msgs[-self.keep_recent:]
        warm_tier = other_msgs[:-self.keep_recent]

        # 2. 对旧消息(Warm Tier)执行并行压缩
        tasks = []
        for msg in warm_tier:
            if len(msg.content) > self.threshold:
                tasks.append(self._compress_to_file(msg, backend))
            else:
                tasks.append(asyncio.sleep(0, result=msg)) # 保持原样

        compressed_warm = await asyncio.gather(*tasks)
        
        # 3. 重组
        return system_msgs + list(compressed_warm) + hot_tier

    async def _compress_to_file(self, msg: BaseMessage, backend) -> BaseMessage:
        """物理压缩：转存为 .txt"""
        content_text = extract_messages_content(msg)
        
        # 即使提取结果是列表，通常单条消息返回单字符串，这里确保鲁棒性
        raw_text = "".join(content_text) if isinstance(content_text, list) else content_text
        
        file_name = f"{msg.type}_{getattr(msg, 'id', 'unknown')}_{int(time.time())}.txt"
        file_path = os.path.join(self.storage_dir, file_name)

        # 执行后端异步写入
        await backend.awrite(file_path, raw_text)

        # 原地更新或返回新对象（LangChain 建议返回新对象或修改内容）
        summary = raw_text[:50].replace("\n", " ")
        msg.content = f"[WARM_STORAGE] {summary}... path: {file_path}, hint: read file for full context"
        return msg