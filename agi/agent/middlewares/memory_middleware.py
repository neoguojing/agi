import asyncio
import os
import time
import json
from typing import Callable, List, Awaitable
import logging
from langchain_core.messages import SystemMessage, BaseMessage,AnyMessage,HumanMessage
from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from deepagents.backends.protocol import BackendProtocol
from agi.utils.common import append_to_system_message, extract_messages_content
from agi.agent.context.checkpoint import checkpoint_to_state

"""
记忆类型：
1. Profile Memory（稳定人格/偏好）： 结构化存储，KV cache
2. Episodic Memory（事件记忆）：时间性强 ，会过期 
3. Semantic Memory（知识抽象）： 这是从 episodic 中提炼出的规律，长期保留；类似人脑 consolidation。推荐知识图谱化

记忆整理：
不要一直加 memory。
要周期性“重写 memory”。
1. 去重（Dedup）
2. 冲突检测
3. 重要度衰减：很多 memory 应该 decay。
4. Episodic → Semantic 抽象

核心原则：

1. Hierarchical Memory： 分层抽象
2. Graph Memory：知识图谱画
3. Self-healing Memory：周期性：

contradiction detection
confidence recalculation
stale pruning
4. Memory as Dataset

未来趋势：

memory 不再是 prompt。

而是：

可训练数据集
"""
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
    """
    def __init__(
        self, 
        backend = None,
        checkpointer = None,
        channels = None,
        config = None,
        memory_paths: List[str] =  ["/memories/facts.md","/memories/preferences.md","/memories/lessons.md"]
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

    def _apply_event_to_messages(
        self,
        messages: list[AnyMessage],
        event: dict | None,
    ) -> list[AnyMessage]:
        """Reconstruct effective messages from raw state messages and a summarization event.

        When a prior summarization event exists, the effective conversation is
        the summary message followed by all messages from `cutoff_index` onward.

        Args:
            messages: Full message list from state.
            event: The `_summarization_event` dict, or `None`.

        Returns:
            The effective message list the model would see.
        """
        if event is None:
            return list(messages)

        try:
            summary_msg = event["summary_message"]
            cutoff_idx = event["cutoff_index"]
        except (KeyError, TypeError) as exc:
            logger.warning("Malformed _summarization_event (missing keys): %s", exc)
            return list(messages)

        if cutoff_idx > len(messages):
            logger.warning(
                "Summarization cutoff_index %d exceeds message count %d; remaining slice will be empty",
                cutoff_idx,
                len(messages),
            )
            return [summary_msg]

        result: list[AnyMessage] = [summary_msg]
        result.extend(messages[cutoff_idx:])
        return result

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
            messages = state.get("messages") or []
            summay_event = state["_summarization_event"]
            if summay_event is not None:
                messages = self._apply_event_to_messages(messages,summay_event)
            # formatted_messages = self._format_messages(messages)
            agent_memory = self._format_agent_memory(request.runtime)

            target_memory_prompt = MEMORY_SYSTEM_PROMPT.format(agent_memory=agent_memory)
            
            # 3. 注入系统 Prompt
            request = request.override(
                system_message=append_to_system_message(request.system_message, target_memory_prompt)
            )

            last_message = request.messages[-1]
            messages.append(last_message)
            if request.messages[-1].type != "human":
                messages.append(HumanMessage(content="Background maintenance tick: analyze memory needs."))
            request.messages = messages

            response = await handler(request)

            return response
        except Exception as e:
            import traceback
            traceback.print_exc()
            logger.error(e)
        
        
    