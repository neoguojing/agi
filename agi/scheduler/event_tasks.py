from typing import Any, List,cast
from pydantic import BaseModel, Field

# 🔄 环境抽象：复用共享的 Runtime 类
from agi.scheduler.memory_task.memory_state import memory_manager
from agi.scheduler.memory_task.runtime import MemoryTaskRuntime
from agi.scheduler.memory_task.memory_models import SummaryRecord
from agi.scheduler.task_hub import TaskContext, hub

from __future__ import annotations
import uuid
from dataclasses import dataclass
from functools import partial
from typing import Mapping
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    ToolMessage,
    AnyMessage,
    get_buffer_string,
)
from langchain_core.messages.utils import trim_messages
from langchain_core.messages import count_tokens_approximately

import logging
logger = logging.getLogger(__name__)
# ------------------------------------------------------------------------------
# 📐 2. 参数契约模型
# ------------------------------------------------------------------------------
class ContextSummarySchema(BaseModel):
    verbose: bool = Field(default=True, description="是否打印微观全量明细表")
    limit: int = Field(default=5000, ge=1, le=10000, description="单次最大扫描的任务实例数")


# ------------------------------------------------------------------------------
# ⛓️ 3. 核心业务逻辑管线
# ------------------------------------------------------------------------------

ContextSize = tuple[str, int | float]

DEFAULT_MESSAGES_TO_KEEP = 20
DEFAULT_TRIM_TOKEN_LIMIT = 12000
DEFAULT_FALLBACK_MESSAGE_COUNT = 100

DEFAULT_SUMMARY_PROMPT = """
You are a conversation summarization assistant.

Summarize the following conversation.

Conversation:
{messages}

Requirements:
- Preserve important facts.
- Preserve user preferences.
- Preserve decisions and conclusions.
- Preserve unresolved tasks.
- Remove repetitive content.

Summary:
""".strip()

@dataclass(slots=True)
class SummarizationResult:
    summarized: bool
    summary: str
    total_tokens: int
    cutoff_index: int
    removed_messages: list[AnyMessage]
    preserved_messages: list[AnyMessage]
    new_messages: list[AnyMessage]

class ConversationSummarizer:
    def __init__(
        self,
        model: BaseChatModel,
        *,
        trigger: ContextSize | list[ContextSize] | None = None,
        keep: ContextSize = ("messages", DEFAULT_MESSAGES_TO_KEEP),
        token_counter=count_tokens_approximately,
        summary_prompt: str = DEFAULT_SUMMARY_PROMPT,
        trim_tokens_to_summarize: int | None = DEFAULT_TRIM_TOKEN_LIMIT,
    ):
        self.model = model
        self.keep = keep
        self.summary_prompt = summary_prompt
        self.trim_tokens_to_summarize = trim_tokens_to_summarize

        if trigger is None:
            self.trigger_conditions: list[ContextSize] = []
        elif isinstance(trigger, list):
            self.trigger_conditions = trigger
        else:
            self.trigger_conditions = [trigger]

        if token_counter is count_tokens_approximately:
            self.token_counter = partial(
                count_tokens_approximately,
                use_usage_metadata_scaling=False,
            )
        else:
            self.token_counter = token_counter

    def _should_summarize(self, messages: list[AnyMessage], total_tokens: int) -> bool:
        if not self.trigger_conditions:
            return False

        for kind, value in self.trigger_conditions:
            if kind == "messages" and len(messages) >= value:
                return True
            if kind == "tokens" and total_tokens >= value:
                return True
            if kind == "fraction":
                max_tokens = self._get_profile_limit()
                if not max_tokens:
                    continue
                if total_tokens >= int(max_tokens * value):
                    return True

        return False

    def _determine_cutoff_index(self, messages: list[AnyMessage]) -> int:
        kind, value = self.keep

        if kind == "messages":
            return self._find_safe_cutoff(messages, int(value))

        if kind == "tokens":
            return self._find_token_based_cutoff(messages, int(value))

        if kind == "fraction":
            max_tokens = self._get_profile_limit()
            if not max_tokens:
                return self._find_safe_cutoff(
                    messages,
                    DEFAULT_MESSAGES_TO_KEEP,
                )

            return self._find_token_based_cutoff(
                messages,
                int(max_tokens * value),
            )

        return 0

    def _find_safe_cutoff(self, messages: list[AnyMessage], keep_count: int) -> int:
        if len(messages) <= keep_count:
            return 0

        target_cutoff = len(messages) - keep_count
        return self._find_safe_cutoff_point(messages, target_cutoff)

    def _find_token_based_cutoff(
        self,
        messages: list[AnyMessage],
        target_tokens: int,
    ) -> int:
        if not messages:
            return 0

        if self.token_counter(messages) <= target_tokens:
            return 0

        left, right = 0, len(messages)
        cutoff = len(messages)
        while left < right:
            mid = (left + right) // 2
            if self.token_counter(messages[mid:]) <= target_tokens:
                cutoff = mid
                right = mid
            else:
                left = mid + 1

        if cutoff >= len(messages):
            cutoff = len(messages) - 1

        return self._find_safe_cutoff_point(messages, cutoff)

    @staticmethod
    def _find_safe_cutoff_point(
        messages: list[AnyMessage],
        cutoff_index: int,
    ) -> int:
        if (
            cutoff_index >= len(messages)
            or not isinstance(messages[cutoff_index], ToolMessage)
        ):
            return cutoff_index

        tool_call_ids: set[str] = set()
        idx = cutoff_index
        while idx < len(messages):
            msg = messages[idx]
            if not isinstance(msg, ToolMessage):
                break
            if msg.tool_call_id:
                tool_call_ids.add(msg.tool_call_id)
            idx += 1

        for i in range(cutoff_index - 1, -1, -1):
            msg = messages[i]
            if not isinstance(msg, AIMessage):
                continue
            if not msg.tool_calls:
                continue
            ai_tool_call_ids = {
                tc.get("id")
                for tc in msg.tool_calls
                if tc.get("id")
            }
            if tool_call_ids & ai_tool_call_ids:
                return i

        return idx

    def _create_summary(self, messages: list[AnyMessage]) -> str:
        if not messages:
            return ""
        trimmed = self._trim_messages(messages)
        if not trimmed:
            return ""
        prompt = self.summary_prompt.format(
            messages=get_buffer_string(trimmed)
        )
        try:
            response = self.model.invoke(prompt)
            return response.text.strip()
        except Exception as e:
            return f"Summary failed: {e}"

    async def asummarize(self, messages: list[AnyMessage]) -> SummarizationResult:
        if not messages:
            return SummarizationResult(False, "", 0, 0, [], messages, messages)

        total_tokens = self.token_counter(messages)
        if not self._should_summarize(messages, total_tokens):
            return SummarizationResult(False, "", total_tokens, 0, [], messages, messages)

        cutoff_index = self._determine_cutoff_index(messages)
        if cutoff_index <= 0:
            return SummarizationResult(False, "", total_tokens, 0, [], messages, messages)

        removed_messages = messages[:cutoff_index]
        preserved_messages = messages[cutoff_index:]
        summary = await self._acreate_summary(removed_messages)

        new_messages = [
            HumanMessage(
                content=f"Conversation Summary:\n\n{summary}",
                additional_kwargs={"lc_source": "summarization"},
            ),
            *preserved_messages,
        ]

        return SummarizationResult(
            summarized=True,
            summary=summary,
            total_tokens=total_tokens,
            cutoff_index=cutoff_index,
            removed_messages=removed_messages,
            preserved_messages=preserved_messages,
            new_messages=new_messages,
        )

    async def _acreate_summary(self, messages: list[AnyMessage]) -> str:
        if not messages:
            return ""
        trimmed = self._trim_messages(messages)
        if not trimmed:
            return ""
        prompt = self.summary_prompt.format(
            messages=get_buffer_string(trimmed)
        )
        try:
            response = await self.model.ainvoke(prompt)
            return response.text.strip()
        except Exception as e:
            return f"Summary failed: {e}"

    def _trim_messages(self, messages: list[AnyMessage]) -> list[AnyMessage]:
        try:
            if self.trim_tokens_to_summarize is None:
                return messages
            return cast(
                list[AnyMessage],
                trim_messages(
                    messages,
                    max_tokens=self.trim_tokens_to_summarize,
                    token_counter=self.token_counter,
                    strategy="last",
                    start_on="human",
                    allow_partial=True,
                    include_system=True,
                ),
            )
        except Exception:
            return messages[-DEFAULT_FALLBACK_MESSAGE_COUNT:]

    def _get_profile_limit(self) -> int | None:
        try:
            profile = self.model.profile
        except Exception:
            return None
        if not isinstance(profile, Mapping):
            return None
        limit = profile.get("max_input_tokens")
        return limit if isinstance(limit, int) else None

    @staticmethod
    def _ensure_message_ids(messages: list[AnyMessage]) -> None:
        for msg in messages:
            if getattr(msg, "id", None) is None:
                msg.id = str(uuid.uuid4())


# ------------------------------------------------------------------------------
# 🚀 执行流水线 (Pipeline Implementation)
# ------------------------------------------------------------------------------

async def execute_context_summary_pipeline(ctx: TaskContext, payload: ContextSummarySchema):
    """
    💡 事件驱动核心流水线
    1. 初始化运行时环境
    2. 获取消息历史与 Token 计算
    3. 根据触发条件判断是否摘要
    4. 调用 LLM 生成摘要并更新到 memory_state 中的 summary_records
    """
    runtime = cast(MemoryTaskRuntime, ctx.runtime)

    # 复用原有的 Summarizer 类作为逻辑内核
    summarizer = ConversationSummarizer(
        model=runtime.llm,
        trigger=[("messages", 100), ("tokens", 60000)],
        keep=("messages", 20),
        summary_prompt=DEFAULT_SUMMARY_PROMPT,
        trim_tokens_to_summarize=DEFAULT_TRIM_TOKEN_LIMIT,
    )

    # 获取当前上下文消息 (来自 TaskContext 或 Runtime 的 message 缓存)
    # 在事件触发场景下，ctx.messages 通常包含了引发事件的对话内容
    current_messages = ctx.messages if hasattr(ctx, "messages") else []
    if not current_messages:
        # 如果 Context 中没带消息，则尝试从 Runtime 的 message list 获取
        current_messages = runtime._messages

    total_tokens = summarizer.token_counter(current_messages)

    # 判断是否需要摘要
    should_summarize = False
    if summarizer.trigger_conditions:
        for kind, value in summarizer.trigger_conditions:
            if kind == "messages" and len(current_messages) >= value:
                should_summarize = True
            if kind == "tokens" and total_tokens >= value:
                should_summarize = True

    if not should_summarize:
        logger.info("[event_summary] ⏳ 当前内容未达摘要触发条件 (Tokens: %d, Msgs: %d)",
                    total_tokens, len(current_messages))
        return

    # 执行核心异步摘要逻辑
    result = await summarizer.asummarize(current_messages)

    if result.summarized:
        logger.info("[event_summary] ✅ 已生成新摘要。长度: %d", len(result.summary))

        # 🌟 更新持久化状态到 memory_state.py 管理的 summary_records 中
        # 我们使用 source_conversation_id 来确保记录是唯一的 (de_dupe)
        # 这里可以使用 ctx.thread_id 作为来源标识
        new_record = SummaryRecord(
            summary=result.summary,
            source_conversation_id=runtime.thread_id,
            confidence=0.95 if result.total_tokens < 50000 else 0.85
        )

        # 使用 MemoryManager 的原子化写入逻辑
        await memory_manager.commit_incremental_memory(
            task_type="summary",
            memory_value=new_record,
            reason=f"Automated summary triggered by event. Source IDs: {result.cutoff_index} items removed."
        )

        # 更新当前上下文中的消息（如果需要的话）
        if hasattr(ctx, "messages"):
            ctx.messages = result.new_messages


# ------------------------------------------------------------------------------
# 📡 事件入口 (Event Entry Points)
# ------------------------------------------------------------------------------

@hub.event(
    event_name="event_summary",
    runtime=me,
    max_retries=3,
    timeout=60
)
async def handle_context_summary(ctx: TaskContext, payload: ContextSummarySchema):
    """Handle context summary event - Triggered by external events."""
    logger.info("[%s] 📩 收到事件 'event_summary' 触发请求。", ctx.trace_id)
    await execute_context_summary_pipeline(ctx, payload)
