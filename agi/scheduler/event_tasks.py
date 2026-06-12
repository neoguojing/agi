from agi.scheduler.task_hub import TaskContext,hub
from pydantic import BaseModel, Field


class ContextSummarySchema(BaseModel):
    verbose: bool = Field(default=True, description="是否打印微观全量明细表")
    limit: int = Field(default=5000, ge=1, le=10000, description="单次最大扫描的任务实例数")


@hub.event(
    event_name="event_summary", 
    runtime=None, 
    max_retries=3,
    timeout=60
)
async def handle_context_summary(ctx: TaskContext, payload: ContextSummarySchema):
    # 与 Cron 任务拥有完全一致的函数签名！
    print(f"[{ctx.trace_id}] 收到用户注册事件: {payload.user_id}")
    db = ctx.runtime.db_conn


from __future__ import annotations

import uuid
from dataclasses import dataclass
from functools import partial
from typing import Any, Mapping, cast

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

    # ========================================================================
    # public
    # ========================================================================

    def summarize(self, messages: list[AnyMessage]) -> SummarizationResult:
        self._ensure_message_ids(messages)

        total_tokens = self.token_counter(messages)

        if not self._should_summarize(messages, total_tokens):
            return SummarizationResult(
                summarized=False,
                summary="",
                total_tokens=total_tokens,
                cutoff_index=0,
                removed_messages=[],
                preserved_messages=messages,
                new_messages=messages,
            )

        cutoff_index = self._determine_cutoff_index(messages)

        if cutoff_index <= 0:
            return SummarizationResult(
                summarized=False,
                summary="",
                total_tokens=total_tokens,
                cutoff_index=0,
                removed_messages=[],
                preserved_messages=messages,
                new_messages=messages,
            )

        removed_messages = messages[:cutoff_index]
        preserved_messages = messages[cutoff_index:]

        summary = self._create_summary(removed_messages)

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

    async def asummarize(self, messages: list[AnyMessage]) -> SummarizationResult:
        self._ensure_message_ids(messages)

        total_tokens = self.token_counter(messages)

        if not self._should_summarize(messages, total_tokens):
            return SummarizationResult(
                summarized=False,
                summary="",
                total_tokens=total_tokens,
                cutoff_index=0,
                removed_messages=[],
                preserved_messages=messages,
                new_messages=messages,
            )

        cutoff_index = self._determine_cutoff_index(messages)

        if cutoff_index <= 0:
            return SummarizationResult(
                summarized=False,
                summary="",
                total_tokens=total_tokens,
                cutoff_index=0,
                removed_messages=[],
                preserved_messages=messages,
                new_messages=messages,
            )

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

    # ========================================================================
    # trigger
    # ========================================================================

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

    # ========================================================================
    # cutoff
    # ========================================================================

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

    # ========================================================================
    # summary
    # ========================================================================

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

    # ========================================================================
    # utils
    # ========================================================================

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


    
    summarizer = ConversationSummarizer(
    model=model,
    trigger=[("messages", 100), ("tokens", 60000)],
    keep=("messages", 20),
)

result = await summarizer.asummarize(messages)

if result.summarized:
    state["messages"] = result.new_messages

    memory_store.save_summary(
        summary=result.summary,
        source_messages=result.removed_messages,
    )