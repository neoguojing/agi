from __future__ import annotations
from typing import Any, List, cast, Mapping, Optional, Literal, Callable, Iterable
from pydantic import BaseModel, Field
import logging
from functools import partial
from datetime import UTC, datetime

from agi.scheduler.memory_task.memory_state import memory_manager
from agi.scheduler.memory_task.runtime import MemoryTaskRuntime, memory_runtime
from agi.scheduler.memory_task.memory_models import SummaryRecord
from agi.scheduler.task_hub import TaskContext, hub

from langchain_core.language_models import BaseChatModel
from deepagents.backends.protocol import BackendProtocol
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    ToolMessage,
    AnyMessage,
    get_buffer_string,
    convert_to_messages,
)
from langchain_core.messages.utils import trim_messages, count_tokens_approximately

from langgraph.graph.message import REMOVE_ALL_MESSAGES

import logging
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------
# 📐 1. Parameter Contracts & Constants
# ------------------------------------------------------------------------------

DEFAULT_SUMMARY_PROMPT = """<role>
Context Extraction Assistant
</role>

<primary_objective>
Your sole objective in this task is to extract the highest quality/most relevant context from the conversation history below.
</primary_objective>

<objective_information>
You're nearing the total number of input tokens you can accept, so you must extract the highest quality/most relevant pieces of information from your conversation history.
This context will then overwrite the conversation history presented below. Because of this, ensure the context you extract is only the most important information to continue working toward your overall goal.
</objective_information>

<instructions>
The conversation history below will be replaced with the context you extract in this step.
You want to ensure that you don't repeat any actions you've already completed, so the context you extract from the conversation history should be focused on the most important information to your overall goal.

You should structure your summary using the following sections. Each section acts as a checklist - you must populate it with relevant information or explicitly state "None" if there is nothing to report for that section:

## SESSION INTENT

What is the user's primary goal or request? What overall task are you trying to accomplish? This should be concise but complete enough to understand the purpose of the entire session.

## SUMMARY

Extract and record all of the most important context from the conversation history. Include important choices, conclusions, or strategies determined during this conversation. Include the reasoning behind key decisions. Document any rejected options and why they were not pursued.

## ARTIFACTS

What artifacts, files, or resources were created, modified, or accessed during this conversation? For file modifications, list specific file paths and briefly describe the changes made to each. This section prevents silent loss of artifact information.

## NEXT STEPS

What specific tasks remain to be completed to achieve the session intent? What should you do next?

</instructions>

The user will message you with the full message history from which you'll extract context to create a replacement. Carefully read through it all and think deeply about what information is most important to your overall goal and should be saved:

With all of this in mind, please carefully read over the entire conversation history, and extract the most important and relevant context to replace it so that you can free up space in the conversation history.
Respond ONLY with the extracted context. Do not include any additional information, or text before or after the extracted context.

<messages>
Messages to summarize:
{messages}
</messages>"""

_DEFAULT_MESSAGES_TO_KEEP = 20
_DEFAULT_TRIM_TOKEN_LIMIT = 4000
_DEFAULT_FALLBACK_MESSAGE_COUNT = 15
_MAX_ARG_LENGTH = 2000
_TRUNCATION_TEXT = "...(argument truncated)"

class ContextSummarySchema(BaseModel):
    current_message_count: int = Field(..., description="当前会话中的总消息条数")
    current_token_count: int = Field(..., description="当前会话的总 Token 数")
    msg_threshold: Optional[int] = Field(None, description="触发摘要的消息条数阈值")
    token_threshold: Optional[int] = Field(None, description="触发摘要的 Token 数量阈值")

# ------------------------------------------------------------------------------
# ⚙️ 2. Core Logic Helpers
# ------------------------------------------------------------------------------

def _get_approximate_token_counter(model: BaseChatModel) -> Callable:
    """
    获取一个近似的 Token 计数器。
    用于执行压缩过程中的精密切片（如二分查找截断点）和输入裁剪。
    """
    if hasattr(model, '_llm_type') and model._llm_type.startswith("anthropic-chat"):
        return partial(count_tokens_approximately, use_usage_metadata_scaling=True, chars_per_token=3.3)
    return partial(count_tokens_approximately, use_usage_metadata_scaling=True)

def _should_summarize(messages: List[AnyMessage], total_tokens: int, trigger_conditions: List[tuple], model: BaseChatModel) -> bool:
    """
    判定是否需要执行摘要压缩。
    检查当前上下文是否满足任何一个触发阈值（消息数、Token 数或 Token 比例）。
    """
    if not trigger_conditions:
        return False
    for kind, value in trigger_conditions:
        if kind == "messages" and len(messages) >= value:
            return True
        if kind == "tokens" and total_tokens >= value:
            return True
        if kind == "fraction":
            try:
                max_input_tokens = getattr(model, "profile", {}).get("max_input_tokens", 0)
                if max_input_tokens and total_tokens >= int(max_input_tokens * value):
                    return True
            except Exception:
                pass
    return False

def _find_safe_cutoff_point(messages: List[AnyMessage], cutoff_index: int) -> int:
    """
    【算法：AI/Tool 消息对保护】
    确保截断点不会将 AIMessage (Tool Call) 和对应的 ToolMessage (Tool Response) 分离开。
    """
    if cutoff_index >= len(messages) or not isinstance(messages[cutoff_index], ToolMessage):
        return cutoff_index
    tool_call_ids: set[str] = set()
    idx = cutoff_index
    while idx < len(messages) and isinstance(messages[idx], ToolMessage):
        tool_msg = messages[idx]
        if tool_msg.tool_call_id:
            tool_call_ids.add(tool_msg.tool_call_id)
        idx += 1
    for i in range(cutoff_index - 1, -1, -1):
        msg = messages[i]
        if isinstance(msg, AIMessage) and msg.tool_calls:
            ai_tool_call_ids = {tc.get("id") for tc in msg.tool_calls if tc.get("id")}
            if tool_call_ids & ai_tool_call_ids:
                return i
    return idx

def _determine_cutoff_index(messages: List[AnyMessage], token_counter: Callable, keep_policy: tuple) -> int:
    """
    【算法：动态截断点计算】
    计算需要保留的上下文窗口边界。
    """
    kind, value = keep_policy
    if kind == "messages":
        if len(messages) <= value:
            return 0
        target_cutoff = len(messages) - value
        return _find_safe_cutoff_point(messages, target_cutoff)

    if kind in {"tokens", "fraction"}:
        target_token_count = value if kind == "tokens" else 0
        if target_token_count <= 0:
            return _find_safe_cutoff_point(messages, max(0, len(messages) - _DEFAULT_MESSAGES_TO_KEEP))

        if token_counter(messages) <= target_token_count:
            return 0

        left, right = 0, len(messages)
        cutoff_candidate = len(messages)
        for _ in range(len(messages).bit_length() + 1):
            if left >= right: break
            mid = (left + right) // 2
            if token_counter(messages[mid:]) <= target_token_count:
                cutoff_candidate = mid
                right = mid
            else:
                left = mid + 1
        if cutoff_candidate >= len(messages):
            cutoff_candidate = max(0, len(messages) - 1)
        return _find_safe_cutoff_point(messages, cutoff_candidate)

    return _find_safe_cutoff_point(messages, max(0, len(messages) - _DEFAULT_MESSAGES_TO_KEEP))

def _truncate_tool_call(tool_call: dict[str, Any]) -> dict[str, Any]:
    args = tool_call.get("args", {})
    truncated_args = {}
    modified = False
    for key, value in args.items():
        if isinstance(value, str) and len(value) > _MAX_ARG_LENGTH:
            truncated_args[key] = value[:20] + _TRUNCATION_TEXT
            modified = True
        else:
            truncated_args[key] = value
    if modified:
        return {**tool_call, "args": truncated_args}
    return tool_call

def _truncate_args(messages: List[AnyMessage]) -> tuple[List[AnyMessage], bool]:
    cutoff_index = len(messages) - _DEFAULT_MESSAGES_TO_KEEP
    if cutoff_index <= 0:
        return messages, False

    truncated_messages = []
    modified = False
    for i, msg in enumerate(messages):
        if i < cutoff_index and isinstance(msg, AIMessage) and msg.tool_calls:
            truncated_tool_calls = []
            msg_modified = False
            for tool_call in msg.tool_calls:
                if tool_call.get("name") in {"write_file", "edit_file"}:
                    truncated_call = _truncate_tool_call(tool_call)
                    if truncated_call != tool_call:
                        msg_modified = True
                    truncated_tool_calls.append(truncated_call)
                else:
                    truncated_tool_calls.append(tool_call)
            if msg_modified:
                truncated_msg = msg.model_copy()
                truncated_msg.tool_calls = truncated_tool_calls
                truncated_messages.append(truncated_msg)
                modified = True
            else:
                truncated_messages.append(msg)
        else:
            truncated_messages.append(msg)
    return truncated_messages, modified

async def _offload_to_backend(backend: Any, messages: List[AnyMessage], thread_id: str) -> Optional[str]:
    path = f"/conversation_history/{thread_id}.md"
    filtered_messages = [msg for msg in messages if not (isinstance(msg, HumanMessage) and msg.additional_kwargs.get("lc_source") == "summarization")]
    timestamp = datetime.now(UTC).isoformat()
    new_section = f"## Summarized at {timestamp}\n\n{get_buffer_string(filtered_messages)}\n\n"

    existing_content = ""
    try:
        responses = await backend.adownload_files([path]) if hasattr(backend, 'adownload_files') else []
        if responses and responses[0].content is not None and responses[0].error is None:
            existing_content = responses[0].content.decode("utf-8")
    except Exception:
        pass

    combined_content = existing_content + new_section
    try:
        if existing_content:
            await backend.aedit(path, existing_content, combined_content)
        else:
            await backend.awrite(path, combined_content)
        return path
    except Exception as e:
        logger.warning("Offload failed: %s", e)
        return None

# ------------------------------------------------------------------------------
# 🚀 3. Event-Driven Pipeline
# ------------------------------------------------------------------------------

async def execute_context_summary_pipeline(ctx: TaskContext, payload: ContextSummarySchema):
    runtime = cast(MemoryTaskRuntime, ctx.runtime)
    token_counter = _get_approximate_token_counter(runtime.llm)

    logger.info("[%s] [event_summary] 🚀 Starting context summary pipeline...", ctx.trace_id)

    effective_messages = await memory_manager.get_effective_summary_context()
    if not effective_messages:
        logger.info("[%s] [event_summary] ⏭️ No messages to summarize.", ctx.trace_id)
        return

    effective_messages, _ = _truncate_args(effective_messages)

    keep_policy = ("messages", _DEFAULT_MESSAGES_TO_KEEP)
    relative_cutoff = _determine_cutoff_index(effective_messages, token_counter, keep_policy)

    if relative_cutoff <= 0:
        logger.info("[%s] [event_summary] ⏭️ Cutoff index 0, nothing to compress.", ctx.trace_id)
        return

    logger.info("[%s] [event_summary] ✂️ Relative cutoff: %d", ctx.trace_id, relative_cutoff)

    thread_id = runtime.thread_id or "unknown"
    prev_summary = await memory_manager.get_current_summary()

    if prev_summary:
        prev_cutoff = getattr(prev_summary, "cutoff_index", 0)
        absolute_cutoff = prev_cutoff + max(0, relative_cutoff - 1)
    else:
        absolute_cutoff = relative_cutoff

    logger.info("[%s] [event_summary] 🎯 Absolute cutoff: %d (prev_cutoff: %d)",
                ctx.trace_id, absolute_cutoff, getattr(prev_summary, "cutoff_index", 0) if prev_summary else 0)

    full_messages = await memory_manager.refresh_messages()
    full_messages = convert_to_messages(full_messages)

    messages_to_summarize = full_messages[:absolute_cutoff]
    logger.info("[%s] [event_summary] 📦 Offloading %d messages to backend...", ctx.trace_id, len(messages_to_summarize))

    def _get_backend(runtime) -> BackendProtocol:
        backend = runtime.backend
        if callable(backend):
            return backend(runtime)
        return backend

    backend = _get_backend(runtime)
    file_path = await _offload_to_backend(backend, messages_to_summarize, thread_id)
    logger.info("[%s] [event_summary] ✅ Offload complete. Path: %s", ctx.trace_id, file_path)

    summarize_input = effective_messages[:relative_cutoff]
    trimmed = trim_messages(
        summarize_input,
        max_tokens=_DEFAULT_TRIM_TOKEN_LIMIT,
        token_counter=token_counter,
        strategy="last",
        start_on="human",
        allow_partial=True,
        include_system=True,
    )

    if not trimmed:
        trimmed = summarize_input[-_DEFAULT_FALLBACK_MESSAGE_COUNT:]

    logger.info("[%s] [event_summary] 📝 Sending %d messages to LLM for summary (approx %d tokens)",
                ctx.trace_id, len(trimmed), token_counter(trimmed))

    prompt = DEFAULT_SUMMARY_PROMPT.format(messages=get_buffer_string(trimmed))
    try:
        response = await runtime.llm.ainvoke(prompt)
        summary_text = response.text.strip()
        logger.info("[%s] [event_summary] ✨ Summary generated. Length: %d chars", ctx.trace_id, len(summary_text))
    except Exception as e:
        logger.error("[%s] [event_summary] ❌ LLM Summary Error: %s", ctx.trace_id, e)
        return

    new_record = SummaryRecord(
        summary=summary_text,
        source_conversation_id=thread_id,
        cutoff_index=absolute_cutoff,
        file_path=file_path,
        reason = f"Event summary triggered {payload}. Absolute cutoff: {absolute_cutoff} messages."
    )

    await memory_manager.set_current_summary(record=new_record)
    logger.info("[%s] [event_summary] 💾 New summary record saved. Cutoff: %d", ctx.trace_id, absolute_cutoff)


    

# ------------------------------------------------------------------------------
# 📡 Event Entry Point
# ------------------------------------------------------------------------------

@hub.event(
    event_name="event_summary",
    runtime=memory_runtime,
    max_retries=3,
    timeout=60
)
async def handle_context_summary(ctx: TaskContext, payload: ContextSummarySchema):
    logger.info("[%s] 📩 Received 'event_summary' trigger.payload=%s", ctx.trace_id,payload)
    runtime = cast(MemoryTaskRuntime, ctx.runtime)
    effective_messages = await memory_manager.get_effective_summary_context()
    if not effective_messages:
        return
    triggers = []
    if payload.msg_threshold is not None:
        triggers.append(("messages", payload.msg_threshold))
    else:
        triggers.append(("messages", 100))
    if payload.token_threshold is not None:
        triggers.append(("tokens", payload.token_threshold))
    else:
        triggers.append(("tokens", 60000))
    if not _should_summarize(effective_messages, payload.current_token_count, triggers, runtime.llm):
        logger.info("[event_summary] ⏳ Skipping: Thresholds not met (Tokens: %d, Msgs: %d)",
                    payload.current_token_count, len(effective_messages))
        return
    await execute_context_summary_pipeline(ctx, payload)
