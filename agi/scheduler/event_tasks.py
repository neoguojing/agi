from typing import Any, List, cast, Mapping, Optional, Literal, Callable, Iterable
from pydantic import BaseModel, Field
import uuid
import logging
import warnings
from dataclasses import dataclass
from functools import partial
from datetime import UTC, datetime
from collections.abc import Iterable as IterableABC

from agi.scheduler.memory_task.memory_state import memory_manager
from agi.scheduler.memory_task.runtime import MemoryTaskRuntime, memory_runtime
from agi.scheduler.memory_task.memory_models import SummaryRecord
from agi.scheduler.task_hub import TaskContext, hub

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    ToolMessage,
    AnyMessage,
    get_buffer_string,
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

    逻辑：
    1. 如果当前索引处是 ToolMessage，则向后扫描所有连续的 ToolMessages。
    2. 记录这些 ToolMessages 涉及的所有 tool_call_id。
    3. 向前回溯搜索，直到找到包含这些 ID 的 AIMessage。
    4. 将截断点移动到该 AIMessage 之前，确保调用链完整。
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

    策略：
    - 基于消息数：直接计算 (总数 - 保留数)。
    - 基于 Token 数：
        采用 二分查找 (Binary Search) 算法。在消息列表中寻找最小索引 i，
        使得 messages[i:] 的 Token 总数 <= 目标保留 Token 数。
        时间复杂度 O(log N * T)，其中 T 为 Token 计数开销。
    - 最后调用 _find_safe_cutoff_point 确保不破坏 Tool 调用链。
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
    """
    对单个工具调用进行参数裁剪。
    针对写文件等可能产生海量文本的工具，将参数截断至 2000 字符，防止摘要 LLM 崩溃。
    """
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
    """
    【特性：工具参数预截断】
    扫描消息列表，对处于保留窗口之外的旧消息中的特定工具调用进行参数裁剪。
    仅针对 'write_file' 和 'edit_file' 工具，因为这些工具最容易产生上下文爆炸。
    """
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
    """
    【特性：历史离线存储】
    在消息被摘要替换前，将其持久化到后端文件。
    采用追加模式，每个摘要事件产生一个带时间戳的章节，确保历史可追溯且不丢失。
    """
    path = f"/conversation_history/{thread_id}.md"
    # 过滤掉之前的摘要消息，防止存储递归摘要
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

def _build_summary_message(summary: str, file_path: Optional[str]) -> List[AnyMessage]:
    """
    构建最终插入会话的摘要消息。
    如果离线存储成功，会在消息中加入文件路径引用，允许 Agent 在需要时通过工具读取历史细节。
    """
    if file_path is not None:
        content = f"You are in the middle of a conversation that has been summarized.\n\nThe full conversation history has been saved to {file_path} should you need to refer back to it for details.\n\nA condensed summary follows:\n\n<summary>\n{summary}\n</summary>"
    else:
        content = f"Here is a summary of the conversation to date:\n\n{summary}"
    return [HumanMessage(content=content, additional_kwargs={"lc_source": "summarization"})]

# ------------------------------------------------------------------------------
# 🚀 3. Event-Driven Pipeline
# ------------------------------------------------------------------------------

async def execute_context_summary_pipeline(ctx: TaskContext, payload: ContextSummarySchema):
    """
    【核心执行流水线】
    实现从有效上下文提取 $\rightarrow$ 摘要生成 $\rightarrow$ 状态同步的完整链路。

    详细步骤：
    1. 获取有效上下文：[上一次摘要消息 + 之后的所有增量消息]。
    2. 参数截断：对旧消息中的巨量工具参数进行裁剪，降低摘要 LLM 负载。
    3. 计算截断索引：
       - 计算相对索引 (relative_cutoff)：决定当前有效列表中哪些部分需要被再次压缩。
       - 映射绝对索引 (absolute_cutoff)：将相对位置转换回原始全量消息流的索引，用于后端存储。
    4. 离线存储：将 [0, absolute_cutoff] 的原始消息流持久化到 Markdown 文件。
    5. 生成摘要：将有效列表中的 [0, relative_cutoff] 部分发送给 LLM。
    6. 状态同步：
       - 更新 SummaryRecord (包含摘要文本和绝对截断点)。
       - 更新 session 消息列表为 [新摘要消息, 保留的增量消息]。
    """
    runtime = cast(MemoryTaskRuntime, ctx.runtime)
    token_counter = _get_approximate_token_counter(runtime.llm)

    # 1. 获取当前有效消息 (Reflects the actual window the LLM sees)
    effective_messages = await memory_manager.get_effective_summary_context()
    if not effective_messages:
        logger.info("[event_summary] ⏭️ No messages to summarize.")
        return

    # 2. 工具参数截断 (Feature from summarization.py)
    effective_messages, _ = _truncate_args(effective_messages)

    # 3. 计算截断位置
    keep_policy = ("messages", _DEFAULT_MESSAGES_TO_KEEP)
    relative_cutoff = _determine_cutoff_index(effective_messages, token_counter, keep_policy)

    if relative_cutoff <= 0:
        logger.info("[event_summary] ⏭️ Cutoff index 0, nothing to compress.")
        return

    # 将相对索引映射回原始消息流的绝对索引
    thread_id = runtime.thread_id or "unknown"
    prev_summary = memory_manager._state.get("summary_records", {}).get(thread_id)

    if prev_summary:
        prev_cutoff = getattr(prev_summary, "cutoff_index", 0)
        # relative_cutoff - 1 是因为 effective_messages[0] 是之前的摘要消息
        absolute_cutoff = prev_cutoff + max(0, relative_cutoff - 1)
    else:
        absolute_cutoff = relative_cutoff

    # 4. 离线存储原始消息
    full_messages = await memory_manager.refresh_messages()
    messages_to_summarize = full_messages[:absolute_cutoff]
    preserved_messages = full_messages[absolute_cutoff:]

    backend = runtime.backend
    file_path = await _offload_to_backend(backend, messages_to_summarize, thread_id)

    # 5. 生成摘要 (Summarize the "effective" part that was cut off)
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

    prompt = DEFAULT_SUMMARY_PROMPT.format(messages=get_buffer_string(trimmed))
    try:
        response = await runtime.llm.ainvoke(prompt)
        summary_text = response.text.strip()
    except Exception as e:
        logger.error("[event_summary] ❌ LLM Summary Error: %s", e)
        return

    # 6. 状态提交与同步
    new_record = SummaryRecord(
        summary=summary_text,
        source_conversation_id=thread_id,
        confidence=0.9 if payload.current_token_count < 100000 else 0.7,
        cutoff_index=absolute_cutoff
    )

    await memory_manager.commit_incremental_memory(
        task_type="summary",
        memory_value=new_record,
        reason=f"Event summary triggered. Absolute cutoff: {absolute_cutoff} messages."
    )

    if hasattr(ctx, "messages"):
        summary_msgs = _build_summary_message(summary_text, file_path)
        ctx.messages = [*summary_msgs, *preserved_messages]

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
    """
    处理上下文摘要事件的入口。

    逻辑分流：
    1. 准备有效上下文 $\rightarrow$ 2. 阈值决策 $\rightarrow$ 3. 执行压缩流水线。
    """
    logger.info("[%s] 📩 Received 'event_summary' trigger.", ctx.trace_id)

    runtime = cast(MemoryTaskRuntime, ctx.runtime)

    # 决策步骤：通过有效上下文和 Payload 统计值判定是否需要压缩
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

    # 仅在满足触发条件时启动昂贵的执行流水线
    if not _should_summarize(effective_messages, payload.current_token_count, triggers, runtime.llm):
        logger.info("[event_summary] ⏳ Skipping: Thresholds not met (Tokens: %d, Msgs: %d)",
                    payload.current_token_count, len(effective_messages))
        return

    await execute_context_summary_pipeline(ctx, payload)
