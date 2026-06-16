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

from agi.config import (
    CONTEXT_MESSAGES_TO_KEEP,
    CONTEXT_TRIM_TOKEN_LIMIT,
    CONTEXT_FALLBACK_MESSAGE_COUNT,
)

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

class ContextSummarySchema(BaseModel):
    current_message_count: int = Field(..., description="当前会话中的总消息条数")
    current_token_count: int = Field(..., description="当前会话的总 Token 数")
    msg_threshold: Optional[int] = Field(30, description="触发摘要的消息条数阈值")
    token_threshold: Optional[int] = Field(20000, description="触发摘要的 Token 数量阈值")

# ------------------------------------------------------------------------------
# ⚙️ 2. Core Logic Helpers
# # ------------------------------------------------------------------------------
# 工具折算：将工具 Schema 转为 JSON 字符串，按 字符数/4（默认比例）向上取整累加 Token。

# 消息遍历：

# 文本与属性：统计纯文本正文、角色(Role)、名称(Name)及工具调用字符串的字符总数。

# 多模态图片：跳过 Base64 字符统计，直接按固定值（默认 85 Token）累加。

# 单条结算：将单条消息的总字符数按 字符数/4 向上取整，并附加固定基础开销（默认 3 Token）。

# 动态校准（可选）：读取历史 AI 消息真实的 API 消耗数据，计算比例 (真实值/估算值)，将总 Token 乘以该比例（限制在 1.0~1.25）进行修正。

# 最终结算：总数再次向上取整并返回。
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
        if kind == "messages" and len(messages) > value:
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
            return _find_safe_cutoff_point(messages, max(0, len(messages) - CONTEXT_MESSAGES_TO_KEEP))

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

    return _find_safe_cutoff_point(messages, max(0, len(messages) - CONTEXT_MESSAGES_TO_KEEP))

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

    keep_policy = ("messages", CONTEXT_MESSAGES_TO_KEEP)
    relative_cutoff = _determine_cutoff_index(effective_messages, token_counter, keep_policy)

    if relative_cutoff <= 0:
        logger.info("[%s] [event_summary] ⏭️ Cutoff index 0, nothing to compress.", ctx.trace_id)
        return

    logger.info("[%s] [event_summary] ✂️ Relative cutoff: %d", ctx.trace_id, relative_cutoff)

    thread_id = runtime.thread_id or "unknown"
    prev_summary = await memory_manager.get_current_summary()
    prev_cutoff = getattr(prev_summary, "cutoff_index", 0)

    if prev_summary:
        absolute_cutoff = prev_cutoff + max(0, relative_cutoff - 1)
    else:
        absolute_cutoff = relative_cutoff

    logger.info("[%s] [event_summary] 🎯 Absolute cutoff: %d (prev_cutoff: %d)",
                ctx.trace_id, absolute_cutoff, getattr(prev_summary, "cutoff_index", 0) if prev_summary else 0)

    full_messages = await memory_manager.refresh_messages()
    full_messages = convert_to_messages(full_messages)

    messages_to_summarize = full_messages[prev_cutoff:absolute_cutoff]
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
    # 精准计量 (max_tokens / token_counter)：支持调用模型自带的分词器精确计算 Token，或按对话条数计算。

    # 方向选择 (strategy)：通常用于保留“最新”（last）的上下文，自动丢弃最旧的记忆。

    # 合法性兜底 (start_on / end_on)：强制截断后的对话列必定以人类消息（HumanMessage）开头，避免因截断导致消息顺序错乱从而触发大模型 API 报错。

    # 人设保护 (include_system)：锁定并保留对话最开头的系统提示词（SystemMessage），防止大模型在长对话后“失忆”或忘记初始指令。

    # 精细切割 (allow_partial)：当遇到单条超长消息时，支持将其从中间切断以填满 Token 额度，而不是粗暴地整条丢弃。
    trimmed = trim_messages(
        summarize_input,
        max_tokens=CONTEXT_TRIM_TOKEN_LIMIT,
        token_counter=token_counter,
        strategy="last",
        start_on="human",
        allow_partial=True,
        include_system=True,
    )

    if not trimmed:
        trimmed = summarize_input[-CONTEXT_FALLBACK_MESSAGE_COUNT:]

    logger.info("[%s] [event_summary] 📝 Sending %d messages to LLM for summary (approx %d tokens)",
                ctx.trace_id, len(trimmed), token_counter(trimmed))

    prompt = DEFAULT_SUMMARY_PROMPT.format(messages=get_buffer_string(trimmed))
    try:
        response = await runtime.llm.ainvoke(prompt)
        summary_text = ""
        if isinstance(response.content, list):
            # 完美兼容常规字符串列表或 OpenAI/Anthropic 风格的文本字典列表
            summary_text = "".join(
                item if isinstance(item, str) else item.get("text", "") 
                for item in response.content
            ).strip()
        else:
            summary_text = str(response.content).strip()
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
        triggers.append(("messages", 30))
    if payload.token_threshold is not None:
        triggers.append(("tokens", payload.token_threshold))
    else:
        triggers.append(("tokens", 20000))
    if not _should_summarize(effective_messages, payload.current_token_count, triggers, runtime.llm):
        logger.info("[event_summary] ⏳ Skipping: Thresholds not met (Tokens: %d, Msgs: %d)",
                    payload.current_token_count, len(effective_messages))
        return
    await execute_context_summary_pipeline(ctx, payload)
