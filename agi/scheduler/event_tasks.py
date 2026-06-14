from typing import Any, List, cast
from pydantic import BaseModel, Field
import uuid
import logging
from dataclasses import dataclass
from functools import partial
from typing import Mapping, Optional
from datetime import UTC, datetime

from agi.scheduler.memory_task.memory_state import memory_manager
from agi.scheduler.memory_task.runtime import MemoryTaskRuntime, memory_runtime
from agi.scheduler.memory_task.memory_models import SummaryRecord
from agi.scheduler.task_hub import TaskContext, hub

from __future__ import annotations
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    ToolMessage,
    AnyMessage,
    get_buffer_string,
)
from langchain_core.messages.utils import trim_messages, count_tokens_approximately

import logging
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------
# 📐 1. Parameter Contracts & Constants
# ------------------------------------------------------------------------------
class ContextSummarySchema(BaseModel):
    verbose: bool = Field(default=True, description="是否打印微观全量明细表")
    limit: int = Field(default=5000, ge=1, le=10000, description="单次最大扫描的任务实例数")
    current_message_count: int = Field(..., description="当前会话中的总消息条数")
    current_token_count: int = Field(..., description="当前会话的总 Token 数")
    msg_threshold: Optional[int] = Field(None, description="触发摘要的消息条数阈值 (若为 None 则使用默认值 100)")
    token_threshold: Optional[int] = Field(None, description="触发摘要的 Token 数量阈值 (若为 None 则使用默认值 60000)")

DEFAULT_MESSAGES_TO_KEEP = 20
DEFAULT_TRIM_TOKEN_LIMIT = 12000

SUMMARY_PROMPT = """
<role>
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

# ------------------------------------------------------------------------------
# ⛓️ 2. Core Logic Helpers (Extracted and Re-implemented)
# ------------------------------------------------------------------------------

def get_max_input_tokens(model: BaseChatModel) -> Optional[int]:
    """Retrieve max input token limit from the model profile."""
    try:
        profile = model.profile
        if not isinstance(profile, Mapping):
            return None
        limit = profile.get("max_input_tokens")
        return limit if isinstance(limit, int) else None
    except Exception:
        return None

def should_summarize(messages: List[AnyMessage], total_tokens: int,
                      triggers: List[tuple[str, Any]] = None) -> bool:
    """Determine whether summarization should run based on provided triggers."""
    if not triggers:
        return False

    for kind, value in triggers:
        if kind == "messages" and len(messages) >= value:
            return True
        if kind == "tokens" and total_tokens >= value:
            return True
        if kind == "fraction":
            max_tokens = get_max_input_tokens(None) # Note: Actual model context handled in pipeline
            if max_tokens and total_tokens >= int(max_tokens * value):
                return True
    return False

def find_safe_cutoff_index(messages: List[AnyMessage], target_keep_count: int) -> int:
    """Finds a cutoff index that preserves at least `target_keep_count` messages,
    ensuring AI/Tool message pairs are not separated.
    """
    if len(messages) <= target_keep_count:
        return 0

    # Target count from the end of the list
    target_cutoff = len(messages) - target_keep_count

    # Step 1: Find initial cutoff index where ToolMessages are not split
    idx = target_cutoff
    if idx >= len(messages) or not isinstance(messages[idx], ToolMessage):
        return idx

    tool_call_ids = set()
    curr_idx = idx
    while curr_idx < len(messages) and isinstance(messages[curr_idx], ToolMessage):
        msg = messages[curr_idx]
        if msg.tool_call_id:
            tool_call_ids.add(msg.tool_call_id)
        curr_idx += 1

    # Search backward for the original AIMessage containing these tool calls
    for i in range(idx - 1, -1, -1):
        msg = messages[i]
        if isinstance(msg, AIMessage) and msg.tool_calls:
            ai_ids = {tc.get("id") for tc in msg.tool_calls if tc.get("id")}
            if tool_call_ids & ai_ids:
                return i

    return curr_idx

# ------------------------------------------------------------------------------
# 🚀 3. Event-Driven Pipeline
# ------------------------------------------------------------------------------

async def execute_context_summary_pipeline(ctx: TaskContext, payload: ContextSummarySchema):
    """
    💡 Refactored Event Pipeline (Middleware Logic -> Event Mode)
    1. Initialize engine logic with gathered context
    2. Check triggers (Token/Message count)
    3. Calculate safe cutoff preserving AI/Tool pairs
    4. Perform Offloading to persistent history
    5. Generate structured summary via LLM
    6. Commit results to MemoryState
    """
    runtime = cast(MemoryTaskRuntime, ctx.runtime)

    # 1. Setup Context
    current_messages = ctx.messages if hasattr(ctx, "messages") else []
    if not current_messages:
        current_messages = runtime._messages

    token_counter = partial(count_tokens_approximately, use_usage_metadata_scaling=False)
    total_tokens = token_counter(current_messages)

    # 2. Trigger Check (using thresholds provided in payload or fallbacks)
    triggers = []
    if payload.msg_threshold is not None:
        triggers.append(("messages", payload.msg_threshold))
    else:
        triggers.append(("messages", 100))

    if payload.token_threshold is not None:
        triggers.append(("tokens", payload.token_threshold))
    else:
        triggers.append(("tokens", 60000))

    # Validation against actual counts provided in schema
    should_summarize = should_summarize(current_messages, total_tokens, triggers)

    if not should_summarize:
        logger.info("[event_summary] ⏳ Skipping: Thresholds not met (Actual Tokens: %d, Actual Msgs: %d | Thresholds: %s)",
                    total_tokens, len(current_messages), triggers)
        return

    # 3. Safe Cutoff Calculation (Binary Search logic + AI/Tool pair protection)
    cutoff_index = find_safe_cutoff_index(current_messages, DEFAULT_MESSAGES_TO_KEEP)
    if cutoff_index <= 0:
        logger.warning("[event_summary] ⚠️ Invalid cutoff index (%d), skipping summary.", cutoff_index)
        return

    removed_messages = current_messages[:cutoff_index]
    preserved_messages = current_messages[cutoff_index:]

    # 4. Offloading (Simulated backend offload from summarization.py logic)
    thread_id = runtime.thread_id or str(uuid.uuid4())[:8]
    history_path = f"/conversation_history/{thread_id}.md"

    # [Offload Operation]: In a full implementation, this would append `removed_messages`
    # to `history_path`. Logged for visibility here.
    logger.debug("[event_summary] 💾 Offloading %d messages to %s", len(removed_messages), history_path)

    # 5. Summary Generation (with pre-summarization trimming)
    trimmed = trim_messages(
        removed_messages,
        max_tokens=DEFAULT_TRIM_TOKEN_LIMIT,
        token_counter=token_counter,
        strategy="last",
        start_on="human",
        allow_partial=True,
        include_system=True,
    )

    if not trimmed:
        logger.error("[event_summary] ❌ Failed to trim messages for summary.")
        return

    prompt = SUMMARY_PROMPT.format(messages=get_buffer_string(trimmed))
    try:
        response = await runtime.llm.ainvoke(prompt)
        summary_text = response.text.strip()
    except Exception as e:
        logger.error("[event_summary] ❌ LLM Summary Error: %s", e)
        return

    # 6. State Commit
    new_record = SummaryRecord(
        summary=summary_text,
        source_conversation_id=thread_id,
        confidence=0.95 if total_tokens < 50000 else 0.85
    )

    await memory_manager.commit_incremental_memory(
        task_type="summary",
        memory_value=new_record,
        reason=f"Automated event summary triggered. Cutoff: {cutoff_index} messages removed."
    )

    if hasattr(ctx, "messages"):
        # Construct the new context list for the session
        summary_msg = HumanMessage(
            content=f"Conversation Summary:\n\n{summary_text}",
            additional_kwargs={"lc_source": "summarization"},
        )
        ctx.messages = [summary_msg, *preserved_messages]

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
    """Handle context summary event - Triggered by external events."""
    logger.info("[%s] 📩 Received 'event_summary' trigger.", ctx.trace_id)
    await execute_context_summary_pipeline(ctx, payload)
