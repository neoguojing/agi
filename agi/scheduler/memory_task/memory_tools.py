from typing import List, Any, Optional,Annotated
from pydantic import Field
from langchain_core.tools import tool
from langchain_core.tools import InjectedToolCallId
from langchain_core.messages import ToolMessage
from langgraph.types import Command
from agi.scheduler.memory_task.memory_models import ProfileMemoryRecord, EpisodicMemoryRecord, SemanticMemoryRecord
from agi.scheduler.memory_task.memory_state import MemoryState
import logging


# 初始化日志记录器
logger = logging.getLogger(__name__)

# =====================================================================
# 1. SYSTEM PROMPT VARIABLE (后台专属审计 Prompt)
# =====================================================================
MEMORY_SYSTEM_PROMPT = """## Background Memory Consolidation Tools
You are running as a background Memory Auditor. Your task is to review the user's entire memory state along with recent conversation logs, and perform deduplication, consolidation, and cleanup.

You have access to three dedicated tools to reconcile the memory state. Think in DELTAS (Changes only):
1. `consolidate_profile_memory`: For auditing and merging user preferences and identity traits.
2. `consolidate_episodic_memory`: For auditing historical milestones and chronological logs.
3. `consolidate_semantic_memory`: For auditing stable system facts and knowledge triples.

**CRITICAL RULE:** Do NOT re-save existing, unchanged memories. Only use these tools if you need to ADD new information, UPDATE modified information, or DELETE obsolete/conflicting information.
"""

# =====================================================================
# 2. TOOL DESCRIPTION VARIABLES (强化增删分离)
# =====================================================================
CONSOLIDATE_PROFILE_MEMORY_DESCRIPTION = """Use this tool to reconcile persistent user preferences, workflow habits, or user identity characteristics.

## Parameter Requirements
- reason: A concise explanation of why this consolidation is being performed.
- upserts: List of NEW or UPDATED records {key, value, confidence}. Existing keys will be overwritten.
- deletions: List of STRING keys to COMPLETELY REMOVE (e.g., ['favorite_ide', 'old_habit']). Use this to delete obsolete preferences.
"""

CONSOLIDATE_EPISODIC_MEMORY_DESCRIPTION = """Use this tool to reconcile significant events, project milestones, or historical context.

## Parameter Requirements
- reason: A concise explanation of why this milestone is being updated or removed.
- upserts: List of NEW or UPDATED records {summary, event_time, participants, confidence}.
- deletions: List of STRING keys (format: 'summary_date' or 'summary_anytime') to COMPLETELY REMOVE (e.g., ['initial draft completed_2026-06-01']). Use this to remove redundant event logs.
"""

CONSOLIDATE_SEMANTIC_MEMORY_DESCRIPTION = """Use this tool to reconcile stable facts, configurations, architectures, or knowledge structures.

## Parameter Requirements
- reason: A concise explanation of why this factual knowledge is being consolidated.
- upserts: List of NEW or UPDATED triples {subject, predicate, object, confidence}.
- deletions: List of STRING keys (format: 'subject:predicate:object') to COMPLETELY REMOVE (e.g., ['database:uses:sqlite']). Use this to clear outdated facts.
"""

# =====================================================================
# 3. TOOL IMPLEMENTATIONS (字典转化与 Reducer 适配)
# =====================================================================

logger = logging.getLogger(__name__)

@tool(description=CONSOLIDATE_PROFILE_MEMORY_DESCRIPTION, return_direct=True)
def consolidate_profile_memory(
    upserts: List[ProfileMemoryRecord],
    deletions: Optional[List[str]],
    reason: Optional[str],
    # tool_call_id: Annotated[str, InjectedToolCallId]
) -> MemoryState: # 1. 强类型返回值约束
    try:
        target_dict = {}

        # 2. 直接调用封装好的 dedup_key，无需关心内部拼接逻辑
        for record in (upserts or []):
            if getattr(record, "dedup_key", None):
                target_dict[record.dedup_key] = record

        for delete_key in (deletions or []):
            if delete_key:
                target_dict[delete_key.strip().lower()] = None

        # 3. 严格遵循 MemoryState 结构的更新载荷
        update_payload: MemoryState = {
            "profile_records": target_dict,
            "organization_reason": reason,
            # "messages": [ToolMessage(f"Updated profile records based on: {reason}", tool_call_id=tool_call_id)] if tool_call_id else [],
        }
        return update_payload

    except Exception as e:
        logger.exception(f"Failed to consolidate profile memory. Reason: {reason}. Error: {e}")
        # 发生异常时返回空更新或携带错误信息的 State，防止 Graph 崩溃


@tool(description=CONSOLIDATE_EPISODIC_MEMORY_DESCRIPTION, return_direct=True)
def consolidate_episodic_memory(
    upserts: List[EpisodicMemoryRecord],
    deletions: Optional[List[str]],
    reason: Optional[str],
    # tool_call_id: Annotated[str, InjectedToolCallId]
) -> MemoryState:
    try:
        target_dict = {}

        for record in (upserts or []):
            if getattr(record, "dedup_key", None):
                target_dict[record.dedup_key] = record

        for delete_key in (deletions or []):
            if delete_key:
                target_dict[delete_key.strip().lower()] = None

        update_payload: MemoryState = {
            "episodic_records": target_dict,
            "organization_reason": reason,
            # "messages": [ToolMessage(f"Updated episodic records based on: {reason}", tool_call_id=tool_call_id)] if tool_call_id else [],
        }
        return update_payload

    except Exception as e:
        logger.exception(f"Failed to consolidate episodic memory. Reason: {reason}. Error: {e}")


@tool(description=CONSOLIDATE_SEMANTIC_MEMORY_DESCRIPTION, return_direct=True)
def consolidate_semantic_memory(
    upserts: List[SemanticMemoryRecord],
    deletions: Optional[List[str]],
    reason: Optional[str],
    # tool_call_id: Annotated[str, InjectedToolCallId]
) -> MemoryState:
    try:
        target_dict = {}

        for record in (upserts or []):
            if getattr(record, "dedup_key", None):
                target_dict[record.dedup_key] = record

        for delete_key in (deletions or []):
            if delete_key:
                target_dict[delete_key.strip().lower()] = None

        update_payload: MemoryState = {
            "semantic_records": target_dict,
            "organization_reason": reason,
            # "messages": [ToolMessage(f"Updated semantic records based on: {reason}", tool_call_id=tool_call_id)] if tool_call_id else [],
        }
        return update_payload

    except Exception as e:
        logger.exception(f"Failed to consolidate semantic memory. Reason: {reason}. Error: {e}")
