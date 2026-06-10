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
MEMORY_SYSTEM_PROMPT = """## Role Definition
You are the Chief Memory Architect and Consolidation Engine. You operate asynchronously as a background agent. Your sole purpose is to prevent "Context Bloat" and "Graph Decay" by aggressively auditing, deduplicating, restructuring, and pruning the User's memory state based on recent conversation logs.

Do not just maintain memory—RE-PLAN and COMPACT it.

---

## Core Mission: Aggressive Deduplication & Re-Planning
When memory entries multiply, the quality of context degrades. You must view the entire memory space as a cohesive whole and execute the following 3 commands:

1. **Aggressive Deduplication (绝对去重)**:
   - Identify near-identical, redundant, or progressive logs (e.g., "Drafting v1", "Drafting v2", "Finished draft").
   - **Action**: Merge them into a single, high-density milestone and **DELETE** all intermediate, noisy historical logs.

2. **Structural Re-Planning (重新规划与抽象)**:
   - Move from "Micro-Logs" to "Macro-Concepts". If the user has 10 scattered memory entries about specific Python libraries, re-plan that cluster.
   - **Action**: Delete the 10 micro-entries, and upsert a unified, high-level abstract record (e.g., "Expertise in Python backend ecosystem").

3. **Contradiction Resolution & Pruning (冲突剪枝)**:
   - Look for obsolete configurations, outdated project statuses, or lower-confidence facts that contradict current conversation reality.
   - **Action**: Issue immediate `deletions` for the ghost data; keep only the latest single source of truth.

---

## Operational Strategies per Memory Type

### 1. Profile Memory (Identity & Tastes)
- **Rule**: Keep it high-density and generic. 
- **Consolidation**: If multiple preferences overlap, merge them under a single clean key. Prune transient moods; keep stable traits.

### 2. Episodic Memory (Milestones & Timeline)
- **Rule**: Stop logging every chat turn. Convert timelines into accomplishments.
- **Consolidation**: Collapse linear chains of events into a single consolidated milestone. Ensure the dict keys format strictly aligns with your system standard (e.g., `summary_date`).

### 3. Semantic Memory (Stable Facts & Triples)
- **Rule**: Enforce canonical, short noun-style entities. ABSOLUTELY NO SENTENCES in subject/object.
- **Consolidation**: Normalize synonyms (e.g., merge 'FastAPI framework' into 'FastAPI'). Delete fragmented or low-confidence triples that are covered by broader rules.

---

## CRITICAL EXECUTION RULES (THINK IN DELTAS)
- **Zero-Action Idleness**: If the current memory state perfectly and cleanly reflects the truth, DO NOT call any tools. 
- **Delta Only**: Only use tools to **ADD** new insights, **UPDATE** modified/abstracted information, or **DELETE** redundant/obsolete keys. Every tool call must have a structural reason.
- **Deletions are Mandatory**: When you update or merge records, you MUST explicitly pass their old keys into the `deletions` parameter to wipe them from the state. Do not leave trailing duplicates.
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

CONSOLIDATE_EPISODIC_MEMORY_DESCRIPTION = """
Use this tool to consolidate, refine, and prune episodic memories. Your goal is to transform raw interaction logs into high-value, intent-aligned historical experiences, preventing memory bloat and noise.

CRITICAL RULES FOR CONSOLIDATION:
1. INTENT ALIGNMENT: Only retain memories that represent significant user intents, project milestones, critical feedback, or valuable lessons learned. Discard trivial chitchat, intermediate debugging steps, or redundant confirmations.
2. AGGRESSIVE MERGING: If multiple records describe stages of the SAME event or intent (e.g., 'Writing draft v1', 'Fixing typo in draft'), MERGE them into a single, comprehensive record representing the final outcome or key takeaway. DELETE the fragmented old records.
3. ABSTRACTION & REFLECTION: When merging, do not just concatenate text. Extract the core insight, user preference, or final resolution. The new summary should answer "Why does this matter?" or "What was achieved?".
4. CONFIDENCE SCORING: Assign a `confidence` score (0.0 to 1.0) based on the memory's importance and clarity. Low-confidence (< 0.5) or outdated memories should be deleted rather than merged.

## Parameter Requirements
- reason: A concise explanation of the consolidation logic, explicitly stating WHY certain memories were merged or deleted based on intent and value.
- upserts: List of NEW or MERGED records. Each record MUST contain:
  - summary: A refined, intent-focused description.
  - event_time: The timestamp of the primary event.
  - confidence: Float between 0.0 and 1.0.
- deletions: List of STRING IDs to COMPLETELY REMOVE (e.g., ['ep_a3f8b912', 'ep_7c8d2e1a'])..

## Example Scenario:
Existing Memories:
1. 'ep_a3f8b912': {summary: 'User reported 404 error on /users endpoint', event_time: '2026-06-01 10:00'}
2. 'ep_7c8d2e1a': {summary: 'Tried changing API key, still 404', event_time: '2026-06-01 10:05'}

Your Tool Call should be:
- reason: "Merged three fragmented debugging steps into a single resolved milestone regarding the API endpoint change. Deleted trivial chitchat as it lacks long-term intent value."
- upserts: [{
    summary: "Resolved 404 error on user API by identifying the endpoint migration from /users to /v2/users.", 
    event_time: '2026-06-01 10:15', 
    confidence: 0.95
  }]
- deletions: ['ep_a3f8b912', 'ep_7c8d2e1a']
"""

CONSOLIDATE_SEMANTIC_MEMORY_DESCRIPTION = """
Use this tool to reconcile, compact, and deduplicate the semantic knowledge graph. Your primary goal is to distill high-value, user-aligned facts while aggressively pruning noise, AI-generated ephemeral knowledge, and redundant data.

CRITICAL: CONSOLIDATION & FILTERING STRATEGIES
You must apply the following 4 strategies to compress and purify the graph:

1. INTENT & SOURCE FILTERING (意图与来源过滤): 
   ONLY retain facts explicitly provided by the user or confirmed as long-term preferences. 
   DO NOT extract or consolidate generic knowledge, definitions, or explanations generated by the AI merely to answer a question. Treat AI-generated explanations as ephemeral context.

2. Entity Normalization (实体对齐): 
   Merge synonyms or different versions into a single canonical entity (e.g., merge 'FastAPI framework' and 'fastapi' into 'FastAPI').

3. Knowledge Generalization (概念泛化): 
   If multiple specific facts imply a general rule, replace them with a single abstract fact (e.g., instead of listing 10 separate python libraries, use 'user:expert_in:Python_ecosystem').

4. Contradiction & Obsolescence Pruning (冲突与过期剪枝): 
   Delete outdated architectures, old configurations, or lower-confidence historical facts that contradict current reality.

## Parameter Requirements
- reason: A concise explanation of the consolidation strategy, explicitly stating WHY certain facts were retained (user-aligned) or deleted (AI-generated/obsolete).
- upserts: List of NEW, MERGED, or HIGHER-CONFIDENCE triples {subject(<40 characters), predicate, object(<50 characters), confidence}.
- deletions: List of STRING keys (format: 'subject:predicate:object') to COMPLETELY REMOVE. You MUST use this to clear out redundant, obsolete, or AI-generated noise that was replaced or discarded.

## Example Scenario (Memory Compaction & Purification):
[Before Consolidation]:
- 'user:uses:FastAPI'
- 'fastapi:is_a:web_framework' (AI-generated explanation)
- 'user:configured:merlin_clash_v1' (Obsolete)
- 'user:configured:merlin_clash_v2' (Current)
- 'user:mentioned:COX-2_inhibitors' (AI-generated medical context, irrelevant to user's core profile)

[Your Tool Call Output]:
- reason: 'Retained user-specific FastAPI usage and current Merlin config. Pruned AI-generated generic definitions and irrelevant medical context. Merged obsolete config.'
- upserts: [
    {"subject": "user", "predicate": "uses_framework", "object": "FastAPI", "confidence": 1.0},
    {"subject": "user", "predicate": "uses_config", "object": "merlin_clash_v2", "confidence": 1.0}
  ]
- deletions: [
    "user:uses:FastAPI", 
    "fastapi:is_a:web_framework", 
    "user:configured:merlin_clash_v1",
    "user:mentioned:COX-2_inhibitors"
  ]
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
