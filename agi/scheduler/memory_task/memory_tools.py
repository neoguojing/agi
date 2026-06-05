from typing import List, Any
from langchain_core.tools import tool
from langgraph.types import Command
from agi.scheduler.memory_task.memory_models import ProfileMemoryRecord,EpisodicMemoryRecord,SemanticMemoryRecord
import logging
# =====================================================================
# 1. SYSTEM PROMPT VARIABLE
# =====================================================================
MEMORY_SYSTEM_PROMPT = """## Long-Term Memory Management Tools
You have access to three dedicated tools to proactively manage your long-term memory. Treat these tools as a "Save" button. Whenever you encounter a "golden" piece of information during the conversation, call the appropriate tool immediately to persist it and ensure it is not lost in subsequent turns:

1. `save_profile_memory`: For storing persistent user preferences, workflow habits, or user identity characteristics.
2. `save_episodic_memory`: For recording significant events, project milestones, or critical design decisions.
3. `save_semantic_memory`: For storing stable knowledge, configurations, or facts about the project/system in subject-predicate-object triples.
"""

# =====================================================================
# 2. TOOL DESCRIPTION VARIABLES
# =====================================================================
SAVE_PROFILE_MEMORY_DESCRIPTION = """Use this tool to explicitly persist the user's stable preferences, habits, or individual characteristics into long-term memory (e.g., 'User always uses VS Code for Python' or 'User prefers concise code explanations').

## When to Use
- The user shares a persistent personal preference or constraint.
- The user specifies preferred formatting, tools, or interaction styles.
- The user explicitly requests: "Remember this about me".

## Parameter Requirements
- records: A list of objects matching the schema: {key: string, value: string, confidence: float}
- reason: A concise explanation of why this information is being saved.
"""

SAVE_EPISODIC_MEMORY_DESCRIPTION = """Use this tool to record significant events, decisions, or project milestones that occurred during the conversation into long-term memory (e.g., 'The project architecture was finalized as microservices' or 'First round of beta testing was completed').

## When to Use
- A critical, irreversible decision or consensus is reached.
- An event occurs that marks a new phase or milestone for the project.
- You need to log historical context tied to a specific timeline or event.

## Parameter Requirements
- records: A list of objects matching the schema: {summary: string, event_time: string, participants: list[string], confidence: float}
- reason: A concise explanation of why this milestone is being saved.
"""

SAVE_SEMANTIC_MEMORY_DESCRIPTION = """Use this tool to save stable facts, configurations, architectures, or knowledge structures into long-term memory using Subject-Predicate-Object triples (e.g., 'production server' -> 'located_in' -> 'us-east-1' or 'database' -> 'uses' -> 'PostgreSQL').

## When to Use
- You discover a stable, factual piece of information about the system, tech stack, or deployment environment.
- Concrete relationships between entities are established.

## Parameter Requirements
- records: A list of objects matching the schema: {subject: string, predicate: string, object: string, confidence: float}
- reason: A concise explanation of why this factual knowledge is being saved.
"""

# =====================================================================
# 3. TOOL IMPLEMENTATIONS
# =====================================================================

# 初始化日志记录器
logger = logging.getLogger(__name__)

@tool(description=SAVE_PROFILE_MEMORY_DESCRIPTION)
def save_profile_memory(
    records: List[ProfileMemoryRecord],  
    reason: str, 
) -> Command[Any]:
    try:
        return Command(
            update={
                "profile_records": records,
                "organization_reason": reason,
            }
        )
    except Exception as e:
        logger.exception(f"Failed to save profile memory. Reason: {reason}. Error: {e}")

@tool(description=SAVE_EPISODIC_MEMORY_DESCRIPTION)
def save_episodic_memory(
    records: List[EpisodicMemoryRecord],  
    reason: str, 
) -> Command[Any]:
    try:
        return Command(
            update={
                "episodic_records": records,
                "organization_reason": reason,
            }
        )
    except Exception as e:
        logger.exception(f"Failed to save episodic memory. Reason: {reason}. Error: {e}")

@tool(description=SAVE_SEMANTIC_MEMORY_DESCRIPTION)
def save_semantic_memory(
    records: List[SemanticMemoryRecord],  
    reason: str, 
) -> Command[Any]:
    try:
        return Command(
            update={
                "semantic_records": records,
                "organization_reason": reason,
            }
        )
    except Exception as e:
        logger.exception(f"Failed to save semantic memory. Reason: {reason}. Error: {e}")