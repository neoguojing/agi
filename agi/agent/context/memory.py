"""Compatibility facade for context memory abstractions.

Usage:
    Prefer importing focused modules in new code:

    - `memory_models` for typed records and patches.
    - `memory_extraction` for LLM JSON schema, prompts, and parsing.
    - `memory_store` for storage protocols/adapters.
    - `memory_tasks` for task configs, scheduling, and task contracts.

    Existing callers may continue importing from `agi.agent.context.memory` while
    the package migrates to the responsibility-specific modules.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

from agi.agent.context.memory_extraction import (
    MEMORY_EXTRACTION_INSTRUCTIONS,
    MEMORY_EXTRACTION_JSON_SCHEMA,
    build_memory_extraction_prompt,
    parse_memory_extraction_result,
)
from agi.agent.context.memory_models import (
    EpisodicMemoryRecord,
    MemoryEvidence,
    MemoryExtractionResult,
    MemoryOperation,
    MemoryOperationType,
    MemoryPatch,
    MemorySourceKind,
    MemoryTarget,
    ProfileMemoryRecord,
    SemanticEntity,
    SemanticMemoryRecord,
    SemanticObject,
    json_ready,
)
from agi.agent.context.memory_store import (
    DEFAULT_LEGACY_MEMORY_PATHS,
    DEFAULT_MEMORY_TARGET_PATHS,
    BackendMemoryStore,
    MemoryStore,
)
from agi.agent.context.memory_tasks import (
    MemoryMaintenanceConfig,
    MemoryTask,
    MemoryTaskConfig,
    MemoryTaskContext,
    MemoryTaskResult,
    MemoryTaskScheduleState,
    MemoryTaskScheduler,
)

logger = logging.getLogger(__name__)

class BaseMemoryExtractionTask(MemoryTask):
    """Base class for tasks that use an LLM to extract memories.
    
    Implements the core logic of:
    1. Preparing the prompt with conversation and existing memory.
    2. Calling the LLM for structured output.
    3. Parsing the output into patches.
    """
    
    def __init__(self, name: str, target: MemoryTarget, config: MemoryTaskConfig):
        self.name = name
        self.target = target
        self.config = config

    def should_run(self, context: MemoryTaskContext) -> bool:
        """Determines if the task should run based on config and message presence."""
        # Basic check: run if enabled and messages are present
        return self.config.enabled and len(context.messages) > 0

    async def _call_llm_for_extraction(self, context: MemoryTaskContext, prompt: str) -> Any:
        """Helper to interact with the LLM using structured output."""
        if context.llm:
            # Use the explicit LLM provided in the context
            llm_with_struct = context.llm.with_structured_output(
                schema=MEMORY_EXTRACTION_JSON_SCHEMA
            )
            
            struct_result = await llm_with_struct.invoke(prompt)
            return struct_result

        logger.error(f"Task {self.name} failed: No LLM provided in MemoryTaskContext.")
        return None

    async def run(self, context: MemoryTaskContext) -> MemoryTaskResult:
        """Executes the memory extraction process for the target memory type."""
        # 1. Prepare conversation history as string
        conversation_text = "\n".join([str(m.content) for m in context.messages])
        
        # 2. Load existing memory for the target to provide context to the LLM
        path = DEFAULT_MEMORY_TARGET_PATHS.get(self.target, "")
        existing_mem_text = context.store.read_text(path)
        
        # 3. Build prompt
        prompt = build_memory_extraction_prompt(
            conversation=conversation_text,
            existing_memory=existing_mem_text
        )
        
        # 4. Get LLM result
        llm_payload = await self._call_llm_for_extraction(context, prompt)
        if not llm_payload:
            return MemoryTaskResult(task_name=self.name, changed=False, summary="LLM call failed")

        # 5. Parse and convert to patches
        extraction = parse_memory_extraction_result(llm_payload)
        patches = extraction.to_patches(reason=f"Automatic {self.target} memory extraction")
        
        # Filter patches to only include the target this task is responsible for
        target_patches = tuple(p for p in patches if p.target == self.target)
        
        return MemoryTaskResult(
            task_name=self.name,
            changed=bool(target_patches),
            summary=f"Extracted {len(target_patches)} patches for {self.target} memory.",
            patches=target_patches
        )

class ProfileMemoryTask(BaseMemoryExtractionTask):
    """Task dedicated to extracting stable user profile and preferences."""
    def __init__(self, config: MemoryTaskConfig):
        super().__init__(
            name="profile_extraction_task",
            target="profile",
            config=config
        )

class EpisodicMemoryTask(BaseMemoryExtractionTask):
    """Task dedicated to extracting time-bound events and experiences."""
    def __init__(self, config: MemoryTaskConfig):
        super().__init__(
            name="episodic_extraction_task",
            target="episodic",
            config=config
        )

class SemanticMemoryTask(BaseMemoryExtractionTask):
    """Task dedicated to extracting graph-ready long-term knowledge."""
    def __init__(self, config: MemoryTaskConfig):
        super().__init__(
            name="semantic_extraction_task",
            target="semantic",
            config=config
        )

async def run_memory_maintenance(
    llm: Any,
    backend: Any,
    messages: Sequence[Any],
    schedule_state_dict: dict[str, str] | None = None,
    tasks: Sequence[MemoryTask] | None = None,
    config: MemoryMaintenanceConfig | None = None,
    apply_patches: bool = True,
) -> tuple[list[MemoryTaskResult], dict[str, str]]:
    """
    High-level entry point to run due memory maintenance tasks.
    
    Encapsulates the creation of MemoryStore, MemoryTaskContext, and 
    MemoryTaskScheduleState to avoid exposing internal task-system classes to the caller.
    
    Args:
        llm: The LLM model to use for extraction.
        backend: The storage backend protocol.
        messages: The conversation history to analyze.
        schedule_state_dict: A dictionary of task names to ISO timestamps.
        tasks: Optional list of custom tasks to run.
        config: Optional configuration for task intervals and confidence.
        apply_patches: Whether to automatically write changes to the store.
    """
    m_config = config or MemoryMaintenanceConfig()

    if tasks is None:
        tasks = [
            ProfileMemoryTask(m_config.profile),
            EpisodicMemoryTask(m_config.episodic),
            SemanticMemoryTask(m_config.semantic),
        ]
    
    # Internalize the store creation
    store = BackendMemoryStore(backend)
    
    # Internalize the context creation
    context = MemoryTaskContext(
        store=store,
        backend=backend,
        llm=llm,
        messages=list(messages),
    )
    
    # Internalize the state management
    state = None
    if schedule_state_dict is not None:
        state = MemoryTaskScheduleState.from_iso_dict(schedule_state_dict)
    
    scheduler = MemoryTaskScheduler()
    results, new_state = await scheduler.run_due_tasks(
        tasks=tasks,
        context=context,
        state=state,
        apply_patches=apply_patches
    )
    
    return results, new_state.to_iso_dict()

def read_memory(
    backend: Any,
    target: MemoryTarget,
    as_jsonl: bool = True,
) -> list[dict[str, Any]] | str:
    """
    High-level entry point to read memory for a specific target.
    
    Encapsulates MemoryStore creation.
    
    Args:
        backend: The backend protocol implementation.
        target: The memory target to read ('profile', 'episodic', or 'semantic').
        as_jsonl: If True, returns a list of records. If False, returns raw text.
    """
    store = BackendMemoryStore(backend)
    path = DEFAULT_MEMORY_TARGET_PATHS.get(target, "")
    
    if as_jsonl:
        return store.read_jsonl(path)
    return store.read_text(path)

def format_memory_for_llm(
    backend: Any,
    target: MemoryTarget,
) -> str:
    """
    Reads memory for a target and formats it as a human-readable string 
    suitable for LLM context injection.
    
    Converts structured JSONL records into a clean text format that the 
    LLM can easily parse as context.
    """
    records = read_memory(backend, target, as_jsonl=True)
    if not records:
        return f"No {target} memory available."

    lines = [f"--- {target.upper()} MEMORY ---"]
    
    for rec in records:
        if target == "profile":
            key = rec.get("key", "unknown")
            val = rec.get("value", "unknown")
            lines.append(f"- {key}: {val}")
        elif target == "episodic":
            summary = rec.get("summary", "No summary")
            time = rec.get("event_time", "Unknown time")
            lines.append(f"- [{time}] {summary}")
        elif target == "semantic":
            subj = rec.get("subject", {}).get("label") or rec.get("subject", {}).get("id", "Unknown")
            pred = rec.get("predicate", "is")
            obj = rec.get("object", {}).get("value") or rec.get("object", {}).get("label") or "Unknown"
            lines.append(f"- {subj} {pred} {obj}")
    
    return "\n".join(lines)

__all__ = [
    "MemoryTarget",
    "MemoryOperationType",
    "MemorySourceKind",
    "MemoryOperation",
    "MemoryPatch",
    "MemoryEvidence",
    "ProfileMemoryRecord",
    "EpisodicMemoryRecord",
    "SemanticEntity",
    "SemanticObject",
    "SemanticMemoryRecord",
    "MemoryExtractionResult",
    "json_ready",
    "MEMORY_EXTRACTION_JSON_SCHEMA",
    "MEMORY_EXTRACTION_INSTRUCTIONS",
    "build_memory_extraction_prompt",
    "parse_memory_extraction_result",
    "DEFAULT_LEGACY_MEMORY_PATHS",
    "DEFAULT_MEMORY_TARGET_PATHS",
    "MemoryStore",
    "BackendMemoryStore",
    "MemoryTaskConfig",
    "MemoryMaintenanceConfig",
    "MemoryTaskContext",
    "MemoryTaskResult",
    "MemoryTaskScheduleState",
    "MemoryTask",
    "MemoryTaskScheduler",
    "ProfileMemoryTask",
    "EpisodicMemoryTask",
    "SemanticMemoryTask",
    "run_memory_maintenance",
    "read_memory",
    "format_memory_for_llm",
]
