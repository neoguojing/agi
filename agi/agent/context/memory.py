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
    """Base class for tasks that use an LLM to extract memories."""
    
    def __init__(self, name: str, target: MemoryTarget, config: MemoryTaskConfig):
        self.name = name
        self.target = target
        self.config = config

    def should_run(self, context: MemoryTaskContext) -> bool:
        # Basic check: run if enabled and messages are present
        return self.config.enabled and len(context.messages) > 0

    async def _call_llm_for_extraction(self, context: MemoryTaskContext, prompt: str) -> Any:
        """
        Helper to interact with the LLM. 
        """
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
    def __init__(self):
        super().__init__(
            name="profile_extraction_task",
            target="profile",
            config=MemoryTaskConfig(interval_seconds=24 * 3600, min_confidence=0.75)
        )

class EpisodicMemoryTask(BaseMemoryExtractionTask):
    """Task dedicated to extracting time-bound events and experiences."""
    def __init__(self):
        super().__init__(
            name="episodic_extraction_task",
            target="episodic",
            config=MemoryTaskConfig(interval_seconds=3600, min_confidence=0.45)
        )

class SemanticMemoryTask(BaseMemoryExtractionTask):
    """Task dedicated to extracting graph-ready long-term knowledge."""
    def __init__(self):
        super().__init__(
            name="semantic_extraction_task",
            target="semantic",
            config=MemoryTaskConfig(interval_seconds=24 * 3600, min_confidence=0.65)
        )

async def run_memory_maintenance(
    context: MemoryTaskContext,
    state: MemoryTaskScheduleState | None = None,
    tasks: Sequence[MemoryTask] | None = None,
    apply_patches: bool = True,
) -> tuple[list[MemoryTaskResult], MemoryTaskScheduleState]:
    """
    High-level entry point to run due memory maintenance tasks.
    
    If no tasks are provided, it defaults to the standard set of extraction tasks.
    """
    if tasks is None:
        tasks = [
            ProfileMemoryTask(),
            EpisodicMemoryTask(),
            SemanticMemoryTask(),
        ]
    
    scheduler = MemoryTaskScheduler()
    return await scheduler.run_due_tasks(
        tasks=tasks,
        context=context,
        state=state,
        apply_patches=apply_patches
    )

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
]
