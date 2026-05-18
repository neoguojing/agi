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

import asyncio
import logging
from typing import Any, Sequence

from agi.agent.context.memory_extraction import (
    MEMORY_EXTRACTION_INSTRUCTIONS,
    build_memory_extraction_prompt,
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
    ProfileMemoryList,
    EpisodicMemoryList,
    SemanticMemoryList,
)
from agi.agent.context.memory_store import (
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
    MessageProvider,
)

logger = logging.getLogger(__name__)


TARGET_SCHEMA_MAP = {
    "profile": ProfileMemoryList,
    "episodic": EpisodicMemoryList,
    "semantic": SemanticMemoryList,
}

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
        return self.config.enabled and len(context.get_messages()) > 0

    async def _call_llm_for_extraction(self, context: MemoryTaskContext, prompt: str) -> Any:
        """Helper to interact with the LLM using structured output."""
        if context.llm:
            try:

                # Use the explicit LLM provided in the context
                schema = TARGET_SCHEMA_MAP.get(self.target)
                llm_with_struct = context.llm.with_structured_output(schema)
                struct_result = await llm_with_struct.ainvoke(prompt)
                return struct_result

            except Exception as e:
                logger.error(e)

        logger.error(f"Task {self.name} failed: No LLM provided in MemoryTaskContext.")
        return None

    async def run(self, context: MemoryTaskContext) -> MemoryTaskResult:
        """Executes the memory extraction process for the target memory type."""
        logger.info(f"Executing memory task: {self.name} for target: {self.target}")
        # 1. Prepare conversation history as string
        conversation_text = "\n".join([str(m.content) for m in context.get_messages()])

        # 2. Load existing memory for the target to provide context to the LLM
        path = DEFAULT_MEMORY_TARGET_PATHS.get(self.target, "")
        existing_mem_text = context.store.read_text(path)

        # 3. Build prompt
        prompt = build_memory_extraction_prompt(
            conversation=conversation_text,
            existing_memory=existing_mem_text,
        )
        
        # 4. Get LLM result
        llm_payload = await self._call_llm_for_extraction(context, prompt)

        if not llm_payload:
            logger.error(f"Task {self.name} failed: LLM returned no result")
            return MemoryTaskResult(task_name=self.name, changed=False, summary="LLM call failed")

        # 5. Parse and convert to patches
        print(f"***************{llm_payload}")
        
        filtered_extraction = MemoryExtractionResult()
        if isinstance(llm_payload,ProfileMemoryList):
            filtered_extraction.profile_memories = [m for m in llm_payload.profile_memories if m.confidence >= self.config.min_confidence]
        if isinstance(llm_payload,EpisodicMemoryList):
            filtered_extraction.episodic_memories = [m for m in llm_payload.episodic_memories if m.confidence >= self.config.min_confidence]
        if isinstance(llm_payload,SemanticMemoryList):
            filtered_extraction.semantic_memories = [m for m in llm_payload.semantic_memories if m.confidence >= self.config.min_confidence]
        
        # Create a new extraction result with filtered records to generate patches
        
        
        patches = filtered_extraction.to_patches(reason=f"Automatic {self.target} memory extraction")
        
        # Filter patches to only include the target this task is responsible for
        target_patches = tuple(p for p in patches if p.target == self.target)

        logger.info(f"Task {self.name} completed: extracted {len(target_patches)} patches for {self.target}")

        return MemoryTaskResult(
            task_name=self.name,
            changed=bool(target_patches),
            summary=f"Extracted {len(target_patches)} patches for {self.target} memory (confidence >= {self.config.min_confidence}).",
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
    messages: Sequence[Any] | MessageProvider,
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
        messages: The conversation history to analyze, or a provider to retrieve it.
        schedule_state_dict: A dictionary of task names to ISO timestamps.
        tasks: Optional list of custom tasks to run.
        config: Optional configuration for task intervals and confidence.
        apply_patches: Whether to automatically write changes to the store.
    """
    try:
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
        if isinstance(messages, MessageProvider):
            context = MemoryTaskContext(
                store=store,
                backend=backend,
                llm=llm,
                message_provider=messages,
            )
        else:
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
    except Exception as e:
        logger.exception("Error during memory maintenance execution")
        return [], schedule_state_dict or {}

class MemoryMaintenanceManager:
    """Manager for the background memory maintenance loop.

    Handles the lifecycle of the background task that periodically
    executes due memory maintenance tasks.
    """
    def __init__(
        self,
        llm: Any,
        backend: Any,
        messages: Sequence[Any] | MessageProvider,
        config: MemoryMaintenanceConfig | None = None,
        initial_state: dict[str, str] | None = None,
        tasks: Sequence[MemoryTask] | None = None,
        apply_patches: bool = True,
        tick_interval: int = 60
    ):
        self.llm = llm
        self.backend = backend
        self.messages = messages
        self.config = config
        self.initial_state = initial_state
        self.tasks = tasks
        self.apply_patches = apply_patches
        self.tick_interval = tick_interval

        self._loop_task: asyncio.Task | None = None
        self._stopped = False

    async def start(self):
        """Starts the background maintenance loop."""
        if self._loop_task and not self._loop_task.done():
            logger.warning("Memory maintenance loop is already running.")
            return

        self._stopped = False
        self._loop_task = asyncio.create_task(self._run_loop())
        logger.info("Started background memory maintenance loop.")

    async def stop(self):
        """Stops the background maintenance loop."""
        self._stopped = True
        if self._loop_task:
            self._loop_task.cancel()
            try:
                await self._loop_task
            except asyncio.CancelledError:
                pass
            self._loop_task = None
        logger.info("Stopped background memory maintenance loop.")

    async def _run_loop(self):
        """Internal loop that periodically checks for due tasks."""
        state_dict = self.initial_state
        try:
            while not self._stopped:
                # run_memory_maintenance handles identifying due tasks and applying patches
                results, state_dict = await run_memory_maintenance(
                    llm=self.llm,
                    backend=self.backend,
                    messages=self.messages,
                    schedule_state_dict=state_dict,
                    tasks=self.tasks,
                    config=self.config,
                    apply_patches=self.apply_patches
                )

                # Optional: log if something was changed
                changed_count = sum(1 for r in results if r.changed)
                if changed_count > 0:
                    logger.info(f"Memory maintenance tick completed: {changed_count} tasks made changes.")

                await asyncio.sleep(self.tick_interval)
        except asyncio.CancelledError:
            # Expected on stop()
            pass
        except Exception as e:
            logger.exception("Critical error in memory maintenance loop")
            raise e

def read_memory(
    backend: Any,
    target: MemoryTarget,
    as_jsonl: bool = True,
    **kwargs: Any,
) -> list[dict[str, Any]] | str:
    """
    High-level entry point to read memory for a specific target.

    Encapsulates MemoryStore creation.

    Args:
        backend: The backend protocol implementation.
        target: The memory target to read ('profile', 'episodic', or 'semantic').
        as_jsonl: If True, returns a list of records. If False, returns raw text.
        **kwargs: Reserved for future retrieval options such as:
            - `limit`: Max number of records to return.
            - `offset`: Number of records to skip.
            - `filter_func`: A predicate to filter records.
            - `sort_by`: Field to sort records by.
    """
    store = BackendMemoryStore(backend)
    path = DEFAULT_MEMORY_TARGET_PATHS.get(target, "")

    if as_jsonl:
        # For now, read all records. kwargs are reserved for future
        # implementation of selective/batch retrieval in BackendMemoryStore.
        return store.read_jsonl(path)
    return store.read_text(path)

def format_memory_for_llm(
    backend: Any,
    target: MemoryTarget | None = None,
    **kwargs: Any,
) -> str:
    """
    Reads memory for a target and formats it as a human-readable string
    suitable for LLM context injection.

    Converts structured JSONL records into a clean text format that the
    LLM can easily parse as context.

    Args:
        backend: The backend protocol implementation.
        target: The memory target to read.
        **kwargs: Passed to `read_memory` to support selective or batch retrieval.
    """
    records = read_memory(backend, target, as_jsonl=True, **kwargs)
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
    "MEMORY_EXTRACTION_INSTRUCTIONS",
    "build_memory_extraction_prompt",
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
    "MemoryMaintenanceManager",
    "read_memory",
    "format_memory_for_llm",
]
