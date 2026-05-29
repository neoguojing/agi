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
from pydantic import ValidationError

from agi.agent.context.memory_extraction import (
    MEMORY_EXTRACTION_INSTRUCTIONS,
    build_memory_extraction_prompt,
)


from agi.agent.context.memory_models import (
    EpisodicMemoryRecord,
    MemoryExtractionResult,
    MemoryOperation,
    MemoryOperationType,
    MemoryPatch,
    MemorySourceKind,
    MemoryTarget,
    ProfileMemoryRecord,
    SemanticMemoryRecord,
    ProfileMemoryList,
    EpisodicMemoryList,
    SemanticMemoryList,
    record_dedup_key,
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
        if not context.llm:
            return None
        try:
            schema = TARGET_SCHEMA_MAP.get(self.target)
            llm_with_struct = context.llm.with_structured_output(schema)
            struct_result = await llm_with_struct.ainvoke(
                prompt, config={"configurable": {"thread_id": str(self.target)}}
            )
            logger.info("Task %s struct result=%s", self.name, struct_result)
            return struct_result
        except Exception:
            logger.exception("Task %s LLM call failed", self.name)
            return None

    async def run(self, context: MemoryTaskContext) -> MemoryTaskResult:
        """Executes the memory extraction process for the target memory type."""
        logger.info(f"Executing memory task: {self.name} for target: {self.target}")
        # 1. Prepare conversation history as string
        conversation_text = "\n".join([str(m.content) for m in context.get_messages()])

        # 2. Load existing memory for the target to provide context to the LLM
        existing_mem_text = await context.store.read_text(self.target)

        # 3. Build prompt
        prompt = build_memory_extraction_prompt(
            conversation=conversation_text,
            existing_memory=existing_mem_text,
            target=self.target,
        )
        
        # 4. Get LLM result
        llm_payload = await self._call_llm_for_extraction(context, prompt)

        if not llm_payload:
            logger.error(f"Task {self.name} failed: LLM returned no result")
            return MemoryTaskResult(task_name=self.name, changed=False, summary="LLM call failed")

        # 5. Parse and convert to patches
        filtered_extraction = MemoryExtractionResult()
        
        if isinstance(llm_payload, ProfileMemoryList):
            filtered_extraction.profile_memories = llm_payload.items
        elif isinstance(llm_payload, EpisodicMemoryList):
            filtered_extraction.episodic_memories = llm_payload.items
        elif isinstance(llm_payload, SemanticMemoryList):
            filtered_extraction.semantic_memories = llm_payload.items
        
        patches = filtered_extraction.to_patches(reason=f"Automatic {self.target} memory extraction")
        patches = self._filter_and_deduplicate_patches(patches, await context.store.read_jsonl(self.target))
        patch_strategy = "replace" if self.target == "profile" else "merge"
        patches = tuple(
            patch.model_copy(update={"strategy": patch_strategy})
            for patch in patches
        )
        
        # Filter patches to only include the target this task is responsible for
        target_patches = tuple(p for p in patches if p.target == self.target)

        logger.info(f"Task {self.name} completed: extracted {len(target_patches)} patches for {self.target}")

        return MemoryTaskResult(
            task_name=self.name,
            changed=bool(target_patches),
            summary=f"Extracted {len(target_patches)} patches for {self.target} memory.",
            patches=target_patches
        )

    def _filter_and_deduplicate_patches(
        self,
        patches: tuple[MemoryPatch, ...],
        existing_records: list[dict[str, Any]],
    ) -> tuple[MemoryPatch, ...]:
        """Filter low-confidence and duplicate operations before persistence."""
        updated: list[MemoryPatch] = []
        seen_keys: set[tuple[Any, ...]] = set()
        _ = existing_records

        for patch in patches:
            operations: list[MemoryOperation] = []
            for op in patch.operations:
                if op.op != "add":
                    operations.append(op)
                    continue

                confidence = float(op.value.get("confidence", 0.5) or 0.0)
                if confidence < self.config.min_confidence:
                    continue

                key = record_dedup_key(self.target, op.value)
                # In replace mode we still remove duplicates inside the current
                # extraction batch, but do not force retention of historical
                # records from existing store.
                if key and key in seen_keys:
                    continue
                if key:
                    seen_keys.add(key)

                operations.append(op)

            if operations:
                updated.append(patch.model_copy(update={"operations": tuple(operations)}))

        return tuple(updated)

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

class MemoryMaintenanceManager:
    """
    Unified memory maintenance manager.

    Responsibilities:
    - Hold long-lived runtime objects
    - Maintain scheduler state
    - Execute due tasks
    - Run optional background loop
    """

    def __init__(
        self,
        llm: Any,
        backend: Any,
        messages: Sequence[Any] | MessageProvider,
        config: MemoryMaintenanceConfig | None = None,
        tasks: Sequence["MemoryTask"] | None = None,
        initial_state: dict[str, str] | None = None,
        apply_patches: bool = True,
        tick_interval: int = 60,
    ):
        self.llm = llm
        self.backend = backend
        self.messages = messages

        self.config = config or MemoryMaintenanceConfig()

        self.apply_patches = apply_patches
        self.tick_interval = tick_interval

        # Runtime state
        self._loop_task: asyncio.Task | None = None
        self._stopped = False

        # ------------------------------------------------------------------
        # Long-lived objects
        # ------------------------------------------------------------------

        self.store = BackendMemoryStore(self.backend, target_paths=self.config.target_paths)

        if isinstance(messages, MessageProvider):
            self.context = MemoryTaskContext(
                store=self.store,
                backend=self.backend,
                llm=self.llm,
                message_provider=messages,
            )
        else:
            self.context = MemoryTaskContext(
                store=self.store,
                backend=self.backend,
                llm=self.llm,
                messages=list(messages),
            )

        self.tasks = list(tasks) if tasks else [
            ProfileMemoryTask(self.config.profile),
            EpisodicMemoryTask(self.config.episodic),
            SemanticMemoryTask(self.config.semantic),
        ]

        self.state = (
            MemoryTaskScheduleState.from_iso_dict(initial_state)
            if initial_state
            else MemoryTaskScheduleState()
        )

        self.scheduler = MemoryTaskScheduler()

    # ======================================================================
    # Public API
    # ======================================================================

    async def tick(self):
        """
        Run one maintenance tick.

        Executes all due tasks and updates internal schedule state.
        """
        results = []
        try:
            
            results, self.state = await self.scheduler.run_due_tasks(
                tasks=self.tasks, 
                context=self.context, 
                state=self.state, 
                apply_patches=self.apply_patches
            )
            
        except Exception:
            logger.exception(
                "Memory task failed!",
            )

        changed_count = sum(1 for r in results if r.changed)

        if changed_count > 0:
            logger.info(
                "Memory maintenance tick completed: %s tasks made changes.",
                changed_count,
            )

    async def start(self):
        """
        Start background maintenance loop and initialize memories.
        """
        if self._loop_task and not self._loop_task.done():
            logger.warning(
                "Memory maintenance loop already running."
            )
            return

        self._stopped = False

        self._loop_task = asyncio.create_task(
            self._run_loop(),
            name="memory-maintenance-loop",
        )

        logger.info(
            "Started memory maintenance loop."
        )

    async def load_memories(self, targets: list[MemoryTarget] = None) -> MemoryExtractionResult:
        """Explicitly load memories from the backend store."""
        logger.info("Loading memories from storage...")
        if targets is None:
            targets = ["profile", "episodic", "semantic"]

        result = MemoryExtractionResult()
        target_to_model = {
            "profile": (result.profile_memories, ProfileMemoryRecord),
            "episodic": (result.episodic_memories, EpisodicMemoryRecord),
            "semantic": (result.semantic_memories, SemanticMemoryRecord),
        }

        for target in targets:
            container, model = target_to_model[target]
            for record in await self.store.read_jsonl(target):
                try:
                    container.append(model.model_validate(record))
                except ValidationError as exc:
                    logger.warning("Invalid %s memory record: %s", target, exc)

        return result

    async def flush(self) -> None:
        """
        Ensure all pending memory changes are persisted to the backend.
        In the current implemention, apply_patch is immediate, so this is primarily
        for triggering a final maintenance tick to consolidate memory.
        """
        logger.info("Flushing memories to storage...")
        await self.tick()

    async def stop(self):
        """
        Stop background maintenance loop and flush final state.
        """
        # 1. Final flush before exiting
        await self.flush()

        self._stopped = True

        if self._loop_task:

            self._loop_task.cancel()

            try:
                await self._loop_task
            except asyncio.CancelledError:
                pass

            self._loop_task = None

        logger.info(
            "Stopped memory maintenance loop."
        )

    async def force_tick(self, target: str | None = None) -> str:
        """
        Force run memory tasks regardless of their schedule.
        If target is provided, only run the task matching that target.
        """
        results = []
        try:
            for task in self.tasks:
                # If a target is specified, only run the task that matches that target
                if target and task.target != target:
                    continue

                if not task.should_run(self.context):
                    continue

                result = await task.run(self.context)
                if self.apply_patches:
                    for patch in result.patches:
                        await self.store.apply_patch(patch)
                results.append(result)

            completed_tasks = [t for t in self.tasks if t.name in [r.task_name for r in results]]
            self.state = self.scheduler.mark_completed(completed_tasks, self.context, self.state)

        except Exception:
            logger.exception("Force memory maintenance tick failed!")

        changed_count = sum(1 for r in results if r.changed)
        if changed_count > 0:
            logger.info("Force memory maintenance tick completed: %s tasks made changes.", changed_count)

        return f"Memory organized. {changed_count} tasks made changes."


    # ======================================================================
    # Internal
    # ======================================================================

    async def _run_loop(self):
        """
        Internal background loop.
        """

        try:

            while not self._stopped:

                try:
                    await self.tick()

                except Exception:
                    logger.exception(
                        "Error during memory maintenance tick"
                    )

                await asyncio.sleep(self.tick_interval)

        except asyncio.CancelledError:
            logger.debug(
                "Memory maintenance loop cancelled."
            )
            raise

__all__ = [
    "MemoryTarget",
    "MemoryOperationType",
    "MemorySourceKind",
    "MemoryOperation",
    "MemoryPatch",
    "ProfileMemoryRecord",
    "EpisodicMemoryRecord",
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
    "MemoryMaintenanceManager"
]
