"""Memory task configuration, scheduler, and task contracts.

Usage:
    A background maintenance loop creates a `MemoryTaskContext`, asks
    `MemoryTaskScheduler` for due tasks, runs them, and applies any emitted
    patches through the context store.

    Example:
        scheduler = MemoryTaskScheduler()
        results, state = await scheduler.run_due_tasks(tasks, context, state)

This module does not define how an LLM extracts memories. Tasks can call the
LLM-facing helpers in `memory_extraction` when they need model extraction.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any, Protocol, Sequence, runtime_checkable

from agi.agent.context.memory_models import MemoryPatch, MemoryTarget
from agi.agent.context.memory_store import DEFAULT_LEGACY_MEMORY_PATHS, DEFAULT_MEMORY_TARGET_PATHS

if TYPE_CHECKING:
    from langchain_core.messages import AnyMessage

    from agi.agent.context.memory_store import MemoryStore
    from deepagents.backends.protocol import BackendProtocol
else:
    AnyMessage = Any
    BackendProtocol = Any
    MemoryStore = Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MemoryTaskConfig:
    """Independent scheduling and quality gates for one memory task."""

    enabled: bool = True
    interval_seconds: int = 3600
    min_confidence: float = 0.5


@dataclass(frozen=True)
class MemoryMaintenanceConfig:
    """Configuration shared by background memory processing."""

    legacy_paths: tuple[str, ...] = DEFAULT_LEGACY_MEMORY_PATHS
    target_paths: dict[MemoryTarget, str] = field(default_factory=lambda: dict(DEFAULT_MEMORY_TARGET_PATHS))
    profile: MemoryTaskConfig = field(default_factory=lambda: MemoryTaskConfig(interval_seconds=24 * 3600, min_confidence=0.75))
    episodic: MemoryTaskConfig = field(default_factory=lambda: MemoryTaskConfig(interval_seconds=3600, min_confidence=0.45))
    semantic: MemoryTaskConfig = field(default_factory=lambda: MemoryTaskConfig(interval_seconds=24 * 3600, min_confidence=0.65))


@dataclass
class MemoryTaskContext:
    """Runtime inputs passed to memory maintenance tasks."""

    store: MemoryStore
    backend: BackendProtocol
    messages: list[AnyMessage] = field(default_factory=list)
    legacy_memory: dict[str, str] = field(default_factory=dict)
    now: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    runtime: Any | None = None
    state: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MemoryTaskResult:
    """Result returned by a memory task."""

    task_name: str
    changed: bool = False
    summary: str = ""
    patches: tuple[MemoryPatch, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryTaskScheduleState:
    """Serializable bookkeeping used by MemoryTaskScheduler."""

    last_run_at: dict[str, datetime] = field(default_factory=dict)

    @classmethod
    def from_iso_dict(cls, values: dict[str, str]) -> "MemoryTaskScheduleState":
        parsed: dict[str, datetime] = {}
        for task_name, value in values.items():
            try:
                parsed[task_name] = datetime.fromisoformat(value)
            except ValueError:
                logger.warning("Skipping invalid task schedule timestamp for %s: %s", task_name, value)
        return cls(last_run_at=parsed)

    def to_iso_dict(self) -> dict[str, str]:
        return {task_name: timestamp.isoformat() for task_name, timestamp in self.last_run_at.items()}


@runtime_checkable
class MemoryTask(Protocol):
    """Independent memory maintenance unit."""

    name: str
    target: MemoryTarget
    config: MemoryTaskConfig

    def should_run(self, context: MemoryTaskContext) -> bool:
        """Return whether this task should run for the current maintenance tick."""
        ...

    async def run(self, context: MemoryTaskContext) -> MemoryTaskResult:
        """Produce patches or reports for this memory task."""
        ...


class MemoryTaskScheduler:
    """Small interval scheduler for independently configured memory tasks."""

    def due_tasks(
        self,
        tasks: Sequence[MemoryTask],
        context: MemoryTaskContext,
        state: MemoryTaskScheduleState | None = None,
    ) -> list[MemoryTask]:
        schedule_state = state or MemoryTaskScheduleState()
        due: list[MemoryTask] = []
        for task in tasks:
            if not task.config.enabled or not task.should_run(context):
                continue

            last_run_at = schedule_state.last_run_at.get(task.name)
            if last_run_at is None:
                due.append(task)
                continue

            next_run_at = last_run_at + timedelta(seconds=task.config.interval_seconds)
            if context.now >= next_run_at:
                due.append(task)

        return due

    def mark_completed(
        self,
        tasks: Sequence[MemoryTask],
        context: MemoryTaskContext,
        state: MemoryTaskScheduleState | None = None,
    ) -> MemoryTaskScheduleState:
        schedule_state = state or MemoryTaskScheduleState()
        for task in tasks:
            schedule_state.last_run_at[task.name] = context.now
        return schedule_state

    async def run_due_tasks(
        self,
        tasks: Sequence[MemoryTask],
        context: MemoryTaskContext,
        state: MemoryTaskScheduleState | None = None,
        *,
        apply_patches: bool = True,
    ) -> tuple[list[MemoryTaskResult], MemoryTaskScheduleState]:
        due = self.due_tasks(tasks, context, state)
        results: list[MemoryTaskResult] = []
        for task in due:
            result = await task.run(context)
            if apply_patches:
                for patch in result.patches:
                    context.store.apply_patch(patch)
            results.append(result)

        return results, self.mark_completed(due, context, state)


class LegacyMemoryInspectionTask:
    """No-op task that proves the MemoryTask contract without rewriting memory."""

    name = "legacy_memory_inspection"
    target: MemoryTarget = "episodic"

    def __init__(self, config: MemoryTaskConfig | None = None) -> None:
        self.config = config or MemoryTaskConfig(enabled=True, interval_seconds=3600)

    def should_run(self, context: MemoryTaskContext) -> bool:
        return self.config.enabled

    async def run(self, context: MemoryTaskContext) -> MemoryTaskResult:
        return MemoryTaskResult(
            task_name=self.name,
            changed=False,
            summary=f"Loaded {len(context.legacy_memory)} legacy memory files for inspection.",
            metadata={"legacy_paths": sorted(context.legacy_memory)},
        )
