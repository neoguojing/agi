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
from agi.agent.context.memory_store import  DEFAULT_MEMORY_TARGET_PATHS

if TYPE_CHECKING:
    from langchain_core.messages import AnyMessage
    from langchain_core.language_models.chat_models import BaseChatModel

    from agi.agent.context.memory_store import MemoryStore
    from deepagents.backends.protocol import BackendProtocol
else:
    AnyMessage = Any
    BaseChatModel = Any
    BackendProtocol = Any
    MemoryStore = Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MemoryTaskConfig:
    """Independent scheduling and quality gates for one memory task.
    
    Defines how often a task should run and the minimum confidence 
    required for a memory to be accepted.
    """

    enabled: bool = True
    interval_seconds: int = 3600
    min_confidence: float = 0.5


@dataclass(frozen=True)
class MemoryMaintenanceConfig:
    """Configuration shared by background memory processing.
    
    Provides a centralized way to configure the intervals and confidence 
    thresholds for all standard memory tasks.
    """

    target_paths: dict[MemoryTarget, str] = field(default_factory=lambda: dict(DEFAULT_MEMORY_TARGET_PATHS))
    profile: MemoryTaskConfig = field(default_factory=lambda: MemoryTaskConfig(interval_seconds=60, min_confidence=0.75))
    episodic: MemoryTaskConfig = field(default_factory=lambda: MemoryTaskConfig(interval_seconds=30, min_confidence=0.45))
    semantic: MemoryTaskConfig = field(default_factory=lambda: MemoryTaskConfig(interval_seconds=60, min_confidence=0.65))


@dataclass
class MemoryTaskContext:
    """Runtime inputs passed to memory maintenance tasks.

    Contains all dependencies required by a task to perform its work,
    including the store, the LLM, and the current conversation history.
    """

    store: MemoryStore
    backend: BackendProtocol
    llm: BaseChatModel
    messages: list[AnyMessage] = field(default_factory=list)
    message_provider: MessageProvider | None = None
    now: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    runtime: Any | None = None
    state: dict[str, Any] = field(default_factory=dict)

    def get_messages(self) -> list[AnyMessage]:
        """Returns the current messages, preferring the provider if available."""
        if self.message_provider:
            return self.message_provider.get_messages()
        return self.messages


@dataclass(frozen=True)
class MemoryTaskResult:
    """Result returned by a memory task.
    
    Contains the outcome of a task execution, including any 
    MemoryPatches that should be applied to the store.
    """

    task_name: str
    changed: bool = False
    summary: str = ""
    patches: tuple[MemoryPatch, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryTaskScheduleState:
    """Serializable bookkeeping used by MemoryTaskScheduler.
    
    Tracks when each task was last executed to determine if it is due 
    based on its `interval_seconds`.
    """

    last_run_at: dict[str, datetime] = field(default_factory=dict)

    @classmethod
    def from_iso_dict(cls, values: dict[str, str]) -> "MemoryTaskScheduleState":
        """Creates a state object from a dictionary of ISO timestamps."""
        parsed: dict[str, datetime] = {}
        for task_name, value in values.items():
            try:
                parsed[task_name] = datetime.fromisoformat(value)
            except ValueError:
                logger.warning("Skipping invalid task schedule timestamp for %s: %s", task_name, value)
        return cls(last_run_at=parsed)

    def to_iso_dict(self) -> dict[str, str]:
        """Converts the state object to a dictionary of ISO timestamps for serialization."""
        return {task_name: timestamp.isoformat() for task_name, timestamp in self.last_run_at.items()}


@runtime_checkable
class MessageProvider(Protocol):
    """Interface for dynamically retrieving the current conversation history."""
    def get_messages(self) -> list[AnyMessage]:
        ...


@runtime_checkable
class MemoryTask(Protocol):
    """Independent memory maintenance unit.

    A protocol that defines the interface for any task that wants to
    be managed by the `MemoryTaskScheduler`.
    """

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
    """Small interval scheduler for independently configured memory tasks.
    
    Manages the execution timing of tasks and ensures that patches 
    emitted by tasks are applied to the store.
    """

    def due_tasks(
        self,
        tasks: Sequence[MemoryTask],
        context: MemoryTaskContext,
        state: MemoryTaskScheduleState | None = None,
    ) -> list[MemoryTask]:
        """Identifies which tasks are due to run based on their configuration and last run time."""
        schedule_state = state or MemoryTaskScheduleState()
        due: list[MemoryTask] = []
        for task in tasks:
            if not task.should_run(context):
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
        """Updates the schedule state to mark the provided tasks as having run at the current time."""
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
        """
        Executes all due tasks and optionally applies their patches to the store.
        
        Returns a list of results and the updated schedule state.
        """
        due = self.due_tasks(tasks, context, state)
        logger.info("ready to run: %s",due)
        results: list[MemoryTaskResult] = []
        for task in due:
            result = await task.run(context)
            if apply_patches:
                for patch in result.patches:
                    context.store.apply_patch(patch)
            results.append(result)

        return results, self.mark_completed(due, context, state)
