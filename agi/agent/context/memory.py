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
    LegacyMemoryInspectionTask,
    MemoryMaintenanceConfig,
    MemoryTask,
    MemoryTaskConfig,
    MemoryTaskContext,
    MemoryTaskResult,
    MemoryTaskScheduleState,
    MemoryTaskScheduler,
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
    "LegacyMemoryInspectionTask",
]
