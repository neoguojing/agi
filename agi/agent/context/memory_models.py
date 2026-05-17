"""Typed memory records and patch primitives.

Usage:
    These classes are the stable data contract between an LLM memory extractor,
    memory maintenance tasks, and storage backends. An LLM extraction result is
    parsed into `MemoryExtractionResult`; callers then convert it to auditable
    `MemoryPatch` objects and apply those patches through a `MemoryStore`.

    Example:
        extraction = MemoryExtractionResult(
            semantic_memories=(SemanticMemoryRecord(...),),
        )
        patches = extraction.to_patches()

Design notes:
    - Profile memory stores stable key/value user profile and preferences.
    - Episodic memory stores time-bound events that can decay or expire.
    - Semantic memory is graph-ready and shaped as subject/predicate/object.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from typing import Any, Literal

# Target identifiers for different memory types
MemoryTarget = Literal["profile", "episodic", "semantic"]
# Types of operations allowed in a memory patch
MemoryOperationType = Literal["add", "update", "delete", "merge", "deprecate"]
# Valid sources for memory extraction
MemorySourceKind = Literal[
    "user_explicit",
    "conversation",
    "inferred",
    "legacy_memory",
    "system",
    "tool",
]


@dataclass(frozen=True)
class MemoryOperation:
    """A single storage-neutral mutation inside a memory patch.
    
    Represents a specific change to a memory record, such as adding a new 
    fact or updating an existing one.
    """

    op: MemoryOperationType
    value: dict[str, Any] = field(default_factory=dict)
    target_id: str | None = None
    reason: str | None = None


@dataclass(frozen=True)
class MemoryPatch:
    """Auditable memory changes emitted by tasks instead of direct writes.
    
    A patch groups multiple operations for a specific memory target. 
    This allows for atomic updates and provides an audit trail (reason, confidence).
    """

    target: MemoryTarget
    operations: tuple[MemoryOperation, ...] = ()
    target_path: str | None = None
    reason: str = ""
    confidence: float = 1.0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def is_empty(self) -> bool:
        """Returns True if the patch contains no operations."""
        return not self.operations

    @classmethod
    def empty(cls, target: MemoryTarget, *, reason: str = "") -> "MemoryPatch":
        """Creates an empty patch for a specific target."""
        return cls(target=target, reason=reason, operations=())


@dataclass(frozen=True)
class MemoryEvidence:
    """Evidence attached to model-extracted memory records.
    
    Provides the 'why' behind a memory, linking it back to a specific 
    message or source for verification.
    """

    source: MemorySourceKind = "conversation"
    content: str = ""
    message_id: str | None = None
    memory_id: str | None = None
    created_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        """Converts the evidence dataclass to a JSON-ready dictionary."""
        return json_ready(asdict(self))


@dataclass(frozen=True)
class ProfileMemoryRecord:
    """Structured model output for stable profile/preference memory.
    
    Used for long-term user attributes (e.g., 'user.language': 'English').
    """

    id: str = ""
    key: str = ""
    value: Any | None = None
    confidence: float = 0.0
    importance: float = 0.0
    source: MemorySourceKind = "inferred"
    evidence: tuple[MemoryEvidence, ...] = ()
    tags: tuple[str, ...] = ()
    created_at: datetime | None = None
    updated_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        """Converts the record to a JSON-ready dictionary with a type marker."""
        record = json_ready(asdict(self))
        record["type"] = "profile"
        return record

    def to_operation(self, op: MemoryOperationType = "add") -> MemoryOperation:
        """Converts this record into a MemoryOperation for use in a patch."""
        return MemoryOperation(op=op, target_id=self.id or None, value=self.to_record())


@dataclass(frozen=True)
class EpisodicMemoryRecord:
    """Structured model output for time-bound event memory.
    
    Used for specific occurrences or experiences (e.g., 'User mentioned they 
    started a new project on 2023-10-01').
    """

    id: str = ""
    summary: str = ""
    event_time: datetime | None = None
    participants: tuple[str, ...] = ()
    outcome: str | None = None
    context: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    importance: float = 0.0
    ttl_days: int | None = None
    expires_at: datetime | None = None
    evidence: tuple[MemoryEvidence, ...] = ()
    tags: tuple[str, ...] = ()
    created_at: datetime | None = None
    updated_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        """Converts the record to a JSON-ready dictionary with a type marker."""
        record = json_ready(asdict(self))
        record["type"] = "episodic"
        return record

    def to_operation(self, op: MemoryOperationType = "add") -> MemoryOperation:
        """Converts this record into a MemoryOperation for use in a patch."""
        return MemoryOperation(op=op, target_id=self.id or None, value=self.to_record())


@dataclass(frozen=True)
class SemanticEntity:
    """Graph-ready node reference used by semantic memory triples.
    
    Represents a concept, person, or object in a knowledge graph.
    """

    id: str = ""
    kind: str = "concept"
    label: str | None = None
    properties: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        """Converts the entity to a JSON-ready dictionary."""
        return json_ready(asdict(self))


@dataclass(frozen=True)
class SemanticObject:
    """Graph-ready object value or node reference for semantic memory.
    
    The target of a predicate in a semantic triple.
    """

    id: str | None = None
    kind: str = "value"
    value: Any | None = None
    label: str | None = None
    properties: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        """Converts the object to a JSON-ready dictionary."""
        return json_ready(asdict(self))


@dataclass(frozen=True)
class SemanticMemoryRecord:
    """Structured model output for graph-ready long-term knowledge.
    
    Represents a triple: Subject -> Predicate -> Object.
    """

    id: str = ""
    subject: SemanticEntity = field(default_factory=SemanticEntity)
    predicate: str = ""
    object: SemanticObject = field(default_factory=SemanticObject)
    qualifiers: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    importance: float = 0.0
    evidence: tuple[MemoryEvidence, ...] = ()
    tags: tuple[str, ...] = ()
    created_at: datetime | None = None
    updated_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        """Converts the record to a JSON-ready dictionary with a type marker."""
        record = json_ready(asdict(self))
        record["type"] = "semantic"
        return record

    def to_operation(self, op: MemoryOperationType = "add") -> MemoryOperation:
        """Converts this record into a MemoryOperation for use in a patch."""
        return MemoryOperation(op=op, target_id=self.id or None, value=self.to_record())


@dataclass(frozen=True)
class MemoryExtractionResult:
    """Top-level structured return for an LLM memory-extraction call.
    
    Aggregates all candidates extracted across different memory types from a 
    single conversation window.
    """

    profile_memories: tuple[ProfileMemoryRecord, ...] = ()
    episodic_memories: tuple[EpisodicMemoryRecord, ...] = ()
    semantic_memories: tuple[SemanticMemoryRecord, ...] = ()
    rejected_candidates: tuple[str, ...] = ()
    notes: str = ""

    def to_patches(self, *, reason: str = "model structured memory extraction") -> tuple[MemoryPatch, ...]:
        """Converts the extraction results into a set of MemoryPatches, one per target."""
        patches: list[MemoryPatch] = []
        if self.profile_memories:
            patches.append(MemoryPatch(target="profile", reason=reason, operations=tuple(m.to_operation() for m in self.profile_memories)))
        if self.episodic_memories:
            patches.append(MemoryPatch(target="episodic", reason=reason, operations=tuple(m.to_operation() for m in self.episodic_memories)))
        if self.semantic_memories:
            patches.append(MemoryPatch(target="semantic", reason=reason, operations=tuple(m.to_operation() for m in self.semantic_memories)))
        return tuple(patches)

    def to_record(self) -> dict[str, Any]:
        """Converts the result to a JSON-ready dictionary."""
        return json_ready(asdict(self))

    @classmethod
    def from_record(cls, payload: dict[str, Any]) -> "MemoryExtractionResult":
        """Coerce an LLM JSON object into typed memory extraction records."""

        from agi.agent.context.memory_extraction import parse_memory_extraction_result

        return parse_memory_extraction_result(payload)


def json_ready(value: Any) -> Any:
    """Convert memory dataclasses and datetimes into JSON-compatible primitives.
    
    Recursively handles dataclasses, lists, and dictionaries to ensure 
    all types are serializable to JSON.
    """

    if isinstance(value, datetime):
        return value.isoformat()
    if is_dataclass(value) and not isinstance(value, type):
        return json_ready(asdict(value))
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items() if item is not None}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value
