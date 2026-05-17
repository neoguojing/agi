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

    op: MemoryOperationType  # The type of operation (e.g., 'add', 'update')
    value: dict[str, Any] = field(default_factory=dict)  # The record data to be written
    target_id: str | None = None  # Unique identifier of the record being targeted
    reason: str | None = None  # Justification for this specific operation


@dataclass(frozen=True)
class MemoryPatch:
    """Auditable memory changes emitted by tasks instead of direct writes.
    
    A patch groups multiple operations for a specific memory target. 
    This allows for atomic updates and provides an audit trail (reason, confidence).
    """

    target: MemoryTarget  # Which memory store this patch applies to
    operations: tuple[MemoryOperation, ...] = ()  # Sequence of mutations to perform
    target_path: str | None = None  # Optional override for the storage path
    reason: str = ""  # Overall reason for this patch (e.g., 'Automatic extraction')
    confidence: float = 1.0  # Model's confidence in the correctness of these changes
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))  # Timestamp of patch creation

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

    source: MemorySourceKind = "conversation"  # Where the information came from
    content: str = ""  # The actual quote or snippet from the source
    message_id: str | None = None  # ID of the message if source is 'conversation'
    memory_id: str | None = None  # ID of the existing memory if source is 'legacy_memory'
    created_at: datetime | None = None  # When the evidence was captured
    metadata: dict[str, Any] = field(default_factory=dict)  # Additional context (e.g., tool output)

    def to_record(self) -> dict[str, Any]:
        """Converts the evidence dataclass to a JSON-ready dictionary."""
        return json_ready(asdict(self))


@dataclass(frozen=True)
class ProfileMemoryRecord:
    """Structured model output for stable profile/preference memory.
    
    Used for long-term user attributes (e.g., 'user.language': 'English').
    """

    id: str = ""  # Unique identifier for the profile record
    key: str = ""  # The attribute name (e.g., 'preferences.theme')
    value: Any | None = None  # The value of the attribute
    confidence: float = 0.0  # Model's confidence in this fact
    importance: float = 0.0  # Relative importance of this fact to the user
    source: MemorySourceKind = "inferred"  # How this fact was discovered
    evidence: tuple[MemoryEvidence, ...] = ()  # Supporting evidence
    tags: tuple[str, ...] = ()  # Categorization tags
    created_at: datetime | None = None  # Initial creation timestamp
    updated_at: datetime | None = None  # Last modification timestamp
    metadata: dict[str, Any] = field(default_factory=dict)  # Extra unstructured data

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

    id: str = ""  # Unique identifier for the episode
    summary: str = ""  # Concise description of the event
    event_time: datetime | None = None  # When the event actually occurred
    participants: tuple[str, ...] = ()  # Entities involved in the event
    outcome: str | None = None  # The result or conclusion of the event
    context: dict[str, Any] = field(default_factory=dict)  # Surrounding circumstances
    confidence: float = 0.0  # Model's confidence in the event's accuracy
    importance: float = 0.0  # How significant this event is
    ttl_days: int | None = None  # Time-to-live in days before the memory expires
    expires_at: datetime | None = None  # Absolute expiration timestamp
    evidence: tuple[MemoryEvidence, ...] = ()  # Supporting evidence
    tags: tuple[str, ...] = ()  # Categorization tags
    created_at: datetime | None = None  # When the record was created
    updated_at: datetime | None = None  # When the record was last updated
    metadata: dict[str, Any] = field(default_factory=dict)  # Extra unstructured data

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

    id: str = ""  # Unique identifier for the entity (e.g., 'person:123')
    kind: str = "concept"  # Type of entity (e.g., 'person', 'location', 'concept')
    label: str | None = None  # Human-readable name of the entity
    properties: dict[str, Any] = field(default_factory=dict)  # Key-value attributes of the entity

    def to_record(self) -> dict[str, Any]:
        """Converts the entity to a JSON-ready dictionary."""
        return json_ready(asdict(self))


@dataclass(frozen=True)
class SemanticObject:
    """Graph-ready object value or node reference for semantic memory.
    
    The target of a predicate in a semantic triple.
    """

    id: str | None = None  # ID if the object is another entity; None if it's a literal value
    kind: str = "value"  # 'entity' if it refers to a node, 'value' if it's a literal
    value: Any | None = None  # The literal value if kind is 'value'
    label: str | None = None  # Human-readable label for the object
    properties: dict[str, Any] = field(default_factory=dict)  # Extra attributes

    def to_record(self) -> dict[str, Any]:
        """Converts the object to a JSON-ready dictionary."""
        return json_ready(asdict(self))


@dataclass(frozen=True)
class SemanticMemoryRecord:
    """Structured model output for graph-ready long-term knowledge.
    
    Represents a triple: Subject -> Predicate -> Object.
    """

    id: str = ""  # Unique identifier for the triple
    subject: SemanticEntity = field(default_factory=SemanticEntity)  # The entity the fact is about
    predicate: str = ""  # The relationship or property (e.g., 'works_at', 'is_a')
    object: SemanticObject = field(default_factory=SemanticObject)  # The value or entity the subject is linked to
    qualifiers: dict[str, Any] = field(default_factory=dict)  # Contextual modifiers (e.g., 'since': '2020')
    confidence: float = 0.0  # Model's confidence in this relationship
    importance: float = 0.0  # Relative importance of this knowledge
    evidence: tuple[MemoryEvidence, ...] = ()  # Supporting evidence
    tags: tuple[str, ...] = ()  # Categorization tags
    created_at: datetime | None = None  # When the record was created
    updated_at: datetime | None = None  # When the record was last updated
    metadata: dict[str, Any] = field(default_factory=dict)  # Extra unstructured data

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

    profile_memories: tuple[ProfileMemoryRecord, ...] = ()  # Extracted profile facts
    episodic_memories: tuple[EpisodicMemoryRecord, ...] = ()  # Extracted events
    semantic_memories: tuple[SemanticMemoryRecord, ...] = ()  # Extracted knowledge triples
    rejected_candidates: tuple[str, ...] = ()  # Facts the model considered but decided to reject
    notes: str = ""  # General model commentary on the extraction process

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
