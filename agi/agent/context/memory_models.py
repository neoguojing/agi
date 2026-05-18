"""Typed memory records and patch primitives.

Usage:
    These classes define the structured contract between:
    - LLM memory extraction
    - memory maintenance tasks
    - storage backends

    The schema is optimized for:
    - structured LLM output
    - JSON serialization
    - patch-based memory updates
    - graph/vector memory systems
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field


# =========================================================
# Type Aliases
# =========================================================

MemoryTarget = Literal["profile", "episodic", "semantic"]

MemoryOperationType = Literal[
    "add",
    "update",
    "delete",
    "merge",
    "deprecate",
]

MemorySourceKind = Literal[
    "user_explicit",
    "conversation",
    "inferred",
    "legacy_memory",
    "system",
    "tool",
]


# =========================================================
# Patch Layer
# =========================================================

class MemoryOperation(BaseModel):
    """A storage-neutral mutation operation."""

    op: MemoryOperationType = Field(
        description="Type of mutation operation."
    )

    value: dict[str, Any] = Field(
        default_factory=dict,
        description="Serialized memory payload."
    )

    target_id: str | None = Field(
        default=None,
        description="Target memory record identifier."
    )

    reason: str | None = Field(
        default=None,
        description="Reason for this operation."
    )


class MemoryPatch(BaseModel):
    """Auditable memory patch containing multiple operations."""

    target: MemoryTarget = Field(
        description="Target memory collection."
    )

    operations: tuple[MemoryOperation, ...] = Field(
        default_factory=tuple,
        description="Operations included in this patch."
    )

    target_path: str | None = Field(
        default=None,
        description="Optional backend-specific storage path."
    )

    reason: str = Field(
        default="",
        description="Reason for this patch."
    )

    confidence: float = Field(
        default=1.0,
        description="Confidence score for this patch."
    )

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Patch creation timestamp."
    )

    @property
    def is_empty(self) -> bool:
        return not self.operations

    @classmethod
    def empty(
        cls,
        target: MemoryTarget,
        *,
        reason: str = "",
    ) -> "MemoryPatch":
        return cls(
            target=target,
            reason=reason,
            operations=(),
        )


# =========================================================
# Evidence
# =========================================================

class MemoryEvidence(BaseModel):
    """Supporting evidence attached to extracted memory."""

    source: MemorySourceKind = Field(
        default="conversation",
        description="Origin of the evidence."
    )

    content: str = Field(
        default="",
        description="Evidence text or source snippet."
    )

    message_id: str | None = Field(
        default=None,
        description="Conversation message identifier."
    )

    memory_id: str | None = Field(
        default=None,
        description="Referenced existing memory identifier."
    )

    created_at: datetime | None = Field(
        default=None,
        description="Evidence creation timestamp."
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional evidence metadata."
    )


# =========================================================
# Profile Memory
# =========================================================

class ProfileMemoryRecord(BaseModel):
    """Stable long-term profile or preference memory."""

    id: str = Field(
        default="",
        description="Unique memory identifier."
    )

    key: str = Field(
        default="",
        description="Profile attribute key."
    )

    value: Any | None = Field(
        default=None,
        description="Profile attribute value."
    )

    confidence: float = Field(
        default=0.0,
        description="Confidence score."
    )

    importance: float = Field(
        default=0.0,
        description="Importance score."
    )

    source: MemorySourceKind = Field(
        default="inferred",
        description="Memory source."
    )

    evidence: tuple[MemoryEvidence, ...] = Field(
        default_factory=tuple,
        description="Supporting evidence."
    )

    tags: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Categorization tags."
    )

    created_at: datetime | None = Field(
        default=None,
        description="Creation timestamp."
    )

    updated_at: datetime | None = Field(
        default=None,
        description="Last update timestamp."
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata."
    )

    def to_operation(
        self,
        op: MemoryOperationType = "add",
    ) -> MemoryOperation:
        return MemoryOperation(
            op=op,
            target_id=self.id or None,
            value=self.model_dump(mode="json"),
        )

class ProfileMemoryList(BaseModel):
    profile_memories: list[ProfileMemoryRecord] = Field(
        description="List of profiles"
    )


# =========================================================
# Episodic Memory
# =========================================================

class EpisodicMemoryRecord(BaseModel):
    """Time-bound event or experience memory."""

    id: str = Field(
        default="",
        description="Unique episodic memory identifier."
    )

    summary: str = Field(
        default="",
        description="Summary of the event."
    )

    event_time: datetime | None = Field(
        default=None,
        description="Time when the event occurred."
    )

    participants: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Entities involved in the event."
    )

    outcome: str | None = Field(
        default=None,
        description="Event outcome."
    )

    context: dict[str, Any] = Field(
        default_factory=dict,
        description="Event context."
    )

    confidence: float = Field(
        default=0.0,
        description="Confidence score."
    )

    importance: float = Field(
        default=0.0,
        description="Importance score."
    )

    ttl_days: int | None = Field(
        default=None,
        description="Memory TTL in days."
    )

    expires_at: datetime | None = Field(
        default=None,
        description="Expiration timestamp."
    )

    evidence: tuple[MemoryEvidence, ...] = Field(
        default_factory=tuple,
        description="Supporting evidence."
    )

    tags: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Categorization tags."
    )

    created_at: datetime | None = Field(
        default=None,
        description="Creation timestamp."
    )

    updated_at: datetime | None = Field(
        default=None,
        description="Last update timestamp."
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata."
    )

    def to_operation(
        self,
        op: MemoryOperationType = "add",
    ) -> MemoryOperation:
        return MemoryOperation(
            op=op,
            target_id=self.id or None,
            value=self.model_dump(mode="json"),
        )

class EpisodicMemoryList(BaseModel):
    episodic_memories: list[EpisodicMemoryRecord] = Field(
        description="List of episodic"
    )

# =========================================================
# Semantic Memory
# =========================================================

class SemanticEntity(BaseModel):
    """Graph entity node."""

    id: str = Field(
        default="",
        description="Entity identifier."
    )

    kind: str = Field(
        default="concept",
        description="Entity type."
    )

    label: str | None = Field(
        default=None,
        description="Human-readable label."
    )

    properties: dict[str, Any] = Field(
        default_factory=dict,
        description="Entity properties."
    )


class SemanticObject(BaseModel):
    """Semantic triple object."""

    id: str | None = Field(
        default=None,
        description="Object entity identifier."
    )

    kind: str = Field(
        default="value",
        description="Object kind."
    )

    value: Any | None = Field(
        default=None,
        description="Literal value."
    )

    label: str | None = Field(
        default=None,
        description="Human-readable label."
    )

    properties: dict[str, Any] = Field(
        default_factory=dict,
        description="Object properties."
    )


class SemanticMemoryRecord(BaseModel):
    """Graph-ready semantic memory triple."""

    id: str = Field(
        default="",
        description="Unique semantic memory identifier."
    )

    subject: SemanticEntity = Field(
        default_factory=SemanticEntity,
        description="Triple subject."
    )

    predicate: str = Field(
        default="",
        description="Relationship predicate."
    )

    object: SemanticObject = Field(
        default_factory=SemanticObject,
        description="Triple object."
    )

    qualifiers: dict[str, Any] = Field(
        default_factory=dict,
        description="Contextual qualifiers."
    )

    confidence: float = Field(
        default=0.0,
        description="Confidence score."
    )

    importance: float = Field(
        default=0.0,
        description="Importance score."
    )

    evidence: tuple[MemoryEvidence, ...] = Field(
        default_factory=tuple,
        description="Supporting evidence."
    )

    tags: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Categorization tags."
    )

    created_at: datetime | None = Field(
        default=None,
        description="Creation timestamp."
    )

    updated_at: datetime | None = Field(
        default=None,
        description="Last update timestamp."
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata."
    )

    def to_operation(
        self,
        op: MemoryOperationType = "add",
    ) -> MemoryOperation:
        return MemoryOperation(
            op=op,
            target_id=self.id or None,
            value=self.model_dump(mode="json"),
        )

class SemanticMemoryList(BaseModel):
    semantic_memories: list[SemanticMemoryRecord] = Field(
        description="List of semantic"
    )

# =========================================================
# Extraction Result
# =========================================================

class MemoryExtractionResult(BaseModel):
    """Structured output returned from LLM memory extraction."""

    profile_memories: list[ProfileMemoryRecord] = Field(
        default_factory=list,
        description="Extracted profile memories."
    )

    episodic_memories: list[EpisodicMemoryRecord] = Field(
        default_factory=list,
        description="Extracted episodic memories."
    )

    semantic_memories: list[SemanticMemoryRecord] = Field(
        default_factory=list,
        description="Extracted semantic memories."
    )

    rejected_candidates: list[str] = Field(
        default_factory=list,
        description="Rejected memory candidates."
    )

    notes: str = Field(
        default="",
        description="Additional extraction notes."
    )

    def to_patches(
        self,
        *,
        reason: str = "model structured memory extraction",
    ) -> tuple[MemoryPatch, ...]:

        patches: list[MemoryPatch] = []

        if self.profile_memories:
            patches.append(
                MemoryPatch(
                    target="profile",
                    reason=reason,
                    operations=tuple(
                        memory.to_operation()
                        for memory in self.profile_memories
                    ),
                )
            )

        if self.episodic_memories:
            patches.append(
                MemoryPatch(
                    target="episodic",
                    reason=reason,
                    operations=tuple(
                        memory.to_operation()
                        for memory in self.episodic_memories
                    ),
                )
            )

        if self.semantic_memories:
            patches.append(
                MemoryPatch(
                    target="semantic",
                    reason=reason,
                    operations=tuple(
                        memory.to_operation()
                        for memory in self.semantic_memories
                    ),
                )
            )

        return tuple(patches)
