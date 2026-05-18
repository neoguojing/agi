"""Typed memory records and patch primitives.

Usage:
    These classes define the structured contract between:
    - LLM memory extraction
    - memory maintenance tasks
    - storage backends

    The schema is optimized for:
    - structured LLM output (Flat and Simple)
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
# Patch Layer (System Internal - Not for LLM Extraction)
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

    created_at: str | None = Field(
        default=None,
        description="Evidence creation timestamp (ISO string)."
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional evidence metadata."
    )


# =========================================================
# Profile Memory (Simplified for LLM)
# =========================================================

class ProfileMemoryRecord(BaseModel):
    """Stable long-term profile or preference memory."""

    id: str = Field(
        default="",
        description="Unique memory identifier."
    )

    key: str = Field(
        default="",
        description="Profile attribute key (e.g., 'favorite_color')."
    )

    value: str = Field(
        default="",
        description="Profile attribute value."
    )

    confidence: float = Field(
        default=0.0,
        description="Confidence score (0.0 to 1.0)."
    )

    importance: float = Field(
        default=0.0,
        description="Importance score (0.0 to 1.0)."
    )

    source: MemorySourceKind = Field(
        default="inferred",
        description="Memory source."
    )

    tags: list[str] = Field(
        default_factory=list,
        description="Categorization tags."
    )

    created_at: str | None = Field(
        default=None,
        description="Creation timestamp (ISO string)."
    )

    updated_at: str | None = Field(
        default=None,
        description="Last update timestamp (ISO string)."
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata."
    )

class ProfileMemoryList(BaseModel):
    profile_memories: list[ProfileMemoryRecord] = Field(
        description="List of profiles"
    )


# =========================================================
# Episodic Memory (Simplified for LLM)
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

    event_time: str | None = Field(
        default=None,
        description="Time when the event occurred (ISO string)."
    )

    participants: list[str] = Field(
        default_factory=list,
        description="Entities involved in the event."
    )

    outcome: str | None = Field(
        default=None,
        description="Event outcome."
    )

    confidence: float = Field(
        default=0.0,
        description="Confidence score (0.0 to 1.0)."
    )

    importance: float = Field(
        default=0.0,
        description="Importance score (0.0 to 1.0)."
    )

    ttl_days: int | None = Field(
        default=None,
        description="Memory TTL in days."
    )

    expires_at: str | None = Field(
        default=None,
        description="Expiration timestamp (ISO string)."
    )

    tags: list[str] = Field(
        default_factory=list,
        description="Categorization tags."
    )

    created_at: str | None = Field(
        default=None,
        description="Creation timestamp (ISO string)."
    )

    updated_at: str | None = Field(
        default=None,
        description="Last update timestamp (ISO string)."
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata."
    )

class EpisodicMemoryList(BaseModel):
    episodic_memories: list[EpisodicMemoryRecord] = Field(
        description="List of episodic"
    )

# =========================================================
# Semantic Memory (Simplified for LLM)
# =========================================================

class SemanticMemoryRecord(BaseModel):
    """Graph-ready semantic memory triple."""

    id: str = Field(
        default="",
        description="Unique semantic memory identifier."
    )

    subject: str = Field(
        default="",
        description="The entity the memory is about."
    )

    predicate: str = Field(
        default="",
        description="The relationship or property (e.g., 'works_at', 'is_a')."
    )

    object: str = Field(
        default="",
        description="The value or target entity of the relationship."
    )

    confidence: float = Field(
        default=0.0,
        description="Confidence score (0.0 to 1.0)."
    )

    importance: float = Field(
        default=0.0,
        description="Importance score (0.0 to 1.0)."
    )

    tags: list[str] = Field(
        default_factory=list,
        description="Categorization tags."
    )

    created_at: str | None = Field(
        default=None,
        description="Creation timestamp (ISO string)."
    )

    updated_at: str | None = Field(
        default=None,
        description="Last update timestamp (ISO string)."
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata."
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
        now_iso = datetime.now(timezone.utc).isoformat()

        def prepare_record(record: BaseModel) -> dict[str, Any]:
            """Helper to dump record and fill in missing system defaults."""
            data = record.model_dump(mode="json")
            
            # 1. Timestamps
            if not data.get("created_at"):
                data["created_at"] = now_iso
            if not data.get("updated_at"):
                data["updated_at"] = now_iso
            
            # 2. Quality Metrics (Default to 0.5 if 0.0 or missing, as 0.0 is often a default)
            if data.get("confidence") == 0.0:
                data["confidence"] = 0.5
            if data.get("importance") == 0.0:
                data["importance"] = 0.5
                
            # 3. Source
            if not data.get("source"):
                data["source"] = "conversation"
                
            # 4. Collections
            if data.get("tags") is None:
                data["tags"] = []
            if data.get("metadata") is None:
                data["metadata"] = {}
                
            return data

        if self.profile_memories:
            patches.append(
                MemoryPatch(
                    target="profile",
                    reason=reason,
                    operations=tuple(
                        MemoryOperation(
                            op="add",
                            target_id=m.id or None,
                            value=prepare_record(m)
                        )
                        for m in self.profile_memories
                    ),
                )
            )

        if self.episodic_memories:
            patches.append(
                MemoryPatch(
                    target="episodic",
                    reason=reason,
                    operations=tuple(
                        MemoryOperation(
                            op="add",
                            target_id=m.id or None,
                            value=prepare_record(m)
                        )
                        for m in self.episodic_memories
                    ),
                )
            )

        if self.semantic_memories:
            patches.append(
                MemoryPatch(
                    target="semantic",
                    reason=reason,
                    operations=tuple(
                        MemoryOperation(
                            op="add",
                            target_id=m.id or None,
                            value=prepare_record(m)
                        )
                        for m in self.semantic_memories
                    ),
                )
            )

        return tuple(patches)
