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
from uuid import uuid4
from pydantic import BaseModel, Field


# =========================================================
# Normalization Helpers (shared by dedup in tasks and store)
# =========================================================


def _normalize_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(value.strip().lower().split())


def _normalize_participants(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    normalized = [_normalize_text(v) for v in value if isinstance(v, str) and _normalize_text(v)]
    return tuple(sorted(set(normalized)))


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
# Dedup
# =========================================================


def record_dedup_key(target: "MemoryTarget", record: dict[str, Any]) -> tuple[Any, ...] | None:
    """Best-effort semantic key used to collapse duplicate memories."""
    if target == "profile":
        key = _normalize_text(record.get("key"))
        value = _normalize_text(record.get("value"))
        return ("profile", key, value) if key and value else None

    if target == "episodic":
        summary = _normalize_text(record.get("summary"))
        participants = _normalize_participants(record.get("participants"))
        event_time = _normalize_text(record.get("event_time"))
        if not summary:
            return None
        return ("episodic", summary, participants, event_time[:10] if event_time else "")

    if target == "semantic":
        subject = _normalize_text(record.get("subject"))
        predicate = _normalize_text(record.get("predicate"))
        obj = _normalize_text(record.get("object"))
        return ("semantic", subject, predicate, obj) if subject and predicate and obj else None

    return None


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

    strategy: Literal["merge", "replace"] = Field(
        default="merge",
        description="How to apply operations: merge into existing records or replace collection."
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
# Profile Memory (Simplified for LLM)
# =========================================================

class ProfileMemoryRecord(BaseModel):

    id: str = Field(
        default_factory=lambda: str(uuid4()),
        description=(
            "System-generated unique memory identifier."
        )
    )

    key: str = Field(
        ...,
        min_length=1,
        description=(
            "REQUIRED. Profile attribute key "
            "(example: 'favorite_language', 'job_title'). "
            "Must not be empty."
        )
    )

    value: str = Field(
        ...,
        min_length=1,
        description=(
            "REQUIRED. Profile attribute value. "
            "Must not be empty."
        )
    )

    confidence: float = Field(
        default=0.5,
        description=(
            "OPTION. Confidence score between "
            "0.0 and 1.0."
        )
    )

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="creation timestamp."
    )

    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="update timestamp."
    )

class ProfileMemoryList(BaseModel):
    """Stable user profile attributes as key-value pairs."""
    items: list[ProfileMemoryRecord] = Field(
        description="Extracted profile memories — stable user attributes, preferences, skills."
    )


# =========================================================
# Episodic Memory (Simplified for LLM)
# =========================================================

class EpisodicMemoryRecord(BaseModel):

    id: str = Field(
        default_factory=lambda: str(uuid4()),
        min_length=1,
        description=(
            "REQUIRED. Unique episodic memory identifier. "
            "Must not be empty."
        )
    )

    summary: str = Field(
        ...,
        min_length=1,
        description=(
            "REQUIRED. Short summary of the event. "
            "Must not be empty."
        )
    )

    event_time: str = Field(
        default="",
        description=(
            "Option. Event occurrence time "
            "in ISO datetime string format. "
            "Must not be empty."
        )
    )

    participants: list[str] = Field(
        ...,
        min_length=1,
        description=(
            "REQUIRED. List of entities or people "
            "involved in the event. "
            "List must not be empty."
        )
    )

    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "REQUIRED. Confidence score "
            "between 0.0 and 1.0."
        )
    )

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="creation timestamp."
    )

    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="update timestamp."
    )

class EpisodicMemoryList(BaseModel):
    """Time-bound events and experiences as structured records."""
    items: list[EpisodicMemoryRecord] = Field(
        description="Extracted episodic memories — concrete events, milestones, decisions."
    )

# =========================================================
# Semantic Memory (Simplified for LLM)
# =========================================================

AgentMemoryPredicate = Literal[
    "is_a",
    "same_as",
    "likes",
    "prefers",
    "uses",
    "owns",
    "knows",
    "member_of",
    "works_at",
    "created",
    "located_in",
    "learned",
    "depends_on",
    "related_to",
]

class SemanticMemoryRecord(BaseModel):

    id: str = Field(
        default_factory=lambda: str(uuid4()),
        min_length=1,
        description=(
            "REQUIRED. Unique semantic memory identifier. "
            "Must not be empty."
        )
    )

    subject: str = Field(
        ...,
        min_length=1,
        description=(
            "REQUIRED. Canonical subject entity name. "
            "Must be a short noun-style entity, not a sentence. "
            "Use concise reusable names such as "
            "'OpenAI', 'Python', 'Tokyo', 'FastAPI'. "
            "Avoid pronouns, full sentences, and excessive details."
        )
    )

    predicate: AgentMemoryPredicate = Field(
        ...,
        description=(
            "REQUIRED. Target entity or value of the relationship. "
            "Prefer short canonical entity names or concise literal values. "
            "Examples: 'OpenAI', 'Python', 'Tokyo', 'backend_development'. "
            "Avoid long natural language sentences."
        )
    )

    object: str = Field(
        ...,
        min_length=1,
        description=(
            "REQUIRED. Target entity or value of the relationship."
        )
    )

    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "REQUIRED. Confidence score "
            "between 0.0 and 1.0."
        )
    )

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="creation timestamp."
    )

    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="update timestamp."
    )

class SemanticMemoryList(BaseModel):
    """Factual knowledge as subject-predicate-object triples."""
    items: list[SemanticMemoryRecord] = Field(
        description="Extracted semantic memories — factual triples, graph-ready knowledge."
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
