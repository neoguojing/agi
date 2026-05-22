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

class ProfileMemoryList(BaseModel):
    """
    Structured long-term profile memory representing stable
    user attributes, preferences, habits, identities,
    settings, skills, or persistent personal information.

    Purpose:
    - Capture stable user characteristics
    - Preserve long-term preferences and identity traits
    - Store reusable personalization information
    - Support future personalization and memory retrieval

    Suitable Memory Types:
    - Preferences
    - Personal settings
    - Long-term goals
    - Skills and expertise
    - Roles and occupations
    - Frequently repeated behaviors
    - Stable relationships
    - Persistent environment information

    Good Examples:
    - "favorite_language" -> "Python"
    - "job_title" -> "Software Engineer"
    - "preferred_database" -> "ClickHouse"
    - "timezone" -> "Asia/Tokyo"
    - "communication_style" -> "concise"

    Bad Examples:
    - "User attended a meeting yesterday"
        -> episodic memory

    - "Python is a programming language"
        -> semantic memory

    - Temporary short-lived states
        -> should not be stored as profile memory

    Extraction Guidelines for LLM:
    - Extract ONLY stable long-term information
    - Avoid temporary conversational details
    - Prefer normalized concise keys
    - Prefer atomic key-value pairs
    - Each memory should contain ONLY ONE fact
    - Do not merge unrelated attributes together

    Key Naming Rules:
    - Use concise snake_case keys
    - Keep keys stable and reusable
    - Avoid natural language sentences

    Good Keys:
    - favorite_language
    - job_title
    - preferred_editor
    - timezone

    Bad Keys:
    - user_really_likes_programming_languages
    - the_user_currently_works_as

    Field Rules:
    - ALL fields are REQUIRED
    - ALL string fields MUST be non-empty
    - confidence MUST be between 0.0 and 1.0
    - do NOT generate placeholder values
    - do NOT generate empty strings
    """
    items: list[ProfileMemoryRecord] = Field(
        description="List of profiles"
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

class EpisodicMemoryList(BaseModel):
    """
    Structured episodic memory representing a specific event,
    activity, interaction, or experience that occurred at a
    particular time.

    Purpose:
    - Capture time-bound experiences and interactions
    - Preserve conversational events as retrievable memories
    - Store meaningful user activities, milestones, decisions,
      meetings, plans, achievements, or incidents
    - Support timeline reconstruction and temporal reasoning

    Extraction Guidelines for LLM:
    - Extract ONLY concrete events or experiences
    - Each memory should represent ONE atomic event
    - The event should be meaningful and retrievable later
    - Avoid vague or generic summaries
    - Avoid duplicating semantic/profile memories
    - Prefer concise factual summaries

    Good Examples:
    - "User started a new job at OpenAI"
    - "User traveled to Tokyo for a conference"
    - "User completed migration from Cassandra to ClickHouse"
    - "User discussed long-term memory architecture design"

    Bad Examples:
    - "User likes Python"                -> profile memory
    - "Python is a programming language" -> semantic memory
    - "User talked about something"      -> too vague

    Field Rules:
    - ALL fields are REQUIRED
    - ALL string fields MUST be non-empty
    - participants list MUST NOT be empty
    - participants items MUST NOT be empty
    - confidence MUST be between 0.0 and 1.0
    - event_time MUST use ISO datetime string format
    - do NOT generate placeholder values
    - do NOT generate empty strings

    Time Rules:
    - Use the actual event occurrence time when available
    - If exact time is unknown, infer the best approximate time
    - Always use ISO-8601 datetime format    
    """
    items: list[EpisodicMemoryRecord] = Field(
        description="List of episodic"
    )

# =========================================================
# Semantic Memory (Simplified for LLM)
# =========================================================

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
            "REQUIRED. Source entity of the relationship. "
            "Example: 'Alice', 'Python', 'OpenAI'."
        )
    )

    predicate: str = Field(
        ...,
        min_length=1,
        description=(
            "REQUIRED. Normalized relationship type. "
            "Use short graph-friendly predicates such as "
            "'works_at', 'likes', 'uses', 'located_in'."
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

class SemanticMemoryList(BaseModel):
    """
    Structured semantic relationship memory.

    Purpose:
    - Extract stable factual relationships
    - Represent knowledge as semantic triples
    - Keep relationships atomic and graph-friendly

    Extraction Rules for LLM:
    - ALL fields are REQUIRED
    - ALL string fields MUST be non-empty
    - subject MUST be a concrete entity
    - predicate MUST be a short normalized relation
    - object MUST be a concrete value or target entity
    - confidence MUST be between 0.0 and 1.0
    - use concise normalized predicates:
        GOOD: works_at, likes, lives_in, uses
        BAD: "is currently working at"
    - each memory should contain ONLY ONE fact
    - do NOT generate placeholder values
    - do NOT generate empty strings
    """
    items: list[SemanticMemoryRecord] = Field(
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
                
            # 3. Source
            if not data.get("source"):
                data["source"] = "conversation"
                
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
