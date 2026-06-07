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

    @property
    def dedup_key(self) -> str:
        """Unique key for deduplication and updates."""
        return self.key.strip().lower() if self.key else ""


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

    @property
    def dedup_key(self) -> str:
        """Unique key for deduplication and updates."""
        summary = self.summary.strip().lower() if getattr(self, "summary", None) else ""
        date_str = self.event_time[:10] if getattr(self, "event_time", None) else "anytime"
        return f"{summary}_{date_str}" if summary else ""


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

    @property
    def dedup_key(self) -> str:
        """Unique key for deduplication and updates."""
        subject = self.subject.strip().lower() if getattr(self, "subject", None) else ""
        predicate = getattr(self, "predicate", "")
        obj = self.object.strip().lower() if getattr(self, "object", None) else ""
        return f"{subject}:{predicate}:{obj}" if subject and obj else ""