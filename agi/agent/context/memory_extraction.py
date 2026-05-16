"""LLM-facing memory extraction schema, prompt, and parser.

Usage:
    1. Build a prompt with `build_memory_extraction_prompt(...)`.
    2. Send that prompt to an LLM using JSON/structured output mode.
    3. Parse the LLM JSON with `parse_memory_extraction_result(...)`.
    4. Convert the typed result to patches via `result.to_patches()`.

This module deliberately does not write memory. It only shapes LLM output so
maintenance tasks can review and persist patches through `MemoryStore`.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, is_dataclass
from datetime import datetime
from typing import Any

from agi.agent.context.memory_models import (
    EpisodicMemoryRecord,
    MemoryEvidence,
    MemoryExtractionResult,
    MemorySourceKind,
    ProfileMemoryRecord,
    SemanticEntity,
    SemanticMemoryRecord,
    SemanticObject,
)

logger = logging.getLogger(__name__)

MEMORY_EXTRACTION_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "profile_memories": {
            "type": "array",
            "description": "Stable user profile, preferences, identity, or durable settings.",
            "items": {
                "type": "object",
                "required": ["key", "value", "confidence"],
                "properties": {
                    "id": {"type": "string"},
                    "key": {"type": "string", "description": "Stable dotted key, e.g. communication.language."},
                    "value": {"description": "JSON value to store for the profile key."},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "importance": {"type": "number", "minimum": 0, "maximum": 1},
                    "source": {"type": "string", "enum": ["user_explicit", "conversation", "inferred", "legacy_memory", "system", "tool"]},
                    "evidence": {"type": "array", "items": {"$ref": "#/$defs/evidence"}},
                    "tags": {"type": "array", "items": {"type": "string"}},
                    "metadata": {"type": "object"},
                },
            },
        },
        "episodic_memories": {
            "type": "array",
            "description": "Time-bound events, experiences, or lessons that may decay.",
            "items": {
                "type": "object",
                "required": ["summary", "confidence"],
                "properties": {
                    "id": {"type": "string"},
                    "summary": {"type": "string"},
                    "event_time": {"type": "string", "format": "date-time"},
                    "participants": {"type": "array", "items": {"type": "string"}},
                    "outcome": {"type": "string"},
                    "context": {"type": "object"},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "importance": {"type": "number", "minimum": 0, "maximum": 1},
                    "ttl_days": {"type": "integer"},
                    "expires_at": {"type": "string", "format": "date-time"},
                    "evidence": {"type": "array", "items": {"$ref": "#/$defs/evidence"}},
                    "tags": {"type": "array", "items": {"type": "string"}},
                    "metadata": {"type": "object"},
                },
            },
        },
        "semantic_memories": {
            "type": "array",
            "description": "Long-term graph-ready knowledge as subject-predicate-object records.",
            "items": {
                "type": "object",
                "required": ["subject", "predicate", "object", "confidence"],
                "properties": {
                    "id": {"type": "string"},
                    "subject": {"$ref": "#/$defs/semantic_entity"},
                    "predicate": {"type": "string"},
                    "object": {"$ref": "#/$defs/semantic_object"},
                    "qualifiers": {"type": "object"},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "importance": {"type": "number", "minimum": 0, "maximum": 1},
                    "evidence": {"type": "array", "items": {"$ref": "#/$defs/evidence"}},
                    "tags": {"type": "array", "items": {"type": "string"}},
                    "metadata": {"type": "object"},
                },
            },
        },
        "rejected_candidates": {"type": "array", "items": {"type": "string"}},
        "notes": {"type": "string"},
    },
    "$defs": {
        "evidence": {
            "type": "object",
            "properties": {
                "source": {"type": "string", "enum": ["user_explicit", "conversation", "inferred", "legacy_memory", "system", "tool"]},
                "content": {"type": "string"},
                "message_id": {"type": "string"},
                "memory_id": {"type": "string"},
                "created_at": {"type": "string", "format": "date-time"},
                "metadata": {"type": "object"},
            },
        },
        "semantic_entity": {
            "type": "object",
            "required": ["id"],
            "properties": {
                "id": {"type": "string"},
                "kind": {"type": "string"},
                "label": {"type": "string"},
                "properties": {"type": "object"},
            },
        },
        "semantic_object": {
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "kind": {"type": "string"},
                "value": {},
                "label": {"type": "string"},
                "properties": {"type": "object"},
            },
        },
    },
}

MEMORY_EXTRACTION_INSTRUCTIONS = """Extract memory candidates from the conversation and return ONLY JSON matching MEMORY_EXTRACTION_JSON_SCHEMA.
Rules:
- Use profile_memories only for stable user identity, preferences, durable settings, and long-lived facts.
- Use episodic_memories for time-bound events, lessons, task outcomes, or experiences that can decay.
- Use semantic_memories for long-term abstract knowledge as graph-ready subject/predicate/object records.
- Include evidence for every accepted memory.
- Reject transient, low-value, or ambiguous candidates in rejected_candidates instead of forcing a memory.
- Do not write files directly; the caller converts this structured result into MemoryPatch objects.
"""


def build_memory_extraction_prompt(*, conversation: str, existing_memory: str = "") -> str:
    """Build an LLM-facing prompt for structured memory extraction."""

    return (
        f"{MEMORY_EXTRACTION_INSTRUCTIONS}\n\n"
        f"MEMORY_EXTRACTION_JSON_SCHEMA:\n{json.dumps(MEMORY_EXTRACTION_JSON_SCHEMA, ensure_ascii=False, indent=2)}\n\n"
        f"EXISTING_MEMORY:\n{existing_memory or '(none)'}\n\n"
        f"CONVERSATION:\n{conversation}\n"
    )


def parse_memory_extraction_result(payload: str | dict[str, Any]) -> MemoryExtractionResult:
    """Parse JSON returned by an LLM into typed memory extraction records."""

    data = json.loads(payload) if isinstance(payload, str) else payload
    if not isinstance(data, dict):
        raise ValueError("Memory extraction payload must be a JSON object")

    return MemoryExtractionResult(
        profile_memories=tuple(_coerce_profile_memory(item) for item in _as_list(data.get("profile_memories"))),
        episodic_memories=tuple(_coerce_episodic_memory(item) for item in _as_list(data.get("episodic_memories"))),
        semantic_memories=tuple(_coerce_semantic_memory(item) for item in _as_list(data.get("semantic_memories"))),
        rejected_candidates=tuple(str(item) for item in _as_list(data.get("rejected_candidates"))),
        notes=str(data.get("notes") or ""),
    )


def _coerce_profile_memory(value: Any) -> ProfileMemoryRecord:
    data = _as_dict(value)
    return ProfileMemoryRecord(
        id=str(data.get("id") or ""),
        key=str(data.get("key") or ""),
        value=data.get("value"),
        confidence=_as_float(data.get("confidence")),
        importance=_as_float(data.get("importance")),
        source=_as_source(data.get("source"), default="inferred"),
        evidence=tuple(_coerce_evidence(item) for item in _as_list(data.get("evidence"))),
        tags=tuple(str(item) for item in _as_list(data.get("tags"))),
        created_at=_as_datetime(data.get("created_at")),
        updated_at=_as_datetime(data.get("updated_at")),
        metadata=_as_dict(data.get("metadata")),
    )


def _coerce_episodic_memory(value: Any) -> EpisodicMemoryRecord:
    data = _as_dict(value)
    ttl_days = data.get("ttl_days")
    return EpisodicMemoryRecord(
        id=str(data.get("id") or ""),
        summary=str(data.get("summary") or ""),
        event_time=_as_datetime(data.get("event_time")),
        participants=tuple(str(item) for item in _as_list(data.get("participants"))),
        outcome=str(data["outcome"]) if data.get("outcome") is not None else None,
        context=_as_dict(data.get("context")),
        confidence=_as_float(data.get("confidence")),
        importance=_as_float(data.get("importance")),
        ttl_days=int(ttl_days) if ttl_days is not None else None,
        expires_at=_as_datetime(data.get("expires_at")),
        evidence=tuple(_coerce_evidence(item) for item in _as_list(data.get("evidence"))),
        tags=tuple(str(item) for item in _as_list(data.get("tags"))),
        created_at=_as_datetime(data.get("created_at")),
        updated_at=_as_datetime(data.get("updated_at")),
        metadata=_as_dict(data.get("metadata")),
    )


def _coerce_semantic_memory(value: Any) -> SemanticMemoryRecord:
    data = _as_dict(value)
    return SemanticMemoryRecord(
        id=str(data.get("id") or ""),
        subject=_coerce_semantic_entity(data.get("subject")),
        predicate=str(data.get("predicate") or ""),
        object=_coerce_semantic_object(data.get("object")),
        qualifiers=_as_dict(data.get("qualifiers")),
        confidence=_as_float(data.get("confidence")),
        importance=_as_float(data.get("importance")),
        evidence=tuple(_coerce_evidence(item) for item in _as_list(data.get("evidence"))),
        tags=tuple(str(item) for item in _as_list(data.get("tags"))),
        created_at=_as_datetime(data.get("created_at")),
        updated_at=_as_datetime(data.get("updated_at")),
        metadata=_as_dict(data.get("metadata")),
    )


def _coerce_evidence(value: Any) -> MemoryEvidence:
    data = _as_dict(value)
    return MemoryEvidence(
        source=_as_source(data.get("source"), default="conversation"),
        content=str(data.get("content") or ""),
        message_id=str(data["message_id"]) if data.get("message_id") is not None else None,
        memory_id=str(data["memory_id"]) if data.get("memory_id") is not None else None,
        created_at=_as_datetime(data.get("created_at")),
        metadata=_as_dict(data.get("metadata")),
    )


def _coerce_semantic_entity(value: Any) -> SemanticEntity:
    data = _as_dict(value)
    return SemanticEntity(
        id=str(data.get("id") or ""),
        kind=str(data.get("kind") or "concept"),
        label=str(data["label"]) if data.get("label") is not None else None,
        properties=_as_dict(data.get("properties")),
    )


def _coerce_semantic_object(value: Any) -> SemanticObject:
    data = _as_dict(value)
    return SemanticObject(
        id=str(data["id"]) if data.get("id") is not None else None,
        kind=str(data.get("kind") or "value"),
        value=data.get("value"),
        label=str(data["label"]) if data.get("label") is not None else None,
        properties=_as_dict(data.get("properties")),
    )


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _as_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, dict):
        return value
    return {}


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_datetime(value: Any) -> datetime | None:
    if value is None or isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            logger.warning("Skipping invalid memory timestamp: %s", value)
    return None


def _as_source(value: Any, *, default: MemorySourceKind) -> MemorySourceKind:
    allowed = {"user_explicit", "conversation", "inferred", "legacy_memory", "system", "tool"}
    if isinstance(value, str) and value in allowed:
        return value  # type: ignore[return-value]
    return default
