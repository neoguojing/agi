"""Memory persistence protocols and backend adapter.

Usage:
    `BackendMemoryStore` wraps the existing DeepAgents `BackendProtocol` and
    exposes async, target-oriented memory helpers: legacy text reads, JSONL
    reads/writes, and `MemoryPatch` application.

    Example:
        store = BackendMemoryStore(backend)
        legacy = await store.read_text("profile")
        await store.apply_patch(patch)

This module should remain storage-focused. It should not call an LLM or decide
which memories to create; those decisions live in extraction/tasks modules.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any, Protocol, Sequence, runtime_checkable

from agi.agent.context.memory_models import (
    MemoryOperation,
    MemoryPatch,
    MemoryTarget,
    record_dedup_key,
)
if TYPE_CHECKING:
    from deepagents.backends.protocol import BackendProtocol
else:
    BackendProtocol = Any

logger = logging.getLogger(__name__)

EPISODIC_RETENTION_DAYS = 30
EPISODIC_MAX_RECORDS = 20
SEMANTIC_RETENTION_DAYS = 180
SEMANTIC_MAX_RECORDS = 100

# Default filesystem paths for the different memory types.
DEFAULT_MEMORY_TARGET_PATHS: dict[MemoryTarget, str] = {
    "profile": "/memories/profile.jsonl",
    "episodic": "/memories/episodic.jsonl",
    # Semantic memory is graph-ready by shape and does not need a separate
    # graph store until a future graph backend is introduced.
    "semantic": "/memories/semantic.jsonl",
}


@runtime_checkable
class MemoryStore(Protocol):
    """Async storage contract for memory maintenance.

    Public methods are target-oriented: callers select a `MemoryTarget`, and the
    store resolves the backing path internally. This keeps path knowledge in the
    storage adapter instead of spreading direct path operations across memory
    extraction, scheduling, or middleware code.
    """

    target_paths: dict[MemoryTarget, str]

    def path_for_target(self, target: MemoryTarget) -> str:
        """Return the backing path configured for a memory target."""
        ...

    async def read_text(self, target: MemoryTarget) -> str:
        """Read raw text for a memory target, returning an empty string if absent."""
        ...

    async def write_text(self, target: MemoryTarget, content: str) -> None:
        """Create or replace raw text for a memory target."""
        ...

    async def read_jsonl(self, target: MemoryTarget) -> list[dict[str, Any]]:
        """Read a target JSONL memory collection into dictionaries."""
        ...

    async def replace_jsonl(self, target: MemoryTarget, records: Sequence[dict[str, Any]]) -> None:
        """Replace a target JSONL memory collection with the supplied records."""
        ...

    async def append_jsonl(self, target: MemoryTarget, records: Sequence[dict[str, Any]]) -> None:
        """Append records to a target JSONL memory collection."""
        ...

    async def apply_patch(self, patch: MemoryPatch) -> None:
        """Apply a storage-neutral memory patch."""
        ...


class BackendMemoryStore:
    """File-backed MemoryStore adapter over the existing BackendProtocol.

    This class translates target-level memory operations (like applying a patch)
    into low-level backend file operations (read, write, edit). Direct path
    access is intentionally kept private to this adapter.
    """

    def __init__(
        self,
        backend: BackendProtocol,
        *,
        target_paths: dict[MemoryTarget, str] | None = None,
    ) -> None:
        self.backend = backend
        self.target_paths = dict(target_paths or DEFAULT_MEMORY_TARGET_PATHS)

    def path_for_target(self, target: MemoryTarget) -> str:
        """Return the configured backend path for a memory target."""
        return self.target_paths[target]

    async def read_text(self, target: MemoryTarget) -> str:
        """Read raw text for the target, handling errors and line-number stripping."""
        path = self.path_for_target(target)
        return await self._read_path(path)

    async def write_text(self, target: MemoryTarget, content: str) -> None:
        """Write raw text for the target, using upload or edit depending on existence."""
        path = self.path_for_target(target)
        await self._write_path(path, content)

    async def read_jsonl(self, target: MemoryTarget) -> list[dict[str, Any]]:
        """Read a target memory collection and parse each line as a JSON object."""
        path = self.path_for_target(target)
        content = await self.read_text(target)
        records: list[dict[str, Any]] = []
        for line_number, line in enumerate(content.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning("Skipping invalid JSONL record in %s:%d: %s", path, line_number, exc)
                continue
            if isinstance(value, dict):
                records.append(value)
            else:
                logger.warning("Skipping non-object JSONL record in %s:%d", path, line_number)

        records, _ = self._apply_retention(records, target, datetime.now(timezone.utc))
        return records

    async def replace_jsonl(self, target: MemoryTarget, records: Sequence[dict[str, Any]]) -> None:
        """Overwrite a target memory collection with a new set of JSONL records."""
        content = "".join(f"{json.dumps(record, ensure_ascii=False, sort_keys=True)}\n" for record in records)
        await self.write_text(target, content)

    async def append_jsonl(self, target: MemoryTarget, records: Sequence[dict[str, Any]]) -> None:
        """Append new JSONL records to a target memory collection."""
        if not records:
            return
        existing = await self.read_text(target)
        addition = "".join(f"{json.dumps(record, ensure_ascii=False, sort_keys=True)}\n" for record in records)
        separator = "" if not existing or existing.endswith("\n") else "\n"
        await self.write_text(target, f"{existing}{separator}{addition}")

    async def apply_patch(self, patch: MemoryPatch) -> None:
        """Apply a MemoryPatch to the target collection selected by the patch."""
        if patch.is_empty:
            return

        records = await self.read_jsonl(patch.target)

        if patch.strategy == "replace":
            incoming: list[dict[str, Any]] = []
            for operation in patch.operations:
                if operation.op != "add":
                    continue
                record = dict(operation.value)
                record.setdefault("created_at", patch.created_at.isoformat())
                record.setdefault("updated_at", patch.created_at.isoformat())
                incoming.append(record)

            incoming, _ = self._deduplicate_records(incoming, patch.target, patch.created_at)
            await self.replace_jsonl(patch.target, incoming)
            return

        changed = False

        for operation in patch.operations:
            if operation.op == "add":
                record = dict(operation.value)
                record.setdefault("created_at", patch.created_at.isoformat())
                record.setdefault("updated_at", patch.created_at.isoformat())
                changed = self._add_or_merge_record(records, patch.target, record, patch.created_at) or changed
            elif operation.op in {"update", "merge"}:
                changed = self._update_record(records, operation, patch.created_at) or changed
            elif operation.op == "delete":
                before = len(records)
                records = [record for record in records if record.get("id") != operation.target_id]
                changed = changed or len(records) != before
            elif operation.op == "deprecate":
                changed = self._deprecate_record(records, operation, patch.created_at) or changed

        records, dedup_changed = self._deduplicate_records(records, patch.target, patch.created_at)
        changed = changed or dedup_changed

        if changed:
            await self.replace_jsonl(patch.target, records)

    async def _read_path(self, path: str) -> str:
        try:
            responses = await self.backend.adownload_files([path])
            if responses and responses[0].content is not None and responses[0].error is None:
                return responses[0].content.decode("utf-8")
            if responses and responses[0].error == "file_not_found":
                return ""
        except (AttributeError, NotImplementedError):
            pass
        except Exception as exc:  # noqa: BLE001 - backend-specific errors should not break reads
            logger.debug("Raw memory download failed for %s: %s", path, exc)

        try:
            content = await self.backend.aread(path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to read memory path %s: %s", path, exc)
            return ""

        if not content or content.startswith("Error:") or content.startswith("System reminder:"):
            return ""
        return _strip_line_numbers(content)

    async def _write_path(self, path: str, content: str) -> None:
        encoded = content.encode("utf-8")
        try:
            responses = await self.backend.aupload_files([(path, encoded)])
            if responses and responses[0].error is None:
                return
        except (AttributeError, NotImplementedError):
            pass

        existing = await self._read_path(path)
        if existing:
            result = await self.backend.aedit(path, existing, content)
            if getattr(result, "error", None):
                raise RuntimeError(result.error)
            return

        result = await self.backend.awrite(path, content)
        if getattr(result, "error", None):
            raise RuntimeError(result.error)

    def _add_or_merge_record(self, records: list[dict[str, Any]], target: MemoryTarget, record: dict[str, Any], timestamp) -> bool:
        """Add record unless a semantic duplicate already exists; merge when duplicate found."""
        candidate = record_dedup_key(target, record)
        if not candidate:
            records.append(record)
            return True

        for existing in records:
            if record_dedup_key(target, existing) != candidate:
                continue
            # Keep existing id/created_at; refresh mutable fields to reduce stale duplicates.
            for key, value in record.items():
                if key in {"id", "created_at"}:
                    continue
                existing[key] = value
            existing["updated_at"] = timestamp.isoformat()
            return True

        records.append(record)
        return True

    def _update_record(self, records: list[dict[str, Any]], operation: MemoryOperation, timestamp) -> bool:
        """Internal helper to find and update a specific record by ID."""
        for record in records:
            if record.get("id") == operation.target_id:
                record.update(operation.value)
                record["updated_at"] = timestamp.isoformat()
                return True
        return False

    def _deprecate_record(self, records: list[dict[str, Any]], operation: MemoryOperation, timestamp) -> bool:
        """Internal helper to mark a record as deprecated without deleting it."""
        for record in records:
            if record.get("id") == operation.target_id:
                record["deprecated_at"] = timestamp.isoformat()
                if operation.reason:
                    record["deprecated_reason"] = operation.reason
                return True
        return False

    def _deduplicate_records(self, records: list[dict[str, Any]], target: MemoryTarget, timestamp) -> tuple[list[dict[str, Any]], bool]:
        """Compact an entire memory collection by semantic keys.

        This cleans up historical duplicates that were already persisted before
        de-dup-on-add was introduced.
        """
        deduped: list[dict[str, Any]] = []
        index_by_key: dict[tuple[Any, ...], int] = {}
        changed = False

        for record in records:
            key = record_dedup_key(target, record)
            if not key:
                deduped.append(record)
                continue

            existing_idx = index_by_key.get(key)
            if existing_idx is None:
                index_by_key[key] = len(deduped)
                deduped.append(record)
                continue

            existing = deduped[existing_idx]
            # Merge duplicate into the first-seen record.
            for field, value in record.items():
                if field in {"id", "created_at"}:
                    continue
                existing[field] = value
            existing["updated_at"] = timestamp.isoformat()
            changed = True

        if len(deduped) != len(records):
            changed = True

        return deduped, changed

    def _apply_retention(self, records: list[dict[str, Any]], target: MemoryTarget, timestamp) -> tuple[list[dict[str, Any]], bool]:
        if target == "profile":
            return records, False

        if target == "episodic":
            cutoff = timestamp - timedelta(days=EPISODIC_RETENTION_DAYS)
            limited = _filter_by_cutoff(records, cutoff)
            sorted_records = sorted(limited, key=_record_sort_key, reverse=True)[:EPISODIC_MAX_RECORDS]
            return sorted_records, len(sorted_records) != len(records)

        if target == "semantic":
            cutoff = timestamp - timedelta(days=SEMANTIC_RETENTION_DAYS)
            limited = _filter_by_cutoff(records, cutoff)
            sorted_records = sorted(limited, key=_record_sort_key, reverse=True)[:SEMANTIC_MAX_RECORDS]
            return sorted_records, len(sorted_records) != len(records)

        return records, False

def _strip_line_numbers(content: str) -> str:
    """Best-effort conversion from backend read() output to raw text.
    
    Removes line number prefixes (e.g., '  1\tContent') that some backends 
    add to their output.
    """

    lines: list[str] = []
    for line in content.splitlines():
        if "\t" in line:
            prefix, value = line.split("\t", 1)
            if prefix.strip().replace(".", "", 1).isdigit():
                lines.append(value)
                continue
        lines.append(line)
    return "\n".join(lines)

def _parse_iso_datetime(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _record_sort_key(record: dict[str, Any]) -> datetime:
    return (
        _parse_iso_datetime(record.get("updated_at"))
        or _parse_iso_datetime(record.get("created_at"))
        or _parse_iso_datetime(record.get("event_time"))
        or datetime.min.replace(tzinfo=timezone.utc)
    )


def _filter_by_cutoff(records: list[dict[str, Any]], cutoff: datetime) -> list[dict[str, Any]]:
    return [record for record in records if _record_sort_key(record) >= cutoff]
