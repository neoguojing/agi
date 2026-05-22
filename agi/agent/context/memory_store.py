"""Memory persistence protocols and backend adapter.

Usage:
    `BackendMemoryStore` wraps the existing DeepAgents `BackendProtocol` and
    exposes memory-oriented helpers: legacy markdown reads, JSONL reads/writes,
    and `MemoryPatch` application.

    Example:
        store = BackendMemoryStore(backend)
        legacy = store.load_legacy_memory()
        store.apply_patch(patch)

This module should remain storage-focused. It should not call an LLM or decide
which memories to create; those decisions live in extraction/tasks modules.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any, Protocol, Sequence, runtime_checkable

from agi.agent.context.memory_models import MemoryOperation, MemoryPatch, MemoryTarget

if TYPE_CHECKING:
    from deepagents.backends.protocol import BackendProtocol
else:
    BackendProtocol = Any

logger = logging.getLogger(__name__)

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
    """Minimal storage contract for memory maintenance.
    
    Defines the required interface for any store that manages memory 
    persistence, whether it's file-based, database-based, or in-memory.
    """

    target_paths: dict[MemoryTarget, str]

    def read_text(self, path: str) -> str:
        """Read raw text from a memory path, returning an empty string if absent."""
        ...

    def write_text(self, path: str, content: str) -> None:
        """Create or replace raw text at a memory path."""
        ...

    def load_legacy_memory(self) -> dict[str, str]:
        """Read existing markdown memory files for backward compatibility."""
        ...

    def read_jsonl(self, path: str) -> list[dict[str, Any]]:
        """Read a JSONL memory file into dictionaries."""
        ...

    def replace_jsonl(self, path: str, records: Sequence[dict[str, Any]]) -> None:
        """Replace a JSONL memory file with the supplied records."""
        ...

    def append_jsonl(self, path: str, records: Sequence[dict[str, Any]]) -> None:
        """Append records to a JSONL memory file."""
        ...

    def apply_patch(self, patch: MemoryPatch) -> None:
        """Apply a storage-neutral memory patch."""
        ...


class BackendMemoryStore:
    """File-backed MemoryStore adapter over the existing BackendProtocol.
    
    This class translates high-level memory operations (like applying a patch) 
    into low-level backend file operations (read, write, edit).
    """

    def __init__(
        self,
        backend: BackendProtocol,
        *,
        target_paths: dict[MemoryTarget, str] | None = None,
    ) -> None:
        self.backend = backend
        self.target_paths = dict(target_paths or DEFAULT_MEMORY_TARGET_PATHS)

    def read_text(self, path: str) -> str:
        """Reads raw text from the backend, handling potential errors and stripping line numbers."""
        try:
            responses = self.backend.download_files([path])
            if responses and responses[0].content is not None and responses[0].error is None:
                return responses[0].content.decode("utf-8")
            if responses and responses[0].error == "file_not_found":
                return ""
        except (AttributeError, NotImplementedError):
            pass
        except Exception as exc:  # noqa: BLE001 - backend-specific errors should not break reads
            logger.debug("Raw memory download failed for %s: %s", path, exc)

        try:
            content = self.backend.read(path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to read memory path %s: %s", path, exc)
            return ""

        if not content or content.startswith("Error:") or content.startswith("System reminder:"):
            return ""
        return _strip_line_numbers(content)

    def write_text(self, path: str, content: str) -> None:
        """Writes raw text to the backend, using upload or edit depending on existence."""
        encoded = content.encode("utf-8")
        try:
            responses = self.backend.upload_files([(path, encoded)])
            if responses and responses[0].error is None:
                return
        except (AttributeError, NotImplementedError):
            pass

        existing = self.read_text(path)
        if existing:
            result = self.backend.edit(path, existing, content)
            if getattr(result, "error", None):
                raise RuntimeError(result.error)
            return

        result = self.backend.write(path, content)
        if getattr(result, "error", None):
            raise RuntimeError(result.error)

    def read_jsonl(self, path: str) -> list[dict[str, Any]]:
        """Reads a file and parses each line as a JSON object."""
        content = self.read_text(path)
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
        return records

    def replace_jsonl(self, path: str, records: Sequence[dict[str, Any]]) -> None:
        """Overwrites a file with a new set of JSONL records."""
        content = "".join(f"{json.dumps(record, ensure_ascii=False, sort_keys=True)}\n" for record in records)
        self.write_text(path, content)

    def append_jsonl(self, path: str, records: Sequence[dict[str, Any]]) -> None:
        """Appends new JSONL records to an existing file."""
        if not records:
            return
        existing = self.read_text(path)
        addition = "".join(f"{json.dumps(record, ensure_ascii=False, sort_keys=True)}\n" for record in records)
        separator = "" if not existing or existing.endswith("\n") else "\n"
        self.write_text(path, f"{existing}{separator}{addition}")

    def apply_patch(self, patch: MemoryPatch) -> None:
        """
        Applies a MemoryPatch to the store.
        
        Reads the current records for the target, iterates through the operations 
        (add, update, delete, deprecate), and writes the updated list back to the backend.
        """
        if patch.is_empty:
            return

        path = patch.target_path or self.target_paths[patch.target]
        records = self.read_jsonl(path)

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
            self.replace_jsonl(path, incoming)
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
            self.replace_jsonl(path, records)

    def _add_or_merge_record(self, records: list[dict[str, Any]], target: MemoryTarget, record: dict[str, Any], timestamp) -> bool:
        """Add record unless a semantic duplicate already exists; merge when duplicate found."""
        candidate = _record_dedup_key(target, record)
        if not candidate:
            records.append(record)
            return True

        for existing in records:
            if _record_dedup_key(target, existing) != candidate:
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
            key = _record_dedup_key(target, record)
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


def _normalize_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(value.strip().lower().split())


def _normalize_participants(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    normalized = [_normalize_text(v) for v in value if isinstance(v, str) and _normalize_text(v)]
    return tuple(sorted(set(normalized)))


def _record_dedup_key(target: MemoryTarget, record: dict[str, Any]) -> tuple[Any, ...] | None:
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
        # Event time often drifts; ignore it for matching to reduce repeated
        # extraction duplicates for the same event.
        return ("episodic", summary, participants, event_time[:10] if event_time else "")

    if target == "semantic":
        subject = _normalize_text(record.get("subject"))
        predicate = _normalize_text(record.get("predicate"))
        obj = _normalize_text(record.get("object"))
        return ("semantic", subject, predicate, obj) if subject and predicate and obj else None

    return None
