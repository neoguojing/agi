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

DEFAULT_LEGACY_MEMORY_PATHS: tuple[str, ...] = (
    "/memories/facts.md",
    "/memories/preferences.md",
    "/memories/lessons.md",
)

DEFAULT_MEMORY_TARGET_PATHS: dict[MemoryTarget, str] = {
    "profile": "/memories/profile.jsonl",
    "episodic": "/memories/episodic.jsonl",
    # Semantic memory is graph-ready by shape and does not need a separate
    # graph store until a future graph backend is introduced.
    "semantic": "/memories/semantic.jsonl",
}


@runtime_checkable
class MemoryStore(Protocol):
    """Minimal storage contract for memory maintenance."""

    legacy_paths: tuple[str, ...]
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
    """File-backed MemoryStore adapter over the existing BackendProtocol."""

    def __init__(
        self,
        backend: BackendProtocol,
        *,
        legacy_paths: Sequence[str] = DEFAULT_LEGACY_MEMORY_PATHS,
        target_paths: dict[MemoryTarget, str] | None = None,
    ) -> None:
        self.backend = backend
        self.legacy_paths = tuple(legacy_paths)
        self.target_paths = dict(target_paths or DEFAULT_MEMORY_TARGET_PATHS)

    def read_text(self, path: str) -> str:
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

    def load_legacy_memory(self) -> dict[str, str]:
        return {path: content for path in self.legacy_paths if (content := self.read_text(path))}

    def read_jsonl(self, path: str) -> list[dict[str, Any]]:
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
        content = "".join(f"{json.dumps(record, ensure_ascii=False, sort_keys=True)}\n" for record in records)
        self.write_text(path, content)

    def append_jsonl(self, path: str, records: Sequence[dict[str, Any]]) -> None:
        if not records:
            return
        existing = self.read_text(path)
        addition = "".join(f"{json.dumps(record, ensure_ascii=False, sort_keys=True)}\n" for record in records)
        separator = "" if not existing or existing.endswith("\n") else "\n"
        self.write_text(path, f"{existing}{separator}{addition}")

    def apply_patch(self, patch: MemoryPatch) -> None:
        if patch.is_empty:
            return

        path = patch.target_path or self.target_paths[patch.target]
        records = self.read_jsonl(path)
        changed = False

        for operation in patch.operations:
            if operation.op == "add":
                record = dict(operation.value)
                record.setdefault("created_at", patch.created_at.isoformat())
                record.setdefault("updated_at", patch.created_at.isoformat())
                records.append(record)
                changed = True
            elif operation.op in {"update", "merge"}:
                changed = self._update_record(records, operation, patch.created_at) or changed
            elif operation.op == "delete":
                before = len(records)
                records = [record for record in records if record.get("id") != operation.target_id]
                changed = changed or len(records) != before
            elif operation.op == "deprecate":
                changed = self._deprecate_record(records, operation, patch.created_at) or changed

        if changed:
            self.replace_jsonl(path, records)

    def _update_record(self, records: list[dict[str, Any]], operation: MemoryOperation, timestamp) -> bool:
        for record in records:
            if record.get("id") == operation.target_id:
                record.update(operation.value)
                record["updated_at"] = timestamp.isoformat()
                return True
        return False

    def _deprecate_record(self, records: list[dict[str, Any]], operation: MemoryOperation, timestamp) -> bool:
        for record in records:
            if record.get("id") == operation.target_id:
                record["deprecated_at"] = timestamp.isoformat()
                if operation.reason:
                    record["deprecated_reason"] = operation.reason
                return True
        return False


def _strip_line_numbers(content: str) -> str:
    """Best-effort conversion from backend read() output to raw text."""

    lines: list[str] = []
    for line in content.splitlines():
        if "\t" in line:
            prefix, value = line.split("\t", 1)
            if prefix.strip().replace(".", "", 1).isdigit():
                lines.append(value)
                continue
        lines.append(line)
    return "\n".join(lines)
