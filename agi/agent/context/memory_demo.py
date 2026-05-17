"""Demo usage for the memory maintenance abstractions.

This module intentionally does not touch MemoryMiddleware. It demonstrates the
intended layering:

1. A MemoryStore owns persistence.
2. A MemoryTask decides what patch to emit.
3. MemoryTaskScheduler decides which independently configured tasks are due.
4. The caller applies MemoryPatch objects through the store.
"""

from __future__ import annotations

import asyncio
import importlib.util
import sys
import types
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class WriteResult:
    error: str | None = None
    path: str | None = None


@dataclass
class EditResult:
    error: str | None = None
    path: str | None = None
    occurrences: int | None = None


@dataclass
class FileUploadResponse:
    path: str
    error: str | None = None


@dataclass
class FileDownloadResponse:
    path: str
    content: bytes | None = None
    error: str | None = None

if __package__:
    from agi.agent.context.memory import (
        BackendMemoryStore,
        MemoryTaskConfig,
        MemoryTaskContext,
        MemoryTaskResult,
        MemoryTaskScheduler,
        MemoryTaskScheduleState,
        build_memory_extraction_prompt,
        parse_memory_extraction_result,
    )
else:
    # Allow `python agi/agent/context/memory_demo.py` in minimal environments
    # where importing agi.agent.context.__init__ would require optional deps.
    context_dir = Path(__file__).parent

    for package_name, package_path in (
        ("agi", context_dir.parent.parent),
        ("agi.agent", context_dir.parent),
        ("agi.agent.context", context_dir),
    ):
        package = sys.modules.get(package_name) or types.ModuleType(package_name)
        package.__path__ = [str(package_path)]
        sys.modules[package_name] = package

    def _load_context_module(module_name: str):
        module_path = context_dir / f"{module_name}.py"
        qualified_name = f"agi.agent.context.{module_name}"
        spec = importlib.util.spec_from_file_location(qualified_name, module_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Unable to load {qualified_name} from {module_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[qualified_name] = module
        spec.loader.exec_module(module)
        return module

    _load_context_module("memory_models")
    memory_extraction = _load_context_module("memory_extraction")
    memory_store = _load_context_module("memory_store")
    memory_tasks = _load_context_module("memory_tasks")

    BackendMemoryStore = memory_store.BackendMemoryStore
    MemoryTaskConfig = memory_tasks.MemoryTaskConfig
    MemoryTaskContext = memory_tasks.MemoryTaskContext
    MemoryTaskResult = memory_tasks.MemoryTaskResult
    MemoryTaskScheduler = memory_tasks.MemoryTaskScheduler
    MemoryTaskScheduleState = memory_tasks.MemoryTaskScheduleState
    build_memory_extraction_prompt = memory_extraction.build_memory_extraction_prompt
    parse_memory_extraction_result = memory_extraction.parse_memory_extraction_result


class InMemoryBackend:
    """Tiny backend used only by this demo."""

    def __init__(self) -> None:
        self.files: dict[str, str] = {}

    def read(self, file_path: str, offset: int = 0, limit: int = 2000) -> str:
        content = self.files.get(file_path)
        if content is None:
            return f"Error: File '{file_path}' not found"
        lines = content.splitlines()
        selected = lines[offset : offset + limit]
        return "\n".join(f"{i + offset + 1:6d}\t{line}" for i, line in enumerate(selected))

    def write(self, file_path: str, content: str) -> WriteResult:
        if file_path in self.files:
            return WriteResult(error=f"Cannot write to {file_path} because it already exists.")
        self.files[file_path] = content
        return WriteResult(path=file_path)

    def edit(self, file_path: str, old_string: str, new_string: str, replace_all: bool = False) -> EditResult:
        existing = self.files.get(file_path)
        if existing is None:
            return EditResult(error=f"Error: File '{file_path}' not found")
        if old_string not in existing:
            return EditResult(error="Error: String not found")
        self.files[file_path] = existing.replace(old_string, new_string, -1 if replace_all else 1)
        return EditResult(path=file_path, occurrences=1)

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        for path, content in files:
            self.files[path] = content.decode("utf-8")
        return [FileUploadResponse(path=path, error=None) for path, _ in files]

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        responses: list[FileDownloadResponse] = []
        for path in paths:
            if path not in self.files:
                responses.append(FileDownloadResponse(path=path, error="file_not_found"))
            else:
                responses.append(FileDownloadResponse(path=path, content=self.files[path].encode("utf-8")))
        return responses


class DemoSemanticTask:
    """Example graph-ready semantic task with an independent schedule."""

    name = "demo_semantic_consolidation"
    target = "semantic"

    def __init__(self) -> None:
        self.config = MemoryTaskConfig(enabled=True, interval_seconds=24 * 3600, min_confidence=0.65)

    def should_run(self, context: MemoryTaskContext) -> bool:
        return True

    async def run(self, context: MemoryTaskContext) -> MemoryTaskResult:
        # In production this prompt would be sent to an LLM with structured output.
        # The simulated JSON below demonstrates the expected LLM return shape.
        _prompt = build_memory_extraction_prompt(
            conversation="User prefers structured memory abstractions.",
        )
        simulated_llm_json = {
            "semantic_memories": [
                {
                    "id": "sem_demo_user_prefers_structured_memory",
                    "subject": {"id": "user", "kind": "person"},
                    "predicate": "prefers_memory_design",
                    "object": {"value": "structured_graph_ready_records", "kind": "preference"},
                    "qualifiers": {"scope": "agent_memory"},
                    "evidence": [{"source": "legacy_memory"}],
                    "confidence": 0.8,
                }
            ]
        }
        extraction = parse_memory_extraction_result(simulated_llm_json)
        patches = extraction.to_patches(reason="demo graph-ready semantic extraction")
        return MemoryTaskResult(
            task_name=self.name,
            changed=bool(patches),
            summary="Parsed one LLM extraction result and emitted semantic patches.",
            patches=patches,
            metadata={"prompt_preview": _prompt[:120]},
        )


async def run_demo() -> dict[str, Any]:
    backend = InMemoryBackend()
    backend.upload_files([
        ("/memories/facts.md", "User prefers structured memory abstractions.\n".encode("utf-8")),
    ])

    store = BackendMemoryStore(backend)
    context = MemoryTaskContext(
        store=store,
        backend=backend,  # type: ignore[arg-type]
        now=datetime.now(timezone.utc),
    )

    scheduler = MemoryTaskScheduler()
    state = MemoryTaskScheduleState()
    results, state = await scheduler.run_due_tasks([DemoSemanticTask()], context, state)

    return {
        "results": [result.summary for result in results],
        "schedule_state": state.to_iso_dict(),
        "semantic_records": store.read_jsonl("/memories/semantic.jsonl"),
    }


if __name__ == "__main__":
    print(asyncio.run(run_demo()))
