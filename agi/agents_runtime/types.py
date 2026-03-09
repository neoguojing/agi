from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence


@dataclass(slots=True)
class AgentRuntimeConfig:
    """统一的运行时配置。"""

    model: str | None = None
    system_prompt: str | None = None
    memory_sources: list[str] = field(default_factory=list)
    skill_sources: list[str] = field(default_factory=list)
    backend: str = "local_shell"
    debug: bool = False


@dataclass(slots=True)
class ContextAssembleResult:
    messages: list[dict[str, str]]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ContextCompactResult:
    changed: bool
    reason: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class MemorySearchResult:
    content: str
    source: str
    score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


class ContextEngine(Protocol):
    def ingest(self, session_id: str, messages: Sequence[dict[str, str]]) -> None:
        ...

    def assemble(self, session_id: str, *, token_budget: int | None = None) -> ContextAssembleResult:
        ...

    def compact(self, session_id: str, *, reason: str) -> ContextCompactResult:
        ...


class MemoryEngine(Protocol):
    def index(self, session_id: str, text: str, metadata: dict[str, Any] | None = None) -> None:
        ...

    def search(self, query: str, *, limit: int = 5) -> list[MemorySearchResult]:
        ...
