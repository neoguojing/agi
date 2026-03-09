from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, Sequence


@dataclass(slots=True)
class AgentRuntimeConfig:
    """统一的运行时配置。"""

    model: str | None = None
    system_prompt: str | None = None
    memory_sources: list[str] = field(default_factory=list)
    skill_sources: list[str] = field(default_factory=list)

    # deepagents backend mode: state/filesystem/local_shell/composite
    backend: str = "local_shell"
    backend_root_dir: str | None = "."
    backend_virtual_mode: bool = True
    backend_routes: dict[str, str] = field(default_factory=dict)
    backend_deny_prefixes: list[str] = field(default_factory=list)

    # long-term memory (CompositeBackend + StoreBackend)
    enable_long_term_memory: bool = True
    long_term_memory_prefix: str = "/memories/"
    auto_create_store_for_dev: bool = False

    use_default_subagents: bool = True
    subagents: list[dict[str, Any]] = field(default_factory=list)
    tools: list[Callable[..., Any]] = field(default_factory=list)

    enable_shell_tool: bool = True
    enable_sandbox_tool: bool = False
    enable_docker_toolchain: bool = False

    # harness-level options
    memory_files: list[str] = field(default_factory=list)
    interrupt_on: dict[str, bool | dict[str, Any]] = field(default_factory=dict)

    debug: bool = False


@dataclass(slots=True)
class ContextAssembleResult:
    messages: list[dict[str, Any]]
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
    def ingest(self, session_id: str, messages: Sequence[dict[str, Any]]) -> None:
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
