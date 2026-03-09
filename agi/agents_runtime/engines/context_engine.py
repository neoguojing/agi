from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Callable

from ..core.types import ContextAssembleResult, ContextCompactResult, ContextEngine


@dataclass(slots=True)
class EngineSpec:
    id: str
    factory: Callable[[], ContextEngine]


class LegacyContextEngine:
    """兼容旧流程的上下文引擎：按会话保存消息并裁剪。"""

    def __init__(self) -> None:
        self._sessions: dict[str, list[dict[str, object]]] = defaultdict(list)

    def ingest(self, session_id: str, messages: list[dict[str, object]]) -> None:
        self._sessions[session_id].extend(messages)

    def assemble(self, session_id: str, *, token_budget: int | None = None) -> ContextAssembleResult:
        msgs = self._sessions.get(session_id, [])
        if token_budget is None or token_budget <= 0:
            return ContextAssembleResult(messages=list(msgs), metadata={"trimmed": False})

        trimmed = msgs[-token_budget:]
        return ContextAssembleResult(
            messages=trimmed,
            metadata={"trimmed": len(trimmed) != len(msgs), "budget": token_budget},
        )

    def compact(self, session_id: str, *, reason: str) -> ContextCompactResult:
        msgs = self._sessions.get(session_id, [])
        if len(msgs) <= 20:
            return ContextCompactResult(changed=False, reason=reason, details={"size": len(msgs)})

        self._sessions[session_id] = msgs[-20:]
        return ContextCompactResult(changed=True, reason=reason, details={"size": 20})


class ContextEngineRegistry:
    def __init__(self) -> None:
        self._factories: dict[str, Callable[[], ContextEngine]] = {}

    def register(self, spec: EngineSpec) -> None:
        self._factories[spec.id] = spec.factory

    def resolve(self, engine_id: str) -> ContextEngine:
        if engine_id not in self._factories:
            raise KeyError(f"Unknown context engine: {engine_id}")
        return self._factories[engine_id]()


_registry = ContextEngineRegistry()
_initialized = False


def ensure_context_engines_initialized() -> None:
    global _initialized
    if _initialized:
        return
    _registry.register(EngineSpec(id="legacy", factory=LegacyContextEngine))
    _initialized = True


def get_context_engine(engine_id: str = "legacy") -> ContextEngine:
    ensure_context_engines_initialized()
    return _registry.resolve(engine_id)
