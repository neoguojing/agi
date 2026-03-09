from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from ..core.types import MemoryEngine, MemorySearchResult


@dataclass(slots=True)
class MemoryStatus:
    primary_failed: bool = False
    last_error: str | None = None
    active_backend: str = "primary"


class InMemoryEngine:
    def __init__(self) -> None:
        self._docs: list[MemorySearchResult] = []

    def index(self, session_id: str, text: str, metadata: dict | None = None) -> None:
        self._docs.append(
            MemorySearchResult(
                content=text,
                source=f"session:{session_id}",
                metadata=metadata or {},
            )
        )

    def search(self, query: str, *, limit: int = 5) -> list[MemorySearchResult]:
        scored = []
        terms = query.lower().split()
        for item in self._docs:
            text = item.content.lower()
            score = sum(1 for term in terms if term in text)
            if score > 0:
                scored.append(
                    MemorySearchResult(
                        content=item.content,
                        source=item.source,
                        score=float(score),
                        metadata=item.metadata,
                    )
                )
        return sorted(scored, key=lambda x: x.score, reverse=True)[:limit]


class SessionMemoryEngine:
    def __init__(self) -> None:
        self._docs: dict[str, list[str]] = defaultdict(list)

    def index(self, session_id: str, text: str, metadata: dict | None = None) -> None:
        _ = metadata
        self._docs[session_id].append(text)

    def search(self, query: str, *, limit: int = 5) -> list[MemorySearchResult]:
        out: list[MemorySearchResult] = []
        for sid, docs in self._docs.items():
            for text in docs:
                if query.lower() in text.lower():
                    out.append(MemorySearchResult(content=text, source=f"fallback:{sid}", score=1.0))
                if len(out) >= limit:
                    return out
        return out


@dataclass(slots=True)
class MemorySearchManager:
    primary: MemoryEngine
    fallback: MemoryEngine
    status: MemoryStatus = field(default_factory=MemoryStatus)

    def index(self, session_id: str, text: str, metadata: dict | None = None) -> None:
        self.primary.index(session_id, text, metadata)
        self.fallback.index(session_id, text, metadata)

    def search(self, query: str, *, limit: int = 5) -> list[MemorySearchResult]:
        try:
            self.status.active_backend = "primary"
            return self.primary.search(query, limit=limit)
        except Exception as exc:  # noqa: BLE001
            self.status.primary_failed = True
            self.status.last_error = str(exc)
            self.status.active_backend = "fallback"
            return self.fallback.search(query, limit=limit)


def create_default_memory_manager() -> MemorySearchManager:
    return MemorySearchManager(primary=InMemoryEngine(), fallback=SessionMemoryEngine())
