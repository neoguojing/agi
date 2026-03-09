from agi.agents_runtime.context_engine import LegacyContextEngine
from agi.agents_runtime.memory_engine import MemorySearchManager, SessionMemoryEngine


class BrokenEngine:
    def index(self, session_id: str, text: str, metadata=None):
        return None

    def search(self, query: str, *, limit: int = 5):
        raise RuntimeError("boom")


def test_legacy_context_engine_compact_and_assemble():
    engine = LegacyContextEngine()
    sid = "s1"

    for i in range(30):
        engine.ingest(sid, [{"role": "user", "content": f"m-{i}"}])

    assembled = engine.assemble(sid, token_budget=10)
    assert len(assembled.messages) == 10
    assert assembled.metadata["trimmed"] is True

    compact = engine.compact(sid, reason="test")
    assert compact.changed is True


def test_memory_search_manager_fallback():
    manager = MemorySearchManager(primary=BrokenEngine(), fallback=SessionMemoryEngine())
    manager.index("a", "hello deep agent")

    out = manager.search("deep")
    assert out
    assert manager.status.primary_failed is True
    assert manager.status.active_backend == "fallback"
