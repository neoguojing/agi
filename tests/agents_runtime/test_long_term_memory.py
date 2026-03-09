from agi.agents_runtime.backend_factory import build_backend
from agi.agents_runtime.session_context import SessionContextManager
from agi.agents_runtime.types import AgentRuntimeConfig


class FakeFS:
    def __init__(self, root_dir=".", virtual_mode=False):
        self.root_dir = root_dir
        self.virtual_mode = virtual_mode


class FakeShell:
    def __init__(self, root_dir="."):
        self.root_dir = root_dir


class FakeState:
    def __init__(self, rt=None):
        self.rt = rt


class FakeStore:
    def __init__(self, rt=None):
        self.rt = rt


class FakeComposite:
    def __init__(self, default, routes):
        self.default = default
        self.routes = routes


def test_composite_backend_routes_memories_to_store():
    cfg = AgentRuntimeConfig(
        backend="composite",
        enable_long_term_memory=True,
        long_term_memory_prefix="/memories/",
    )
    dep = {
        "FilesystemBackend": FakeFS,
        "LocalShellBackend": FakeShell,
        "StateBackend": FakeState,
        "StoreBackend": FakeStore,
        "CompositeBackend": FakeComposite,
    }

    result = build_backend(cfg, dep)
    assert result.requires_store is True

    backend = result.backend(object())
    assert "/memories/" in backend.routes
    assert isinstance(backend.routes["/memories/"], FakeStore)


def test_session_context_manager_thread_and_memory_path():
    manager = SessionContextManager()
    state = manager.get_or_create("s-1")

    config = manager.to_run_config(state, extra_context={"user_id": "u-1"})
    assert config["configurable"]["thread_id"] == "s-1"
    assert config["context"]["user_id"] == "u-1"

    assert manager.memory_path("research", "notes.txt") == "/memories/research/notes.txt"
