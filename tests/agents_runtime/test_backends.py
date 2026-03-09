from agi.agents_runtime.backend_factory import BackendFactoryError, build_backend
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


class FakeComposite:
    def __init__(self, default, routes):
        self.default = default
        self.routes = routes


def test_build_filesystem_backend_with_virtual_mode():
    cfg = AgentRuntimeConfig(backend="filesystem", backend_root_dir="/tmp/work", backend_virtual_mode=True)
    dep = {
        "FilesystemBackend": FakeFS,
        "LocalShellBackend": FakeShell,
        "StateBackend": FakeState,
        "CompositeBackend": FakeComposite,
    }

    result = build_backend(cfg, dep)
    assert isinstance(result.backend, FakeFS)
    assert result.backend.virtual_mode is True


def test_build_composite_backend_returns_factory():
    cfg = AgentRuntimeConfig(backend="composite", backend_routes={"/memories/": "filesystem"})
    dep = {
        "FilesystemBackend": FakeFS,
        "LocalShellBackend": FakeShell,
        "StateBackend": FakeState,
        "CompositeBackend": FakeComposite,
    }

    result = build_backend(cfg, dep)
    factory = result.backend
    out = factory(object())
    assert isinstance(out, FakeComposite)
    assert "/memories/" in out.routes


def test_unknown_backend_mode_raises():
    cfg = AgentRuntimeConfig(backend="unknown")
    dep = {
        "FilesystemBackend": FakeFS,
        "LocalShellBackend": FakeShell,
        "StateBackend": FakeState,
        "CompositeBackend": FakeComposite,
    }

    try:
        build_backend(cfg, dep)
        raised = False
    except BackendFactoryError:
        raised = True

    assert raised is True
