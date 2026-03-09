from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..core.types import AgentRuntimeConfig


@dataclass(slots=True)
class BackendBuildResult:
    backend: Any
    description: str
    requires_store: bool = False


class BackendFactoryError(ValueError):
    pass


class PolicyWrapper:
    """通用后端策略包装：按前缀禁写/禁编辑。"""

    def __init__(self, inner: Any, deny_prefixes: list[str] | None = None) -> None:
        self.inner = inner
        self.deny_prefixes = [p if p.endswith("/") else f"{p}/" for p in (deny_prefixes or [])]

    def _deny(self, path: str) -> bool:
        return any(path.startswith(prefix) for prefix in self.deny_prefixes)

    def __getattr__(self, item: str) -> Any:
        return getattr(self.inner, item)

    def write(self, file_path: str, content: str):
        if self._deny(file_path):
            return {"error": f"Writes are not allowed under {file_path}"}
        return self.inner.write(file_path, content)

    def edit(self, file_path: str, old_string: str, new_string: str, replace_all: bool = False):
        if self._deny(file_path):
            return {"error": f"Edits are not allowed under {file_path}"}
        return self.inner.edit(file_path, old_string, new_string, replace_all)


def build_backend(config: AgentRuntimeConfig, dep: dict[str, Any]) -> BackendBuildResult:
    mode = config.backend
    fs = dep.get("FilesystemBackend")
    shell = dep.get("LocalShellBackend")
    state = dep.get("StateBackend")
    store_backend = dep.get("StoreBackend")
    composite = dep.get("CompositeBackend")

    if mode == "local_shell":
        backend = shell(root_dir=config.backend_root_dir or ".")
        desc = "LocalShellBackend(root_dir=...)"
        requires_store = False
    elif mode == "filesystem":
        backend = fs(root_dir=config.backend_root_dir or ".", virtual_mode=config.backend_virtual_mode)
        desc = "FilesystemBackend(root_dir=..., virtual_mode=True)"
        requires_store = False
    elif mode == "state":
        backend = state
        desc = "StateBackend(runtime)"
        requires_store = False
    elif mode == "composite":
        if not composite:
            raise BackendFactoryError("CompositeBackend is not available in deepagents package")

        prefix = config.long_term_memory_prefix if config.long_term_memory_prefix.endswith("/") else f"{config.long_term_memory_prefix}/"
        routes = config.backend_routes or ({prefix: "store"} if config.enable_long_term_memory else {})
        requires_store = any(v == "store" for v in routes.values())

        def _factory(rt):
            route_map: dict[str, Any] = {}
            for route_prefix, backend_name in routes.items():
                if backend_name == "filesystem":
                    route_map[route_prefix] = fs(root_dir=config.backend_root_dir or ".", virtual_mode=config.backend_virtual_mode)
                elif backend_name == "state":
                    route_map[route_prefix] = state(rt)
                elif backend_name == "local_shell":
                    route_map[route_prefix] = shell(root_dir=config.backend_root_dir or ".")
                elif backend_name == "store":
                    route_map[route_prefix] = store_backend(rt)
                else:
                    raise BackendFactoryError(f"Unsupported route backend: {backend_name}")
            return composite(default=state(rt), routes=route_map)

        backend = _factory
        desc = "CompositeBackend(default=StateBackend, routes=...; recommended /memories/ -> StoreBackend)"
    else:
        raise BackendFactoryError(f"Unknown backend mode: {mode}")

    if config.backend_deny_prefixes and mode in {"filesystem", "local_shell"}:
        backend = PolicyWrapper(backend, deny_prefixes=config.backend_deny_prefixes)
        desc += " + PolicyWrapper(deny_prefixes=...)"

    return BackendBuildResult(backend=backend, description=desc, requires_store=requires_store)
