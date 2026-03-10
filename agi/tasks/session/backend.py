from __future__ import annotations

from typing import Any

from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore

from agi.config import AGI_LONG_TERM_MEMORY_ENABLED, AGI_MEMORY_PATH_PREFIX


def make_store_namespace(ctx: Any) -> tuple[str, ...]:
    runtime = ctx.runtime
    configurable = runtime.config.get("configurable", {})
    runtime_context = getattr(runtime, "context", None)

    tenant_id = configurable.get("tenant_id") or getattr(runtime_context, "tenant_id", None)
    assistant_id = configurable.get("assistant_id") or getattr(runtime_context, "assistant_id", None) or "deepagent"
    user_id = configurable.get("user_id") or getattr(runtime_context, "user_id", None) or "default"

    return (str(tenant_id or "global"), str(assistant_id), "filesystem", str(user_id))


def make_session_backend(runtime: Any):
    from agi.deepagents.backends import CompositeBackend, StateBackend, StoreBackend

    memory_prefix = AGI_MEMORY_PATH_PREFIX
    if not memory_prefix.endswith("/"):
        memory_prefix = memory_prefix + "/"

    return CompositeBackend(
        default=StateBackend(runtime),
        routes={memory_prefix: StoreBackend(runtime, namespace=make_store_namespace)},
    )


def default_store() -> BaseStore:
    return InMemoryStore()


def default_checkpointer() -> MemorySaver:
    return MemorySaver()


def resolve_session_components(
    *,
    backend: Any = None,
    store: BaseStore | None = None,
    checkpointer: Any = None,
    enable_long_term_memory: bool | None = None,
) -> tuple[Any, BaseStore | None, Any]:
    enabled = AGI_LONG_TERM_MEMORY_ENABLED if enable_long_term_memory is None else enable_long_term_memory

    resolved_backend = backend
    resolved_store = store
    resolved_checkpointer = checkpointer

    if enabled and resolved_backend is None:
        resolved_backend = make_session_backend

    if enabled and resolved_store is None:
        resolved_store = default_store()

    if resolved_checkpointer is None:
        resolved_checkpointer = default_checkpointer()

    return resolved_backend, resolved_store, resolved_checkpointer
