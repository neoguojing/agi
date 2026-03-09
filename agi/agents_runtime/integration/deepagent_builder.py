from __future__ import annotations

from typing import Any

from .backend_factory import build_backend
from ..core.subagents import build_default_subagents
from ..sandbox.system_tools import (
    docker_build_toolchain,
    execute_shell,
    run_in_sandbox,
    sandbox_download_file,
    sandbox_execute,
    sandbox_shutdown,
    sandbox_upload_file,
)
from ..core.types import AgentRuntimeConfig


class DeepAgentUnavailableError(RuntimeError):
    """DeepAgents 未安装或不可用。"""


def _import_deepagents() -> dict[str, Any]:
    try:
        from deepagents import create_deep_agent
        from deepagents.backends import CompositeBackend, FilesystemBackend, LocalShellBackend, StateBackend, StoreBackend
        from deepagents.middleware import MemoryMiddleware, SkillsMiddleware
    except Exception as exc:  # noqa: BLE001
        raise DeepAgentUnavailableError(
            "deepagents is required for the new runtime framework. "
            "Please install dependency before running."
        ) from exc

    return {
        "create_deep_agent": create_deep_agent,
        "FilesystemBackend": FilesystemBackend,
        "LocalShellBackend": LocalShellBackend,
        "StateBackend": StateBackend,
        "CompositeBackend": CompositeBackend,
        "StoreBackend": StoreBackend,
        "MemoryMiddleware": MemoryMiddleware,
        "SkillsMiddleware": SkillsMiddleware,
    }


def _sandbox_code_tools(config: AgentRuntimeConfig) -> list[Any]:
    tools: list[Any] = []
    if config.enable_sandbox_tool:
        tools.extend([sandbox_execute, sandbox_upload_file, sandbox_download_file, sandbox_shutdown])
    if config.enable_docker_toolchain:
        tools.append(docker_build_toolchain)
    return tools


def _merge_subagents(config: AgentRuntimeConfig) -> list[dict[str, Any]]:
    subagents = list(config.subagents)
    if config.use_default_subagents:
        subagents.extend(
            build_default_subagents(
                model=config.model,
                skill_sources=config.skill_sources,
                code_tools=_sandbox_code_tools(config),
            )
        )
    return subagents


def _build_tools(config: AgentRuntimeConfig) -> list[Any]:
    tools = list(config.tools)
    if config.enable_shell_tool:
        tools.append(execute_shell)
    if config.enable_sandbox_tool:
        tools.append(run_in_sandbox)
    if config.enable_docker_toolchain:
        tools.append(docker_build_toolchain)
    return tools


def _build_runtime_extras(config: AgentRuntimeConfig, backend_requires_store: bool) -> dict[str, Any]:
    extra: dict[str, Any] = {}
    need_checkpointer = backend_requires_store or bool(config.interrupt_on)

    if need_checkpointer:
        try:
            from langgraph.checkpoint.memory import MemorySaver

            extra["checkpointer"] = MemorySaver()
        except Exception:
            pass

    if backend_requires_store and config.auto_create_store_for_dev:
        try:
            from langgraph.store.memory import InMemoryStore

            extra["store"] = InMemoryStore()
        except Exception:
            pass

    return extra


def build_deep_agent(config: AgentRuntimeConfig):
    dep = _import_deepagents()

    # lazy import to avoid importing agi.config when users only use lightweight modules
    from agi.config import OLLAMA_DEFAULT_MODE

    backend_result = build_backend(config, dep)
    middleware = []
    if config.memory_sources:
        middleware.append(dep["MemoryMiddleware"](backend=backend_result.backend, sources=config.memory_sources))
    if config.skill_sources:
        middleware.append(dep["SkillsMiddleware"](backend=backend_result.backend, sources=config.skill_sources))

    return dep["create_deep_agent"](
        model=config.model or OLLAMA_DEFAULT_MODE,
        tools=_build_tools(config),
        system_prompt=config.system_prompt,
        middleware=middleware,
        subagents=_merge_subagents(config),
        skills=config.skill_sources or None,
        memory=config.memory_files or None,
        interrupt_on=config.interrupt_on or None,
        backend=backend_result.backend,
        debug=config.debug,
        name="agi-main-agent",
        **_build_runtime_extras(config, backend_result.requires_store),
    )
