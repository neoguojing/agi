from __future__ import annotations

from typing import Any

from .types import AgentRuntimeConfig


class DeepAgentUnavailableError(RuntimeError):
    """DeepAgents 未安装或不可用。"""


def _import_deepagents() -> dict[str, Any]:
    try:
        from deepagents import create_deep_agent
        from deepagents.backends import FilesystemBackend, LocalShellBackend
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
        "MemoryMiddleware": MemoryMiddleware,
        "SkillsMiddleware": SkillsMiddleware,
    }


def build_deep_agent(config: AgentRuntimeConfig):
    dep = _import_deepagents()

    # lazy import to avoid importing agi.config when users only use lightweight modules
    from agi.config import OLLAMA_DEFAULT_MODE

    backend_type = dep["LocalShellBackend"] if config.backend == "local_shell" else dep["FilesystemBackend"]
    middleware = []
    if config.memory_sources:
        middleware.append(dep["MemoryMiddleware"](backend=backend_type, sources=config.memory_sources))
    if config.skill_sources:
        middleware.append(dep["SkillsMiddleware"](backend=backend_type, sources=config.skill_sources))

    return dep["create_deep_agent"](
        model=config.model or OLLAMA_DEFAULT_MODE,
        system_prompt=config.system_prompt,
        middleware=middleware,
        debug=config.debug,
    )
