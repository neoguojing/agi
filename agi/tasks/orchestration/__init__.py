from __future__ import annotations

from .registry import (
    clear_external_skills,
    clear_external_tools,
    get_registered_skills,
    get_registered_tools,
    register_builtin_skill,
    register_builtin_tool,
    register_external_skill,
    register_external_tool,
    unregister_builtin_skill,
    unregister_builtin_tool,
    unregister_external_skill,
    unregister_external_tool,
)
from agi.tasks.session import make_session_backend, resolve_session_components


def build_main_agent(*args, **kwargs):
    """Lazy import to avoid heavy deepagents dependency at package import time."""
    from .deepagent_builder import build_main_agent as _build_main_agent

    return _build_main_agent(*args, **kwargs)


__all__ = [
    "build_main_agent",
    "register_builtin_tool",
    "register_external_tool",
    "unregister_builtin_tool",
    "unregister_external_tool",
    "register_builtin_skill",
    "register_external_skill",
    "unregister_builtin_skill",
    "unregister_external_skill",
    "clear_external_tools",
    "clear_external_skills",
    "get_registered_tools",
    "get_registered_skills",
    "make_session_backend",
    "resolve_session_components",
]
