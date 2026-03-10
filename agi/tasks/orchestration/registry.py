from __future__ import annotations

from threading import RLock
from typing import Any, Iterable

from agi.tasks.rag import rag_builtin_tools
from agi.tasks.simple_tools import simple_tools


class ToolSkillRegistry:
    """Runtime registry for tools and skill sources.

    Tools and skills are separated into builtin and external groups, both of
    which can be dynamically updated at runtime.
    """

    def __init__(self):
        self._lock = RLock()
        self._builtin_tools: list[Any] = [*simple_tools, *rag_builtin_tools]
        self._external_tools: list[Any] = []
        self._builtin_skills: list[str] = []
        self._external_skills: list[str] = []

    @staticmethod
    def _append_unique(items: list[Any], value: Any) -> None:
        if value not in items:
            items.append(value)

    @staticmethod
    def _extend_unique(items: list[Any], values: Iterable[Any]) -> None:
        for value in values:
            if value not in items:
                items.append(value)

    def register_tool(self, tool: Any, *, builtin: bool = False) -> None:
        with self._lock:
            target = self._builtin_tools if builtin else self._external_tools
            self._append_unique(target, tool)

    def unregister_tool(self, tool: Any, *, builtin: bool = False) -> None:
        with self._lock:
            target = self._builtin_tools if builtin else self._external_tools
            if tool in target:
                target.remove(tool)

    def register_skill(self, source: str, *, builtin: bool = False) -> None:
        with self._lock:
            target = self._builtin_skills if builtin else self._external_skills
            self._append_unique(target, source)

    def unregister_skill(self, source: str, *, builtin: bool = False) -> None:
        with self._lock:
            target = self._builtin_skills if builtin else self._external_skills
            if source in target:
                target.remove(source)

    def clear_external_tools(self) -> None:
        with self._lock:
            self._external_tools = []

    def clear_external_skills(self) -> None:
        with self._lock:
            self._external_skills = []

    def get_tools(
        self,
        *,
        include_builtin: bool = True,
        include_external: bool = True,
        extra_tools: Iterable[Any] | None = None,
    ) -> list[Any]:
        with self._lock:
            result: list[Any] = []
            if include_builtin:
                self._extend_unique(result, self._builtin_tools)
            if include_external:
                self._extend_unique(result, self._external_tools)
            if extra_tools:
                self._extend_unique(result, extra_tools)
            return result

    def get_skills(
        self,
        *,
        include_builtin: bool = True,
        include_external: bool = True,
        extra_skills: Iterable[str] | None = None,
    ) -> list[str] | None:
        with self._lock:
            result: list[str] = []
            if include_builtin:
                self._extend_unique(result, self._builtin_skills)
            if include_external:
                self._extend_unique(result, self._external_skills)
            if extra_skills:
                self._extend_unique(result, extra_skills)
            return result or None


_registry = ToolSkillRegistry()


def register_builtin_tool(tool: Any) -> None:
    _registry.register_tool(tool, builtin=True)


def register_external_tool(tool: Any) -> None:
    _registry.register_tool(tool, builtin=False)


def unregister_builtin_tool(tool: Any) -> None:
    _registry.unregister_tool(tool, builtin=True)


def unregister_external_tool(tool: Any) -> None:
    _registry.unregister_tool(tool, builtin=False)


def register_builtin_skill(source: str) -> None:
    _registry.register_skill(source, builtin=True)


def register_external_skill(source: str) -> None:
    _registry.register_skill(source, builtin=False)


def unregister_builtin_skill(source: str) -> None:
    _registry.unregister_skill(source, builtin=True)


def unregister_external_skill(source: str) -> None:
    _registry.unregister_skill(source, builtin=False)


def clear_external_tools() -> None:
    _registry.clear_external_tools()


def clear_external_skills() -> None:
    _registry.clear_external_skills()


def get_registered_tools(
    *,
    include_builtin: bool = True,
    include_external: bool = True,
    extra_tools: Iterable[Any] | None = None,
) -> list[Any]:
    return _registry.get_tools(
        include_builtin=include_builtin,
        include_external=include_external,
        extra_tools=extra_tools,
    )


def get_registered_skills(
    *,
    include_builtin: bool = True,
    include_external: bool = True,
    extra_skills: Iterable[str] | None = None,
) -> list[str] | None:
    return _registry.get_skills(
        include_builtin=include_builtin,
        include_external=include_external,
        extra_skills=extra_skills,
    )
