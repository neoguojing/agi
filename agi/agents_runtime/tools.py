from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass(slots=True)
class ToolSpec:
    name: str
    description: str
    handler: Callable[..., Any]
    modality_tags: set[str] = field(default_factory=set)
    safety_class: str = "normal"


class ToolRegistry:
    """统一工具注册/权限筛选，后续可接策略流水线。"""

    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        self._tools[spec.name] = spec

    def get(self, name: str) -> ToolSpec:
        return self._tools[name]

    def list_tools(self, *, tags: set[str] | None = None, allow_safety: set[str] | None = None) -> list[ToolSpec]:
        items = list(self._tools.values())
        if tags:
            items = [item for item in items if item.modality_tags.intersection(tags)]
        if allow_safety:
            items = [item for item in items if item.safety_class in allow_safety]
        return items

    def to_deepagents_tools(self, *, tags: set[str] | None = None, allow_safety: set[str] | None = None) -> list[Callable[..., Any]]:
        return [spec.handler for spec in self.list_tools(tags=tags, allow_safety=allow_safety)]
