from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

TodoStatus = Literal["pending", "in_progress", "completed"]


@dataclass(slots=True)
class TodoItem:
    id: str
    content: str
    status: TodoStatus = "pending"


class TodoManager:
    """轻量计划能力：模拟 harness write_todos 的任务状态管理。"""

    def __init__(self) -> None:
        self._todos: dict[str, list[TodoItem]] = {}

    def write_todos(self, session_id: str, todos: list[dict[str, Any]]) -> list[TodoItem]:
        out: list[TodoItem] = []
        for raw in todos:
            item = TodoItem(
                id=str(raw.get("id", "")),
                content=str(raw.get("content", "")).strip(),
                status=raw.get("status", "pending"),
            )
            out.append(item)
        self._todos[session_id] = out
        return out

    def list_todos(self, session_id: str) -> list[TodoItem]:
        return list(self._todos.get(session_id, []))

    def update_status(self, session_id: str, todo_id: str, status: TodoStatus) -> bool:
        items = self._todos.get(session_id, [])
        for item in items:
            if item.id == todo_id:
                item.status = status
                return True
        return False
