from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass(slots=True)
class SessionState:
    session_id: str
    thread_id: str
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    turn_count: int = 0
    last_active_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def touch(self) -> None:
        self.turn_count += 1
        self.last_active_at = datetime.now(timezone.utc).isoformat()


class SessionContextManager:
    """会话上下文管理：维护 session -> thread 映射，兼容跨线程长期记忆调用。"""

    def __init__(self) -> None:
        self._sessions: dict[str, SessionState] = {}

    def get_or_create(self, session_id: str, *, thread_id: str | None = None) -> SessionState:
        if session_id not in self._sessions:
            tid = thread_id or session_id
            self._sessions[session_id] = SessionState(session_id=session_id, thread_id=tid)
        state = self._sessions[session_id]
        state.touch()
        return state

    @staticmethod
    def memory_path(*segments: str) -> str:
        cleaned = [seg.strip("/") for seg in segments if seg and seg.strip("/")]
        suffix = "/".join(cleaned)
        return f"/memories/{suffix}" if suffix else "/memories/"

    @staticmethod
    def to_run_config(state: SessionState, *, extra_context: dict[str, Any] | None = None) -> dict[str, Any]:
        context = {
            "session_id": state.session_id,
            "thread_id": state.thread_id,
            "turn_count": state.turn_count,
        }
        if extra_context:
            context.update(extra_context)
        return {
            "configurable": {"thread_id": state.thread_id},
            "context": context,
        }
