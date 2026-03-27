# browser_event_manager.py
import logging
from asyncio import Queue
from pathlib import Path
from typing import Any
from playwright.async_api import Page
from .browser_types import (
    MAX_BROWSER_EVENTS, MAX_STATE_MESSAGES, USER_EVENT_TYPES, BROWSER_OBSERVER_SCRIPT
)

logger = logging.getLogger(__name__)

class BrowserEventManager:
    """
    负责管理浏览器事件、运行时状态和页面监听。
    """
    def __init__(self, storage_dir: Path):
        self.storage_dir = storage_dir
        self._history: list[dict[str, Any]] = []
        self._events: list[dict[str, Any]] = []
        self._state_message_queue: Queue[dict[str, Any]] = Queue()
        self._state_messages: list[dict[str, Any]] = []
        self._event_seq = 0
        self._active_page_id: str | None = None
        self._page_titles: dict[str, str | None] = {}
        self._page_runtime_state: dict[str, dict[str, Any]] = {}
        self._instrumented_pages: set[str] = set()
        self._binding_registered = False

    def add_to_history(self, entry: dict[str, Any]):
        self._history.append(entry)

    def get_history(self) -> list[dict[str, Any]]:
        return list(self._history)

    def get_recent_events(self, limit: int = 5) -> list[dict[str, Any]]:
        if limit <= 0:
            return []
        return [dict(event) for event in self._events[-limit:]]

    def peek_state_messages(self, limit: int = 1) -> list[dict[str, Any]]:
        if limit <= 0:
            return []
        return [dict(message) for message in self._state_messages[-limit:]]

    def drain_state_messages(self, limit: int = 1) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        while limit > 0 and not self._state_message_queue.empty():
            messages.append(self._state_message_queue.get_nowait())
            limit -= 1
        return messages

    def record_event(self, event_type: str, *, page: Page | None = None, metadata: dict[str, Any] | None = None) -> None:
        self._event_seq += 1
        page_id = self._page_id(page) if page is not None else self._active_page_id
        url = getattr(page, "url", None) if page is not None else None
        event = {
            "seq": self._event_seq,
            "type": event_type,
            "page_id": page_id,
            "url": url,
            "metadata": dict(metadata or {}),
        }
        self._events.append(event)
        self._update_page_runtime_state(event)
        self._publish_state_message(event)
        if len(self._events) > MAX_BROWSER_EVENTS:
            self._events = self._events[-MAX_BROWSER_EVENTS:]

    def _update_page_runtime_state(self, event: dict[str, Any]) -> None:
        page_id = event.get("page_id")
        if not page_id or page_id == "page:none":
            return

        runtime_state = self._page_runtime_state.setdefault(
            page_id,
            {
                "load_state": "idle",
                "last_event_type": None,
                "last_user_event": None,
                "last_interaction": None,
                "user_interaction_count": 0,
                "title": self._page_titles.get(page_id),
                "url": event.get("url"),
            },
        )
        event_type = str(event.get("type"))
        metadata = event.get("metadata", {}) if isinstance(event.get("metadata"), dict) else {}

        runtime_state["last_event_type"] = event_type
        runtime_state["url"] = metadata.get("url") or event.get("url") or runtime_state.get("url")
        if metadata.get("title"):
            runtime_state["title"] = metadata.get("title")

        if event_type in {"page_domcontentloaded", "page_navigated", "page_hashchange", "page_popstate"}:
            runtime_state["load_state"] = "loading"
        elif event_type in {"page_load", "page_load_state", "page_capture"}:
            runtime_state["load_state"] = "loaded"
        elif event_type in {"page_closed", "browser_closed"}:
            runtime_state["load_state"] = "closed"

        if event_type in USER_EVENT_TYPES or event_type.startswith("action_"):
            runtime_state["last_interaction"] = {
                "type": event_type,
                "url": runtime_state.get("url"),
                "target": metadata.get("target"),
                "timestamp": metadata.get("timestamp"),
            }
            if event_type in USER_EVENT_TYPES:
                runtime_state["last_user_event"] = {
                    "type": event_type,
                    "url": runtime_state.get("url"),
                    "target": metadata.get("target"),
                    "timestamp": metadata.get("timestamp"),
                }
            runtime_state["user_interaction_count"] = int(runtime_state.get("user_interaction_count", 0)) + 1

    def _publish_state_message(self, event: dict[str, Any]) -> None:
        message = {
            "kind": "browser_state",
            "event": dict(event),
            "event_version": self._event_seq,
        }
        self._state_messages.append(message)
        if len(self._state_messages) > MAX_STATE_MESSAGES:
            self._state_messages = self._state_messages[-MAX_STATE_MESSAGES:]
        self._state_message_queue.put_nowait(message)

    def update_page_title(self, page: Page, title: str):
        page_id = self._page_id(page)
        self._page_titles[page_id] = title

    def set_active_page(self, page: Page | None) -> None:
        if page is None:
            return
        self._active_page_id = self._page_id(page)
        self.record_event("active_page_changed", page=page, metadata={"page_count": -1}) # Placeholder count

    def get_page_runtime_state(self, page_id: str) -> dict[str, Any]:
        return self._page_runtime_state.get(page_id, {})

    def infer_load_state(self, last_event: dict[str, Any] | None) -> str:
        if not last_event:
            return "idle"
        event_type = last_event.get("type")
        if event_type in {"page_load", "page_load_state", "page_capture"}:
            return "loaded"
        if event_type in {"page_domcontentloaded", "page_navigated"}:
            return "loading"
        if event_type in {"page_closed", "browser_closed"}:
            return "closed"
        return "idle"

    def _page_id(self, page: Page | None) -> str:
        if page is None:
            return "page:none"
        return f"page:{id(page)}"