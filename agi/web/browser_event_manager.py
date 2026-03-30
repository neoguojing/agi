"""Minimal browser event/state manager.

Design goal:
- Keep only enough runtime info for agent planning.
- Do not inject client-side scripts or collect heavy DOM events.
"""

import logging
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

from playwright.async_api import BrowserContext, Page

from .browser_types import BrowserEvent, BrowserEventType, PageRuntimeState

logger = logging.getLogger(__name__)


class BrowserEventManager:
    """Store a compact event log and page runtime summaries.

    This manager intentionally avoids page script injection. It only tracks
    events emitted from Python-side backend operations.
    """

    def __init__(self, storage_dir: Path):
        self.storage_dir = storage_dir
        # 1) 动作历史：供 backend 统计 history_length 和回放动作。
        self._history: List[Dict[str, Any]] = []
        # 2) 最近事件：供 agent 判断最近浏览器动态。
        self._recent_events: Deque[BrowserEvent] = deque(maxlen=10)
        # 3) 待消费消息：给上层做流式同步（peek/drain）。
        self._pending_messages: Deque[Dict[str, Any]] = deque(maxlen=10)
        # 4) 页面运行态：按 page_id 维护最小状态(url/title/load_state)。
        self._page_runtime_state: Dict[str, PageRuntimeState] = {}
        # 5) 已追踪页面：保证 instrument_page 幂等。
        self._tracked_pages: set[str] = set()

    def _page_id(self, page: Page | None) -> str:
        if page is None:
            return "page:none"
        return f"page:{id(page)}"

    def set_active_page(self, page: Optional[Page]) -> None:
        """Set active page and initialize compact runtime state if needed."""
        if page is None:
            return

        page_id = self._page_id(page)
        if page_id not in self._page_runtime_state:
            self._page_runtime_state[page_id] = PageRuntimeState(
                page_id=page_id,
                url=page.url,
                title="",
                load_state="unknown",
                last_update=datetime.now().isoformat(),
            )

    def update_page_state(self, page: Page, **kwargs: Any) -> None:
        """Update compact page runtime fields (url/title/load_state only)."""
        page_id = self._page_id(page)
        if page_id not in self._page_runtime_state:
            self._page_runtime_state[page_id] = PageRuntimeState(page_id=page_id, url=page.url)

        state = self._page_runtime_state[page_id]
        for key in ("url", "title", "load_state"):
            if key in kwargs and kwargs[key] is not None:
                setattr(state, key, kwargs[key])
        state.last_update = datetime.now().isoformat()

    def _normalize_event_type(self, event_type: BrowserEventType | str) -> BrowserEventType:
        if isinstance(event_type, BrowserEventType):
            return event_type
        try:
            return BrowserEventType(event_type)
        except ValueError:
            # fallback for unknown event name: keep minimal signal
            return BrowserEventType.NAVIGATION_START

    def record_event(
        self,
        event_type: BrowserEventType | str,
        page: Optional[Page],
        metadata: Dict[str, Any] | None = None,
    ) -> None:
        """Record one minimal event and sync page state hints."""
        safe_metadata = metadata or {}
        normalized_type = self._normalize_event_type(event_type)
        page_id = self._page_id(page)

        event = BrowserEvent(
            type=normalized_type,
            timestamp=datetime.now().isoformat(),
            page_id=page_id,
            metadata=safe_metadata,
        )
        self._recent_events.append(event)
        self._pending_messages.append({"type": "event", "data": event.to_dict()})

        if page is not None:
            self.update_page_state(
                page,
                url=safe_metadata.get("url", page.url),
                title=safe_metadata.get("title"),
                load_state=safe_metadata.get("load_state"),
            )

    def add_to_history(self, history_entry: Dict[str, Any]) -> None:
        self._history.append(history_entry)

    async def register_context_instrumentation(self, context: BrowserContext) -> None:
        """No-op instrumentation hook.

        We still listen for new pages to keep active page state accurate.
        """

        def _on_new_page(page: Page) -> None:
            self.set_active_page(page)
            self.record_event(
                BrowserEventType.PAGE_OPENED,
                page=page,
                metadata={"url": page.url, "load_state": "domcontentloaded"},
            )

        context.on("page", _on_new_page)

    async def instrument_page(self, page: Page, *, source: str) -> None:
        """Mark page as tracked without JS injection."""
        page_id = self._page_id(page)
        if page_id in self._tracked_pages:
            return
        self._tracked_pages.add(page_id)
        self.set_active_page(page)
        self.update_page_state(page, title=await page.title(), load_state="ready")
        self.record_event(BrowserEventType.INSTRUMENTED, page=page, metadata={"source": source, "url": page.url})

    def peek_state_messages(self, limit: int = 1) -> List[Dict[str, Any]]:
        return list(self._pending_messages)[:limit]

    def drain_state_messages(self, limit: int = 1) -> List[Dict[str, Any]]:
        messages: List[Dict[str, Any]] = []
        while self._pending_messages and len(messages) < limit:
            messages.append(self._pending_messages.popleft())
        return messages

    def get_recent_events(self, limit: int = 5) -> List[Dict[str, Any]]:
        events = list(self._recent_events)
        events.reverse()
        return [event.to_dict() for event in events[:limit]]

    def get_history(self) -> List[Dict[str, Any]]:
        return self._history.copy()

    def get_page_runtime_state(self, page_id: str) -> Optional[PageRuntimeState]:
        return self._page_runtime_state.get(page_id)

    def clear(self) -> None:
        """Clear all in-memory runtime state."""
        self._history.clear()
        self._recent_events.clear()
        self._pending_messages.clear()
        self._page_runtime_state.clear()
        self._tracked_pages.clear()
