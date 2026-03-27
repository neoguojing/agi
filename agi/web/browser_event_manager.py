# browser_event_manager.py
import logging
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Deque, Dict, List, Any, Optional, Set
from playwright.async_api import Page, BrowserContext

logger = logging.getLogger(__name__)

class BrowserEventManager:
    def __init__(self, storage_dir: Path):
        self.storage_dir = storage_dir
        self._history: List[Dict[str, Any]] = []
        self._recent_events: Deque[Dict[str, Any]] = deque(maxlen=20)
        self._state_messages: Deque[Dict[str, Any]] = deque(maxlen=10)
        self._active_page: Optional[Page] = None
        self._page_runtime_state: Dict[str, Dict[str, Any]] = {}
        self._instrumented_pages: Set[str] = set()

    def set_active_page(self, page: Optional[Page]) -> None:
        self._active_page = page

    def get_page_runtime_state(self, page_id: str) -> Dict[str, Any]:
        return self._page_runtime_state.get(page_id, {})

    def update_page_title(self, page: Page, new_title: str) -> None:
        page_id = self._page_id(page)
        state = self._page_runtime_state.setdefault(page_id, {})
        old_title = state.get("title", "")
        if old_title != new_title:
            state["title"] = new_title
            logger.debug("Updated title for %s: %s", page_id, new_title)

    def record_event(self, event_type: str, page: Optional[Page], metadata: Dict[str, Any]={}) -> None:
        page_id = self._page_id(page)
        event = {
            "timestamp": datetime.now().isoformat(),
            "type": event_type,
            "page_id": page_id,
            "metadata": metadata,
        }
        self._recent_events.append(event)
        self._state_messages.append({
            "type": "event",
            "data": event
        })

    def log_history(self, action: str, url: str = "", metadata: Dict[str, Any] = {}) -> None:
        entry = {
            "action": action,
            "timestamp": datetime.now().isoformat(),
            "url": url,
            "metadata": metadata,
        }
        self._history.append(entry)
        self._state_messages.append({
            "type": "history_update",
            "data": entry
        })
        logger.info(f"Action: {action}, URL: {url}")

    def get_history(self) -> List[Dict[str, Any]]:
        return self._history.copy()

    def get_recent_events(self, limit: int = 5) -> List[Dict[str, Any]]:
        return list(self._recent_events)[-limit:]

    def peek_state_messages(self, limit: int = 1) -> List[Dict[str, Any]]:
        return list(self._state_messages)[-limit:]

    def drain_state_messages(self, limit: int = 1) -> List[Dict[str, Any]]:
        messages = list(self._state_messages)[:limit]
        for _ in range(min(limit, len(self._state_messages))):
            self._state_messages.popleft()
        return messages

    def _page_id(self, page: Optional[Page]) -> str:
        if page is None:
            return "page:none"
        return f"page:{id(page)}"

    def _page_is_closed(self, page: Page) -> bool:
        is_closed = getattr(page, "is_closed", None)
        if callable(is_closed):
            try:
                return bool(is_closed())
            except Exception:
                return True
        return False

    # --- 事件管理逻辑 ---
    async def register_context_instrumentation(self, context: BrowserContext) -> None:
        """Register global event listeners for the browser context."""
        context.on("page", self._handle_new_page)
        logger.debug("Registered context instrumentation.")

    async def instrument_page(self, page: Page, *, source: str) -> None:
        """Install DOM event observers and state tracking for a specific page."""
        page_id = self._page_id(page)
        if page_id in self._instrumented_pages:
            return

        self._instrumented_pages.add(page_id)

        # Install JavaScript event observer
        BROWSER_OBSERVER_SCRIPT = """
        // A global variable to store the last known page title.
        window.__agiLastTitle = document.title;

        // Function to send an event to the Python backend.
        function __agiSendEvent(eventType, payload) {
            // The '__agiRecordBrowserEvent' binding will be exposed by Python.
            window.__agiRecordBrowserEvent({ type: eventType, ...payload });
        }

        // Observe DOM mutations for dynamic content changes.
        const mutationObserver = new MutationObserver((mutations) => {
            let shouldNotify = false;
            for (const mutation of mutations) {
                if (mutation.type === 'childList' && mutation.addedNodes.length > 0) {
                    shouldNotify = true;
                    break;
                }
            }
            if (shouldNotify) {
                __agiSendEvent('dom_mutation', { timestamp: Date.now() });
            }
        });
        mutationObserver.observe(document, { childList: true, subtree: true });

        // Listen for beforeunload to detect navigations away from the page.
        window.addEventListener('beforeunload', () => {
            __agiSendEvent('navigation_start', { url: window.location.href });
        });

        // Listen for custom events dispatched by user interactions.
        document.addEventListener('click', (event) => {
            const target = event.target;
            const rect = target.getBoundingClientRect();
            __agiSendEvent('click_intercepted', {
                element_tag: target.tagName,
                element_id: target.id,
                element_class: target.className,
                element_text: target.innerText?.substring(0, 100), // Truncate long text
                clientX: event.clientX,
                clientY: event.clientY,
                viewportX: rect.left,
                viewportY: rect.top,
                timestamp: Date.now(),
            });
        });

        // Periodically check for title changes.
        setInterval(() => {
            if (document.title !== window.__agiLastTitle) {
                window.__agiLastTitle = document.title;
                __agiSendEvent('title_changed', { title: document.title });
            }
        }, 500);

        // Notify of initial state.
        __agiSendEvent('page_ready', { title: document.title, url: window.location.href });
        """

        try:
            await page.add_init_script(content=BROWSER_OBSERVER_SCRIPT)
            # Expose the Python callback to JavaScript
            await page.expose_binding("__agiRecordBrowserEvent", self._on_browser_event, handle=True)
            # 修复：使用 page.context.pages 获取页签数量
            page_count = len([p for p in page.context.pages if not self._page_is_closed(p)])
            self.record_event(
                "page_instrumented",
                page=page,
                metadata={"source": source, "page_count": page_count},
            )
        except Exception as e:
            logger.warning("Failed to instrument page %s: %s", page_id, e)

    async def _handle_new_page(self, page: Page) -> None:
        """Handle a new page being opened within the context."""
        # 修复：使用 page.context.pages 获取页签数量
        page_count = len([p for p in page.context.pages if not self._page_is_closed(p)])
        self.record_event(
            "page_opened",
            page=page,
            metadata={"page_count": page_count},
        )
        await self.instrument_page(page, source="new_page")
        self.update_page_title(page, await page.title())

    async def _on_browser_event(self, source, event_data: dict[str, Any]) -> None:
        """Callback for events emitted by the JavaScript observer script."""
        self.record_event(
            event_data.get("type", "unknown"),
            page=source.page,
            metadata=event_data,
        )
        self.update_page_title(source.page, event_data.get("title"))