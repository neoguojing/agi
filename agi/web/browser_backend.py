# browser_backend_core.py
import json
import logging
from asyncio import Lock
from pathlib import Path
from typing import Any
from playwright.async_api import Browser, BrowserContext, Page, async_playwright
from .browser_types import (
    DEFAULT_USER_AGENT, DEFAULT_VIEWPORT, DEFAULT_WAIT_UNTIL,
    STATE_SNAPSHOT_FILENAME, PLAYWRIGHT_STORAGE_STATE_FILENAME,
    PageInfo, QueryMatch, WaitUntilState
)
from .browser_protocal import AbstractBrowserBackend
from .browser_action_executor import BrowserActionExecutor
from .browser_event_manager import BrowserEventManager
from .browser_state_persister import BrowserStatePersister

logger = logging.getLogger(__name__)

class StatefulBrowserBackend(AbstractBrowserBackend):
    """Stateful Playwright backend for browser automation."""

    def __init__(
        self,
        storage_dir: str,
        headless: bool = False,
        timeout: int = 30_000,
        max_content_length: int = 2_000_000,
        max_retry: int = 2,
    ):
        self.headless = headless
        self.timeout = timeout
        self.max_content_length = max_content_length
        self.max_retry = max_retry

        self.storage_dir = Path(storage_dir).resolve()
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        # 初始化子模块
        self._executor = BrowserActionExecutor(self.storage_dir, self.timeout, self.max_content_length, self.max_retry)
        restored_snapshot = self._load_persisted_state_snapshot()
        self._event_manager = BrowserEventManager(self.storage_dir)
        self._persister = BrowserStatePersister(self.storage_dir, restored_snapshot)

        self._init_lock = Lock()
        self._playwright = None
        self._browser: Browser | None = None
        self._context: BrowserContext | None = None
        self._page: Page | None = None

    def _page_id(self, page: Page | None) -> str:
        if page is None:
            return "page:none"
        return f"page:{id(page)}"

    def _page_is_closed(self, page: Page | None) -> bool:
        if page is None:
            return True
        is_closed = getattr(page, "is_closed", None)
        if callable(is_closed):
            try:
                return bool(is_closed())
            except Exception:
                return False
        return False

    def _load_persisted_state_snapshot(self) -> dict[str, Any] | None:
        snapshot_path = self.storage_dir / STATE_SNAPSHOT_FILENAME
        if not snapshot_path.exists():
            return None
        try:
            data = json.loads(snapshot_path.read_text())
            return data if isinstance(data, dict) else None
        except Exception:
            logger.debug("Failed to load persisted browser state snapshot", exc_info=True)
            return None

    def get_running_loop(self):
        import asyncio

        return asyncio.get_running_loop()
    
    async def initialize(self) -> None:
        """Initialize the shared browser session lazily."""
        async with self._init_lock:
            if self._browser is not None:
                return

        logger.info("Launching browser backend")
        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(
            headless=self.headless,
            args=["--no-sandbox", "--disable-setuid-sandbox"],
        )

        context_kwargs: dict[str, Any] = {
            "viewport": DEFAULT_VIEWPORT,
            "user_agent": DEFAULT_USER_AGENT,
        }
        snapshot_paths = self._persister.get_persistent_paths()
        if snapshot_paths[1].exists():
            context_kwargs["storage_state"] = str(snapshot_paths[1])

        self._context = await self._browser.new_context(**context_kwargs)
        
        # 将事件注册逻辑委托给 EventManager
        await self._event_manager.register_context_instrumentation(self._context)
        
        self._page = await self._context.new_page()
        self._event_manager.set_active_page(self._page)
        
        # 将页面仪器化逻辑也委托给 EventManager
        await self._event_manager.instrument_page(self._page, source="initialize")
        
        logger.info("Browser backend ready")

    async def close(self) -> None:
        """Close the browser session and release Playwright resources."""
        self._event_manager.record_event(
            "browser_closed",
            page=self._page,
            metadata={"storage_dir": str(self.storage_dir)},
        )
        try:
            if self._context is not None:
                await self._persister.persist_playwright_storage_state(self._context)
            if self._page is not None:
                await self._page.close()
            if self._context is not None:
                await self._context.close()
            if self._browser is not None:
                await self._browser.close()
            if self._playwright is not None:
                await self._playwright.stop()
        finally:
            self._page = None
            self._context = None
            self._browser = None
            self._playwright = None
            self._event_manager.set_active_page(None)
            # Clear internal states
            self._event_manager._page_runtime_state.clear()
            self._event_manager._state_messages.clear()
            self._event_manager._instrumented_pages.clear()
            logger.info("Browser backend closed")

    @property
    def is_closed(self) -> bool:
        if self._page is None or self._context is None or self._browser is None:
            return True
        is_closed_attr = getattr(self._page, "is_closed", lambda: True)
        return is_closed_attr()

    async def ensure_page(self) -> Page:
        """Return the active page, initializing the backend when needed."""
        if self.is_closed:
            await self.initialize()
            if self._context is not None:
                live_pages = [p for p in self._context.pages if not self._page_is_closed(p)]
                if live_pages:
                    self._page = live_pages[-1]
                    self._event_manager.set_active_page(self._page)
                    # 确保新激活的页面也被仪器化
                    await self._event_manager.instrument_page(self._page, source="ensure_page")

            if self._page is None:
                msg = "Browser page is not available after initialization"
                raise RuntimeError(msg)
        return self._page

    # --- 实现所有抽象方法 (与之前相同) ---

    async def navigate(self, url: str, wait_until: WaitUntilState = "networkidle") -> PageInfo:
        page = await self.ensure_page()
        operation = await self._executor.navigate(page, url, wait_until)
        return await self._executor.run_page_action(
            page, "navigate", operation, capture_url=url,
            history_entry={"action": "navigate", "url": url, "wait_until": wait_until},
            event_recorder=self._event_manager,
            state_persister=self._persister
        )

    async def click(self, selector: str) -> PageInfo:
        page = await self.ensure_page()
        operation = await self._executor.click(page, selector)
        return await self._executor.run_page_action(
            page, "click", operation, capture_url=None,
            history_entry={"action": "click", "selector": selector},
            event_recorder=self._event_manager,
            state_persister=self._persister
        )

    async def click_by_text(self, text: str) -> PageInfo:
        page = await self.ensure_page()
        operation = await self._executor.click_by_text(page, text)
        return await self._executor.run_page_action(
            page, "click_by_text", operation, capture_url=None,
            history_entry={"action": "click_by_text", "text": text},
            event_recorder=self._event_manager,
            state_persister=self._persister
        )

    async def fill(self, selector: str, value: str) -> PageInfo:
        page = await self.ensure_page()
        operation = await self._executor.fill(page, selector, value)
        return await self._executor.run_page_action(
            page, "fill", operation, capture_url=None,
            history_entry={"action": "fill", "selector": selector, "value": value},
            event_recorder=self._event_manager,
            state_persister=self._persister
        )

    async def fill_by_label(self, label_text: str, value: str) -> PageInfo:
        page = await self.ensure_page()
        operation = await self._executor.fill_by_label(page, label_text, value)
        return await self._executor.run_page_action(
            page, "fill_by_label", operation, capture_url=None,
            history_entry={"action": "fill_by_label", "label_text": label_text, "value": value},
            event_recorder=self._event_manager,
            state_persister=self._persister
        )

    async def fill_human_like(self, selector: str, value: str) -> PageInfo:
        page = await self.ensure_page()
        operation = await self._executor.fill_human_like(page, selector, value)
        return await self._executor.run_page_action(
            page, "fill_human_like", operation, capture_url=None,
            history_entry={"action": "fill_human_like", "selector": selector, "value": value},
            event_recorder=self._event_manager,
            state_persister=self._persister
        )

    async def find_elements(self, selector: str) -> list[QueryMatch]:
        page = await self.ensure_page()
        self._event_manager.set_active_page(page)
        return await self._executor.find_elements(page, selector)

    async def get_screenshot(self, *, full_page: bool = True) -> str:
        page = await self.ensure_page()
        return await self._executor.get_screenshot(page, full_page=full_page)
    
    async def read_screenshot_bytes(self, *, full_page: bool = True) -> tuple[str, bytes] | None:
        """Capture a screenshot for OCR/inspection and return both path and raw bytes."""
        screenshot_path = await self.get_screenshot(full_page=full_page)
        if not screenshot_path:
            return None
        return screenshot_path, Path(screenshot_path).read_bytes()

    def get_history(self) -> list[dict[str, Any]]:
        return self._event_manager.get_history()

    def peek_state_messages(self, limit: int = 1) -> list[dict[str, Any]]:
        return self._event_manager.peek_state_messages(limit)

    def drain_state_messages(self, limit: int = 1) -> list[dict[str, Any]]:
        return self._event_manager.drain_state_messages(limit)

    def get_recent_events(self, limit: int = 5) -> list[dict[str, Any]]:
        return self._event_manager.get_recent_events(limit)

    def get_state_snapshot(self, *, user_id: str | None = None, last_result: PageInfo | None = None) -> dict[str, Any]:
        current_page_state = self._event_manager.get_page_runtime_state(self._page_id(self._page)) if self._page else {}
        snapshot = {
            "current_url": self._page.url if self._page else None,
            "current_title": self._page.title() if self._page else None,
            "load_state": current_page_state.get("load_state", "unknown"),
            "user_interaction_count": current_page_state.get("user_interaction_count", 0),
            "history_length": len(self._event_manager.get_history()),
            "user_id": user_id,
            "last_result": last_result.__dict__ if last_result else None,
        }
        return snapshot
