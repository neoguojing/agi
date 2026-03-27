# browser_backend_core.py
import json
import logging
from asyncio import Lock
from pathlib import Path
from typing import Any
from playwright.async_api import Browser, BrowserContext, Page, async_playwright
from .browser_types import (
    DEFAULT_USER_AGENT, DEFAULT_VIEWPORT, DEFAULT_WAIT_UNTIL,
    STATE_SNAPSHOT_FILENAME, PLAYWRIGHT_STORAGE_STATE_FILENAME, BROWSER_OBSERVER_SCRIPT,
    PageInfo, QueryMatch, WaitUntilState
)
from .browser_protocal import AbstractBrowserBackend
from .browser_action_executor import BrowserActionExecutor
from .browser_event_manager import BrowserEventManager
from .browser_state_persister import BrowserStatePersister

logger = logging.getLogger(__name__)

class StatefulBrowserBackend(AbstractBrowserBackend): # 继承抽象类
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
            # executable_path="/usr/bin/google-chrome", # Note: This path might need configuration
        )

        context_kwargs: dict[str, Any] = {
            "viewport": DEFAULT_VIEWPORT,
            "user_agent": DEFAULT_USER_AGENT,
        }
        snapshot_paths = self._persister.get_persistent_paths()
        if snapshot_paths[1].exists(): # Check for PLAYWRIGHT_STORAGE_STATE_FILENAME
            context_kwargs["storage_state"] = str(snapshot_paths[1])

        self._context = await self._browser.new_context(**context_kwargs)
        await self._register_context_instrumentation(self._context)
        self._page = await self._context.new_page()
        self._event_manager.set_active_page(self._page)
        await self._instrument_page(self._page, source="initialize")
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
                live_pages = [p for p in self._context.pages if not getattr(p, 'is_closed', lambda: True)()]
                if live_pages:
                    self._page = live_pages[-1]
                    self._event_manager.set_active_page(self._page)
                    await self._instrument_page(self._page, source="ensure_page")

            if self._page is None:
                msg = "Browser page is not available after initialization"
                raise RuntimeError(msg)
        return self._page

    # --- 实现所有抽象方法 ---

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

    def get_history(self) -> list[dict[str, Any]]:
        return self._event_manager.get_history()

    def peek_state_messages(self, limit: int = 1) -> list[dict[str, Any]]:
        return self._event_manager.peek_state_messages(limit)

    def drain_state_messages(self, limit: int = 1) -> list[dict[str, Any]]:
        return self._event_manager.drain_state_messages(limit)

    def get_recent_events(self, limit: int = 5) -> list[dict[str, Any]]:
        return self._event_manager.get_recent_events(limit)

    def get_state_snapshot(self, *, user_id: str | None = None, last_result: PageInfo | None = None) -> dict[str, Any]:
        """Generate a minimal snapshot of the current browser state."""
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

    def get_state_snapshot_full(self, *, user_id: str | None = None, last_result: PageInfo | None = None) -> dict[str, Any]:
        """Generate a comprehensive snapshot of the current browser state."""
        restored = self._persister._restored_state_snapshot or {}
        
        # 获取所有活动页签的 ID
        all_page_ids = []
        if self._context:
            all_page_ids = [
                self._page_id(p) for p in self._context.pages if not self._page_is_closed(p)
            ]

        # 当前页签的状态
        current_page_id = self._page_id(self._page)
        current_page_state = self._event_manager.get_page_runtime_state(current_page_id) if self._page else {}

        # 组合快照数据
        snapshot = {
            "version": "1.0",
            "user_id": user_id,
            "last_result": last_result.__dict__ if last_result else None,
            "current_page_id": current_page_id,
            "all_page_ids": all_page_ids,
            "active_page": {
                "id": current_page_id,
                "url": self._page.url if self._page else None,
                "title": self._page.title() if self._page else None,
                "runtime_state": current_page_state,
            },
            "history": self._event_manager.get_history(),
            "recent_events": self._event_manager.get_recent_events(),
            "restored_snapshot": restored,
        }
        return snapshot

    # --- 补全其他动作方法 ---
    async def go_back(self) -> PageInfo:
        page = await self.ensure_page()
        async def operation(p: Page):
            await p.go_back()
            return None
        return await self._executor.run_page_action(
            page, "go_back", operation, capture_url=None,
            history_entry={"action": "go_back"},
            event_recorder=self._event_manager,
            state_persister=self._persister
        )

    async def go_forward(self) -> PageInfo:
        page = await self.ensure_page()
        async def operation(p: Page):
            await p.go_forward()
            return None
        return await self._executor.run_page_action(
            page, "go_forward", operation, capture_url=None,
            history_entry={"action": "go_forward"},
            event_recorder=self._event_manager,
            state_persister=self._persister
        )

    async def reload(self) -> PageInfo:
        page = await self.ensure_page()
        async def operation(p: Page):
            await p.reload()
            return None
        return await self._executor.run_page_action(
            page, "reload", operation, capture_url=None,
            history_entry={"action": "reload"},
            event_recorder=self._event_manager,
            state_persister=self._persister
        )

    async def get_page_source(self) -> str:
        page = await self.ensure_page()
        try:
            return await page.content()
        except Exception as e:
            logger.error(f"Failed to get page source: {e}")
            return ""

    async def get_current_url(self) -> str:
        page = await self.ensure_page()
        try:
            return page.url
        except Exception as e:
            logger.error(f"Failed to get current URL: {e}")
            return ""

    async def get_title(self) -> str:
        page = await self.ensure_page()
        try:
            return await page.title()
        except Exception as e:
            logger.error(f"Failed to get page title: {e}")
            return ""

    async def evaluate(self, expression: str) -> Any:
        page = await self.ensure_page()
        try:
            result = await page.evaluate(expression)
            return result
        except Exception as e:
            logger.error(f"Failed to evaluate expression: {e}")
            return None

    async def wait_for_selector(self, selector: str, timeout: int = 30000) -> bool:
        page = await self.ensure_page()
        try:
            await page.wait_for_selector(selector, timeout=timeout)
            return True
        except Exception as e:
            logger.error(f"Failed to wait for selector '{selector}': {e}")
            return False

    async def wait_for_function(self, function: str, timeout: int = 30000) -> bool:
        page = await self.ensure_page()
        try:
            await page.wait_for_function(function, timeout=timeout)
            return True
        except Exception as e:
            logger.error(f"Failed to wait for function: {e}")
            return False

    async def close_other_tabs(self) -> None:
        """关闭除当前页签外的所有页签"""
        if not self._context:
            return
        current_page = await self.ensure_page()
        pages_to_close = [p for p in self._context.pages if p != current_page and not self._page_is_closed(p)]
        for page in pages_to_close:
            try:
                await page.close()
                self._event_manager.record_event("page_closed", page=page)
            except Exception as e:
                logger.warning(f"Failed to close page: {e}")

    async def switch_to_tab(self, tab_index: int) -> PageInfo | None:
        """切换到指定索引的页签"""
        if not self._context:
            return None
        pages = [p for p in self._context.pages if not self._page_is_closed(p)]
        if 0 <= tab_index < len(pages):
            self._page = pages[tab_index]
            self._event_manager.set_active_page(self._page)
            await self._instrument_page(self._page, source="switch_tab")
            # 返回新页签的信息
            return await self._executor.capture_page_info(self._page, self._page.url, None, self._event_manager)
        return None

    async def open_new_tab(self, url: str = "about:blank") -> PageInfo:
        """打开一个新页签并导航到指定URL"""
        if not self._context:
            await self.initialize()
        new_page = await self._context.new_page()
        self._event_manager.set_active_page(new_page)
        await self._instrument_page(new_page, source="open_new_tab")
        return await self.navigate(url)

    # --- 内部辅助方法 ---
    async def _register_context_instrumentation(self, context: BrowserContext) -> None:
        """Register global event listeners for the browser context."""
        context.on("page", self._handle_new_page)
        # logger.debug("Registered context instrumentation for %s", self._page_id(context.pages[-1] if context.pages else None))

    async def _instrument_page(self, page: Page, *, source: str) -> None:
        """Install DOM event observers and state tracking for a specific page."""
        page_id = self._page_id(page)
        if page_id in self._event_manager._instrumented_pages:
            return

        self._event_manager._instrumented_pages.add(page_id)

        # 1. 安装 JavaScript 事件观察者
        try:
            await page.add_init_script(content=BROWSER_OBSERVER_SCRIPT)
            await page.expose_binding("__agiRecordBrowserEvent", self._on_browser_event, handle=True)
            self._event_manager.record_event(
                "page_instrumented",
                page=page,
                metadata={"source": source, "page_count": len(self._context.pages)},
            )
        except Exception as e:
            logger.warning("Failed to instrument page %s: %s", page_id, e)

    async def _handle_new_page(self, page: Page) -> None:
        """Handle a new page being opened within the context."""
        self._event_manager.record_event(
            "page_opened",
            page=page,
            metadata={"page_count": len(self._context.pages)},
        )
        await self._instrument_page(page, source="new_page")
        self._event_manager.update_page_title(page, await page.title())

    async def _on_browser_event(self, source, event_data: dict[str, Any]) -> None:
        """Callback for events emitted by the JavaScript observer script."""
        self._event_manager.record_event(
            event_data.get("type", "unknown"),
            page=source.page,
            metadata=event_data,
        )
        self._event_manager.update_page_title(source.page, event_data.get("title"))