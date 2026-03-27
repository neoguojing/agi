# browser_backend_core.py
import json
import logging
import random
import uuid
from asyncio import Lock
from collections.abc import AsyncIterator, Awaitable, Callable
from functools import partial
from pathlib import Path
from typing import Any, List, Dict, Optional
from playwright.async_api import (
    Browser, BrowserContext, Page, Response, TimeoutError as PlaywrightTimeoutError,
    async_playwright
)
from .browser_types import (
    DEFAULT_USER_AGENT, DEFAULT_VIEWPORT, DEFAULT_WAIT_UNTIL,
    STATE_SNAPSHOT_FILENAME, PLAYWRIGHT_STORAGE_STATE_FILENAME,
    PageInfo, QueryMatch, WaitUntilState, MAX_FIND_RESULTS, DEFAULT_CLICK_TIMEOUT_MS,
    DEFAULT_SCROLL_TIMEOUT_MS, DEFAULT_SMART_WAIT_TIMEOUT_MS, DEFAULT_CAPTURE_DELAY_MS
)
from .browser_protocal import AbstractBrowserBackend
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
        self._executor = None  # 不再单独使用 executor，逻辑直接内联
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

    async def navigate(self, url: str, wait_until: WaitUntilState = "networkidle") -> PageInfo:
        """Navigate to a URL and capture the resulting page state."""
        page = await self.ensure_page()
        
        # 使用 partial 绑定参数，避免 pickle 问题
        operation = partial(self._do_navigate, url=url, wait_until=wait_until)
        
        return await self._run_page_action(
            action="navigate",
            operation=operation,
            capture_url=url,
            history_entry={"action": "navigate", "url": url, "wait_until": wait_until},
        )

    async def click(self, selector: str) -> PageInfo:
        """Click an element identified by CSS selector."""
        page = await self.ensure_page()
        
        operation = partial(self._do_click, selector=selector)
        
        return await self._run_page_action(
            action="click",
            operation=operation,
            capture_url=None,
            history_entry={"action": "click", "selector": selector},
        )

    async def click_by_text(self, text: str) -> PageInfo:
        """Click the first element matching visible text."""
        page = await self.ensure_page()
        
        operation = partial(self._do_click_by_text, text=text)
        
        return await self._run_page_action(
            action="click_by_text",
            operation=operation,
            capture_url=None,
            history_entry={"action": "click_by_text", "text": text},
        )

    async def fill(self, selector: str, value: str) -> PageInfo:
        """Fill an editable element with text."""
        page = await self.ensure_page()
        
        operation = partial(self._do_fill, selector=selector, value=value)
        
        return await self._run_page_action(
            action="fill",
            operation=operation,
            capture_url=None,
            history_entry={"action": "fill", "selector": selector, "value": value},
        )

    async def fill_by_label(self, label_text: str, value: str) -> PageInfo:
        """Fill the first input associated with a matching label."""
        page = await self.ensure_page()
        
        operation = partial(self._do_fill_by_label, label_text=label_text, value=value)
        
        return await self._run_page_action(
            action="fill_by_label",
            operation=operation,
            capture_url=None,
            history_entry={"action": "fill_by_label", "label_text": label_text, "value": value},
        )

    async def fill_human_like(self, selector: str, value: str) -> PageInfo:
        """Type into a field character-by-character to mimic human input."""
        page = await self.ensure_page()
        
        operation = partial(self._do_fill_human_like, selector=selector, value=value)
        
        return await self._run_page_action(
            action="fill_human_like",
            operation=operation,
            capture_url=None,
            history_entry={"action": "fill_human_like", "selector": selector, "value": value},
        )

    async def find_elements(self, selector: str) -> List[QueryMatch]:
        """Return text and attributes for elements matching a CSS selector."""
        page = await self.ensure_page()
        self._event_manager.set_active_page(page)
        
        try:
            elements = await page.query_selector_all(selector)
            results: List[QueryMatch] = []
            for element in elements[:MAX_FIND_RESULTS]:
                text = await element.inner_text()
                attributes = await element.evaluate(
                    """el => {
                        const obj = {};
                        for (const attr of el.attributes) obj[attr.name] = attr.value;
                        return obj;
                    }"""
                )
                results.append(QueryMatch(selector=selector, text=text, attributes=attributes or {}))
            return results
        except Exception:
            logger.exception("find_elements failed for selector=%s", selector)
            return []

    async def get_screenshot(self, *, full_page: bool = True) -> str:
        """Capture a screenshot for OCR/inspection and return the absolute file path."""
        page = await self.ensure_page()
        
        try:
            screenshot_path = await self._take_screenshot(page, prefix="screenshot", full_page=full_page)
            return str(screenshot_path)
        except Exception:
            logger.exception("Screenshot failed")
            return ""
    
    async def read_screenshot_bytes(self, *, full_page: bool = True) -> tuple[str, bytes] | None:
        """Capture a screenshot for OCR/inspection and return both path and raw bytes."""
        screenshot_path = await self.get_screenshot(full_page=full_page)
        if not screenshot_path:
            return None
        return screenshot_path, Path(screenshot_path).read_bytes()

    def get_history(self) -> List[Dict[str, Any]]:
        """Return a copy of the recorded browser action history."""
        return self._event_manager.get_history()

    def peek_state_messages(self, limit: int = 1) -> List[Dict[str, Any]]:
        """Return recently published state messages without consuming them."""
        return self._event_manager.peek_state_messages(limit)

    def drain_state_messages(self, limit: int = 1) -> List[Dict[str, Any]]:
        """Drain pending state messages for downstream synchronizers."""
        return self._event_manager.drain_state_messages(limit)

    def get_recent_events(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Return a copy of the most recent browser/page events."""
        return self._event_manager.get_recent_events(limit)

    def get_state_snapshot(self, *, user_id: str | None = None, last_result: PageInfo | None = None) -> dict[str, Any]:
        """Return the current browser/context/page state for agent decision making."""
        current_page_state_obj = self._event_manager.get_page_runtime_state(self._page_id(self._page)) if self._page else None
        
        snapshot = {
            "current_url": self._page.url if self._page else None,
            "current_title": self._page.title() if self._page else None,
            "load_state": current_page_state_obj.load_state if current_page_state_obj else "unknown",
            "user_interaction_count": current_page_state_obj.user_interaction_count if current_page_state_obj else 0,
            "history_length": len(self._event_manager.get_history()),
            "user_id": user_id,
            "last_result": last_result,
        }
        return snapshot

    # --- Internal Action Implementations (from BrowserActionExecutor) ---

    async def _run_page_action(
        self,
        action: str,
        operation: Callable[[Page], Awaitable[Any]],
        *,
        capture_url: str | None,
        history_entry: dict[str, Any] | None = None,
    ) -> PageInfo:
        """Execute a browser action with retries and optional OCR fallback."""
        page = await self.ensure_page()
        self._event_manager.set_active_page(page)
        
        last_error: Exception | None = None
        
        for attempt in range(self.max_retry + 1):
            try:
                logger.info("Browser action '%s' attempt %s/%s", action, attempt + 1, self.max_retry + 1)
                response = await operation(page)
                
                # 智能等待
                await self._smart_wait(page)
                
                # 记录历史
                if history_entry is not None:
                    self._event_manager.add_to_history(history_entry)
                
                # 持久化状态
                if self._context is not None:
                    await self._persister.persist_playwright_storage_state(self._context)

                # 捕获页面信息
                return await self._capture_page_info(page, capture_url or page.url, response)
            
            except PlaywrightTimeoutError as exc:
                last_error = exc
                logger.warning("Browser action '%s' timed out on attempt %s/%s", action, attempt + 1, self.max_retry + 1)
            except Exception as exc:
                logger.exception("Browser action '%s' failed", action)
                return self._build_error_page_info(page.url, str(exc), action=action, attempt=attempt)

        return self._build_error_page_info(
            page.url,
            f"Max retries exceeded: {last_error}",
            action=action,
            attempt=self.max_retry,
        )

    async def _do_navigate(self, p: Page, url: str, wait_until: str) -> Response | None:
        """Navigate to a URL."""
        return await p.goto(url, wait_until=wait_until, timeout=self.timeout)

    async def _do_click(self, p: Page, selector: str) -> None:
        """Click an element."""
        await self._scroll_into_view(p, selector)
        await self._human_delay(100, 400)
        await p.click(selector, timeout=DEFAULT_CLICK_TIMEOUT_MS)

    async def _do_click_by_text(self, p: Page, text: str) -> None:
        """Click by text."""
        elements = await p.query_selector_all(f"text={text}")
        if not elements:
            raise ValueError(f"No element with text '{text}'")
        await elements[0].scroll_into_view_if_needed(timeout=DEFAULT_SCROLL_TIMEOUT_MS)
        await self._human_delay(100, 400)
        await elements[0].click(timeout=DEFAULT_CLICK_TIMEOUT_MS)

    async def _do_fill(self, p: Page, selector: str, value: str) -> None:
        """Fill an input field."""
        await self._scroll_into_view(p, selector)
        await p.fill(selector, value, timeout=DEFAULT_CLICK_TIMEOUT_MS)

    async def _do_fill_by_label(self, p: Page, label_text: str, value: str) -> None:
        """Fill by label text."""
        element = await p.query_selector(f"label:has-text('{label_text}') >> input")
        if element is None:
            raise ValueError(f"No input for label '{label_text}'")
        await element.scroll_into_view_if_needed(timeout=DEFAULT_SCROLL_TIMEOUT_MS)
        await element.fill(value)

    async def _do_fill_human_like(self, p: Page, selector: str, value: str) -> None:
        """Fill human-like."""
        await self._scroll_into_view(p, selector)
        await p.focus(selector)
        await p.fill(selector, "")
        for char in value:
            await p.keyboard.type(char, delay=random.randint(50, 150))
            await self._human_delay(p)

    async def _capture_page_info(self, page: Page, url: str, response: Response | None) -> PageInfo:
        """Capture normalized page metadata after an action completes."""
        try:
            html_repr = await self.extract_ui(page)
            
            page_text = await page.inner_text("body")
            page_title = await page.title()
            
            take_screenshot = self._should_capture_screenshot(
                page_text=page_text,
                response=response,
            )

            screenshot_path: str | None = None
            if take_screenshot:
                screenshot_path = str(await self._take_screenshot(page, prefix="page", full_page=True))

            page_info = PageInfo(
                url=page.url,
                title=page_title,
                html=html_repr,
                text="",
                screenshot_path=screenshot_path,
                metadata={
                    "requested_url": url,
                    "status": response.status if response is not None else 200,
                    "content_length": len(html_repr),
                    "text_length": len(page_text),
                    "has_screenshot": screenshot_path is not None,
                    "ocr_ready": screenshot_path is not None,
                    "history_length": len(self._event_manager.get_history()),
                }
            )

            return page_info
        except Exception as exc:
            logger.exception("Failed to capture page info for %s", url)
            return self._build_error_page_info(url, str(exc), action="capture")

    async def extract_ui(self, page: Page):
        """Extract UI elements from the page."""
        return await page.evaluate(""" () => {
            function getSelector(el) {
                if (el.id) return "#" + el.id;
                if (el.name) return `[name="${el.name}"]`;
                return el.tagName.toLowerCase();
            }

            function isVisible(el) {
                return !!(el.offsetParent);
            }

            function getText(el) {
                return (
                    el.innerText ||
                    el.value ||
                    el.getAttribute("aria-label") ||
                    el.title ||
                    ""
                ).trim();
            }

            const elements = Array.from(
                document.querySelectorAll('input, button, textarea, select, a')
            )
            .filter(isVisible)
            .map((el, idx) => {
                const rect = el.getBoundingClientRect();
                return {
                    id: idx + 1,
                    type: el.tagName.toLowerCase(),
                    text: getText(el),
                    href: el.href || "",
                    role: el.getAttribute("role") || "",
                    placeholder: el.placeholder || "",
                    selector: getSelector(el),
                    x: rect.x,
                    y: rect.y,
                    width: rect.width,
                    height: rect.height
                };
            })
            .filter(el => el.text.length > 0 || el.type === "input");

            return {
                page: {
                    title: document.title,
                    url: location.href
                },
                elements
            };
        } """)

    def _should_capture_screenshot(self, *, page_text: str, response: Response | None) -> bool:
        """Determine if a screenshot should be captured."""
        normalized_text = page_text.lower().strip()
        return (
            len(page_text.strip()) < self.min_text_length
            or (response is not None and response.status != 200)
            or any(keyword in normalized_text for keyword in self.ocr_keywords)
        )

    async def _take_screenshot(self, page: Page, *, prefix: str, full_page: bool = False) -> Path:
        """Take a screenshot and save to storage."""
        file_path = self.storage_dir / f"{prefix}_{uuid.uuid4().hex[:10]}.png"
        await page.screenshot(path=str(file_path), full_page=full_page)
        logger.info("Screenshot saved to %s", file_path)
        return file_path

    async def _smart_wait(self, page: Page, delay: int = DEFAULT_CAPTURE_DELAY_MS) -> None:
        """Wait for network stability, then add a small human-like delay."""
        try:
            await page.wait_for_load_state("networkidle", timeout=DEFAULT_SMART_WAIT_TIMEOUT_MS)
        except PlaywrightTimeoutError:
            logger.debug("networkidle wait timed out; continuing with fallback delay")
        await page.wait_for_timeout(delay)

    async def _scroll_into_view(self, page: Page, selector: str) -> None:
        """Scroll a target element into the viewport when possible."""
        try:
            element = await page.query_selector(selector)
            if element is not None:
                await element.scroll_into_view_if_needed(timeout=DEFAULT_SCROLL_TIMEOUT_MS)
        except Exception:
            logger.debug("Failed to scroll selector into view: %s", selector, exc_info=True)

    async def _human_delay(self, min_ms: int = 200, max_ms: int = 800) -> None:
        """Sleep briefly to simulate human interaction cadence."""
        await page.wait_for_timeout(random.randint(min_ms, max_ms))

    def _build_error_page_info(self, url: str, error: str, **metadata: Any) -> PageInfo:
        """Build an error PageInfo object."""
        return PageInfo(
            url=url,
            title=None,
            html=None,
            text=None,
            screenshot_path=None,
            metadata={"error": error, **metadata},
        )

    @property
    def min_text_length(self) -> int:
        return 50

    @property
    def min_html_length(self) -> int:
        return 100

    @property
    def ocr_keywords(self) -> List[str]:
        return ["captcha", "验证", "blocked"]
