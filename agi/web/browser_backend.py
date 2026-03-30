# browser_backend_core.py
import json
import logging
import random
import uuid
from asyncio import Lock
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional
from playwright.async_api import (
    Browser, BrowserContext, Page, Response, TimeoutError as PlaywrightTimeoutError,
    async_playwright
)
from .browser_types import (
    BrowserRuntimeState,
    DEFAULT_USER_AGENT, DEFAULT_VIEWPORT, DEFAULT_WAIT_UNTIL,
    STATE_SNAPSHOT_FILENAME, PLAYWRIGHT_STORAGE_STATE_FILENAME,
    BrowserHistoryEntry,
    BrowserSessionSnapshot,
    PageInfo, QueryMatch, WaitUntilState, MAX_FIND_RESULTS, DEFAULT_CLICK_TIMEOUT_MS,
    normalize_browser_session_snapshot,
    DEFAULT_SCROLL_TIMEOUT_MS, DEFAULT_SMART_WAIT_TIMEOUT_MS, DEFAULT_CAPTURE_DELAY_MS
)
from .browser_protocal import AbstractBrowserBackend
from .browser_state_persister import BrowserStatePersister

logger = logging.getLogger(__name__)

class StatefulBrowserBackend(AbstractBrowserBackend):
    """Stateful Playwright backend for browser automation.

    设计说明：
    - 对外暴露的动作接口（navigate/click/fill...）尽量保持薄封装。
    - 错误恢复（浏览器被关闭后重拉）统一在 `_run_page_action` 里处理，
      避免每个接口都复制一套复杂 try/except。
    - 状态持久化与历史记录在动作成功后集中处理，保证行为一致性。
    """

    def __init__(
        self,
        storage_dir: str,
        headless: bool = False,
        timeout: int = 60_000,
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
        restored_snapshot = self._load_persisted_state_snapshot()
        self._history: list[BrowserHistoryEntry] = []
        self._page_runtime_state: dict[str, PageInfo] = {}
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
        is_closed = getattr(page, "is_closed", lambda: True)
        if callable(is_closed):
            try:
                return bool(is_closed())
            except Exception:
                return False
        return False

    def _update_page_runtime_state(self, page: Page, *, load_state: str | None = None) -> None:
        page_id = self._page_id(page)
        previous = self._page_runtime_state.get(page_id)
        metadata = dict(previous.metadata) if previous else {}
        metadata["load_state"] = load_state or metadata.get("load_state", "unknown")
        self._page_runtime_state[page_id] = PageInfo(
            url=page.url,
            title=previous.title if previous else None,
            html=previous.html if previous else None,
            text=previous.text if previous else None,
            screenshot_path=previous.screenshot_path if previous else None,
            metadata=metadata,
        )
        logger.debug("Updated runtime state for %s: url=%s load_state=%s", page_id, page.url, metadata["load_state"])

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
                # 检查浏览器是否真的还活着
                try:
                    if self._page and not self._page_is_closed(self._page):
                        return
                except Exception:
                    logger.warning("Browser page check failed, reinitializing")

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
        
        self._page = await self._context.new_page()
        self._update_page_runtime_state(self._page, load_state="ready")
        
        logger.info("Browser backend ready")

    async def close(self) -> None:
        """Close the browser session and release Playwright resources."""
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
            self._page_runtime_state.clear()
            self._history.clear()
            logger.info("Browser backend closed")

    @property
    def is_closed(self) -> bool:
        if self._page is None or self._context is None or self._browser is None:
            return True
        try:
            is_closed_attr = getattr(self._page, "is_closed", lambda: True)
            return is_closed_attr()
        except Exception:
            return True

    async def ensure_page(self) -> Page:
        """Return the active page, initializing the backend when needed."""
        if self.is_closed:
            await self.initialize()
            if self._context is not None:
                live_pages = [p for p in self._context.pages if not self._page_is_closed(p)]
                if live_pages:
                    self._page = live_pages[-1]
                    self._update_page_runtime_state(self._page, load_state="ready")

            if self._page is None:
                msg = "Browser page is not available after initialization"
                raise RuntimeError(msg)
        return self._page

    def _is_recoverable_browser_error(self, exc: Exception) -> bool:
        """是否属于可通过重建浏览器会话恢复的错误。"""
        error_str = str(exc).lower()
        return "targetclosederror" in error_str or "closed" in error_str

    async def _recover_browser_session(self, action: str) -> Page:
        """关闭并重建浏览器，返回可用 page。"""
        logger.warning("Recovering browser session for action=%s", action)
        await self.close()
        await self.initialize()
        return await self.ensure_page()

    async def _run_page_action(
        self,
        *,
        action: str,
        operation: Callable[[Page], Awaitable[Response | None]],
        history_entry: dict[str, Any] | None,
        capture_url: str | None = None,
    ) -> PageInfo:
        """执行页面动作并自动处理重试/恢复。

        流程：
        1) 执行一次动作；
        2) 若失败且是关闭类错误，则自动重建浏览器并重试一次；
        3) 成功后统一等待、记录 history、持久化状态并返回页面快照。
        """
        page = await self.ensure_page()
        attempts = 2  # 首次 + 1 次恢复后重试

        for attempt in range(1, attempts + 1):
            try:
                response = await operation(page)
                await self._smart_wait(page)
                if history_entry:
                    structured_entry: BrowserHistoryEntry = {
                        "action": str(history_entry.get("action", action)),
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "params": {k: v for k, v in history_entry.items() if k != "action"},
                    }
                    self._history.append(structured_entry)
                    logger.info("Recorded browser history entry: %s", structured_entry)
                if self._context is not None:
                    await self._persister.persist_playwright_storage_state(self._context)
                return await self._capture_page_info(
                    page,
                    capture_url or page.url,
                    response,
                    capture_content=False,
                )
            except PlaywrightTimeoutError as exc:
                logger.warning("%s timed out: %s", action, exc)
                return self._build_error_page_info(page.url, str(exc), action=action)
            except Exception as exc:
                can_retry = attempt < attempts and self._is_recoverable_browser_error(exc)
                if can_retry:
                    logger.warning("%s failed due to closed page, retrying once", action)
                    page = await self._recover_browser_session(action)
                    continue
                logger.exception("%s failed", action)
                return self._build_error_page_info(page.url, str(exc), action=action)

        return self._build_error_page_info("", "unexpected action runner state", action=action)

    async def navigate(self, url: str, wait_until: WaitUntilState = "networkidle") -> PageInfo:
        """Navigate to a URL and capture the resulting page state."""
        return await self._run_page_action(
            action="navigate",
            operation=lambda page: page.goto(url, wait_until=wait_until, timeout=self.timeout),
            history_entry={"action": "navigate", "url": url, "wait_until": wait_until},
            capture_url=url,
        )

    async def click(self, selector: str) -> PageInfo:
        """Click an element identified by CSS selector."""
        async def _operation(page: Page) -> None:
            await self._scroll_into_view(page, selector)
            await self._human_delay(100, 400)
            await page.click(selector, timeout=DEFAULT_CLICK_TIMEOUT_MS)
            return None

        return await self._run_page_action(
            action="click",
            operation=_operation,
            history_entry={"action": "click", "selector": selector},
        )

    async def click_by_text(self, text: str) -> PageInfo:
        """Click the first element matching visible text."""
        async def _operation(page: Page) -> None:
            elements = await page.query_selector_all(f"text={text}")
            if not elements:
                raise ValueError(f"No element with text '{text}'")
            await elements[0].scroll_into_view_if_needed(timeout=DEFAULT_SCROLL_TIMEOUT_MS)
            await self._human_delay(100, 400)
            await elements[0].click(timeout=DEFAULT_CLICK_TIMEOUT_MS)
            return None

        return await self._run_page_action(
            action="click_by_text",
            operation=_operation,
            history_entry={"action": "click_by_text", "text": text},
        )

    async def fill(self, selector: str, value: str) -> PageInfo:
        """Fill an editable element with text."""
        async def _operation(page: Page) -> None:
            await self._scroll_into_view(page, selector)
            await page.fill(selector, value, timeout=DEFAULT_CLICK_TIMEOUT_MS)
            return None

        return await self._run_page_action(
            action="fill",
            operation=_operation,
            history_entry={"action": "fill", "selector": selector, "value": value},
        )

    async def fill_by_label(self, label_text: str, value: str) -> PageInfo:
        """Fill the first input associated with a matching label."""
        async def _operation(page: Page) -> None:
            element = await page.query_selector(f"label:has-text('{label_text}') >> input")
            if element is None:
                raise ValueError(f"No input for label '{label_text}'")
            await element.scroll_into_view_if_needed(timeout=DEFAULT_SCROLL_TIMEOUT_MS)
            await element.fill(value)
            return None

        return await self._run_page_action(
            action="fill_by_label",
            operation=_operation,
            history_entry={"action": "fill_by_label", "label_text": label_text, "value": value},
        )

    async def fill_human_like(self, selector: str, value: str) -> PageInfo:
        """Type into a field character-by-character to mimic human input."""
        async def _operation(page: Page) -> None:
            await self._scroll_into_view(page, selector)
            await page.focus(selector)
            await page.fill(selector, "")
            for char in value:
                await page.keyboard.type(char, delay=random.randint(50, 150))
                await self._human_delay()
            return None

        return await self._run_page_action(
            action="fill_human_like",
            operation=_operation,
            history_entry={"action": "fill_human_like", "selector": selector, "value": value},
        )

    async def find_elements(self, selector: str) -> List[QueryMatch]:
        """Return text and attributes for elements matching a CSS selector."""
        page = await self.ensure_page()
        self._update_page_runtime_state(page, load_state="ready")
        
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

    def _page_summary(self, result: PageInfo | None, fallback_state: PageInfo | None = None) -> dict[str, Any]:
        source = result or fallback_state
        if source is None:
            return {
                "url": "",
                "title": None,
                "html": None,
                "text": None,
                "screenshot_path": None,
                "metadata": {"load_state": "unknown"},
            }
        return {
            "url": source.url,
            "title": source.title,
            "html": source.html,
            "text": source.text,
            "screenshot_path": source.screenshot_path,
            "metadata": dict(source.metadata),
        }

    def get_state_snapshot(
        self,
        *,
        user_id: str | None = None,
        last_result: PageInfo | None = None,
        previous_result: PageInfo | None = None,
    ) -> BrowserSessionSnapshot:
        """Return a compact state snapshot for middleware/LLM planning.

        Schema is intentionally minimal:
        - current_page: latest known page summary.
        - previous_page: immediate previous page summary (if any).
        """
        current_page_state = self._page_runtime_state.get(self._page_id(self._page)) if self._page else None
        browser_state: BrowserRuntimeState = {
            "is_open": not self.is_closed,
            "is_closed": self.is_closed,
        }
        snapshot: BrowserSessionSnapshot = {
            "browser": browser_state,
            "current_page": self._page_summary(last_result, current_page_state),
            "previous_page": self._page_summary(previous_result) if previous_result else None,
        }
        snapshot = normalize_browser_session_snapshot(snapshot)
        logger.debug(
            "Generated browser snapshot: is_open=%s current_url=%s",
            snapshot["browser"]["is_open"],
            snapshot["current_page"].get("url"),
        )
        return snapshot

    def get_history(self) -> list[dict[str, Any]]:
        """Return a copy of structured browser history entries."""
        logger.debug("Returning browser history, size=%s", len(self._history))
        return [dict(entry) for entry in self._history]

    # --- Internal Action Implementations ---

    async def _capture_page_info(self, page: Page, url: str, response: Response | None, capture_content: bool = True) -> PageInfo:
        """Capture normalized page metadata after an action completes."""
        try:
            html_repr = await self.extract_ui(page)
            
            page_text = await page.inner_text("body")
            page_title = await page.title()
            
            # 仅在需要内容时才进行截图和 OCR
            screenshot_path = None
            if capture_content:
                screenshot_path = str(await self._take_screenshot(page, prefix="page", full_page=True))

            page_info = PageInfo(
                url=page.url,
                title=page_title,
                html=html_repr,
                text="",
                screenshot_path=screenshot_path,
                metadata={
                    "status": response.status if response is not None else 200,
                    "html_length": len(html_repr),
                    "text_length": len(page_text),
                    "has_screenshot": screenshot_path is not None,
                    "history_length": len(self._history),
                }
            )
            self._page_runtime_state[self._page_id(page)] = page_info
            logger.info("Captured page info: url=%s title=%s", page_info.url, page_info.title)

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
        page = await self.ensure_page()
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
