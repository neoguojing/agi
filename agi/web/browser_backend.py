# browser_backend_core.py
import json
import logging
import random
import uuid
from asyncio import Lock
from pathlib import Path
from typing import Any, List, Dict, Optional
from playwright.async_api import (
    Browser, BrowserContext, Page, Response, TimeoutError as PlaywrightTimeoutError,
    async_playwright
)
from .browser_types import (
    DEFAULT_USER_AGENT, DEFAULT_VIEWPORT, DEFAULT_WAIT_UNTIL,
    STATE_SNAPSHOT_FILENAME, PLAYWRIGHT_STORAGE_STATE_FILENAME,
    BrowserPageState,
    BrowserSessionSnapshot,
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
        is_closed = getattr(page, "is_closed", lambda: True)
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
        
        try:
            # 直接导航，不使用 operation 包装
            response = await page.goto(url, wait_until=wait_until, timeout=self.timeout)
            
            # 智能等待
            await self._smart_wait(page)
            
            # 记录历史
            history_entry = {"action": "navigate", "url": url, "wait_until": wait_until}
            self._event_manager.add_to_history(history_entry)
            
            # 持久化状态
            if self._context is not None:
                await self._persister.persist_playwright_storage_state(self._context)

            # 捕获页面信息（仅记录，不截图/OCR）
            return await self._capture_page_info(page, url, response, capture_content=False)
        
        except PlaywrightTimeoutError as exc:
            logger.warning("Navigation timed out for %s", url)
            return self._build_error_page_info(url, str(exc), action="navigate")
        except Exception as exc:
            # 检查是否是页面关闭的错误，需要重新初始化
            error_str = str(exc)
            if "TargetClosedError" in error_str or "closed" in error_str.lower():
                logger.warning("Page was closed during navigation, reinitializing browser for %s", url)
                try:
                    await self.close()
                    await self.initialize()
                    page = await self.ensure_page()
                    # 重试导航
                    response = await page.goto(url, wait_until=wait_until, timeout=self.timeout)
                    await self._smart_wait(page)
                    history_entry = {"action": "navigate", "url": url, "wait_until": wait_until}
                    self._event_manager.add_to_history(history_entry)
                    if self._context is not None:
                        await self._persister.persist_playwright_storage_state(self._context)
                    return await self._capture_page_info(page, url, response, capture_content=False)
                except Exception as retry_exc:
                    logger.exception("Retry navigation failed for %s", url)
                    return self._build_error_page_info(url, str(retry_exc), action="navigate")
            else:
                logger.exception("Navigation failed for %s", url)
                return self._build_error_page_info(url, str(exc), action="navigate")

    async def click(self, selector: str) -> PageInfo:
        """Click an element identified by CSS selector."""
        page = await self.ensure_page()
        
        try:
            # 直接点击，不使用 operation 包装
            await self._scroll_into_view(page, selector)
            await self._human_delay(100, 400)
            await page.click(selector, timeout=DEFAULT_CLICK_TIMEOUT_MS)
            
            # 智能等待
            await self._smart_wait(page)
            
            # 记录历史
            history_entry = {"action": "click", "selector": selector}
            self._event_manager.add_to_history(history_entry)
            
            # 持久化状态
            if self._context is not None:
                await self._persister.persist_playwright_storage_state(self._context)

            # 捕获页面信息（仅记录，不截图/OCR）
            return await self._capture_page_info(page, None, None, capture_content=False)
        
        except PlaywrightTimeoutError as exc:
            logger.warning("Click timed out for selector %s", selector)
            return self._build_error_page_info(page.url, str(exc), action="click")
        except Exception as exc:
            # 检查是否是页面关闭的错误，需要重新初始化
            error_str = str(exc)
            if "TargetClosedError" in error_str or "closed" in error_str.lower():
                logger.warning("Page was closed during click, reinitializing browser for selector %s", selector)
                try:
                    await self.close()
                    await self.initialize()
                    page = await self.ensure_page()
                    # 重试点击
                    await self._scroll_into_view(page, selector)
                    await self._human_delay(100, 400)
                    await page.click(selector, timeout=DEFAULT_CLICK_TIMEOUT_MS)
                    await self._smart_wait(page)
                    history_entry = {"action": "click", "selector": selector}
                    self._event_manager.add_to_history(history_entry)
                    if self._context is not None:
                        await self._persister.persist_playwright_storage_state(self._context)
                    return await self._capture_page_info(page, None, None, capture_content=False)
                except Exception as retry_exc:
                    logger.exception("Retry click failed for selector %s", selector)
                    return self._build_error_page_info(page.url, str(retry_exc), action="click")
            else:
                logger.exception("Click failed for selector %s", selector)
                return self._build_error_page_info(page.url, str(exc), action="click")

    async def click_by_text(self, text: str) -> PageInfo:
        """Click the first element matching visible text."""
        page = await self.ensure_page()
        
        try:
            # 直接通过文本点击
            elements = await page.query_selector_all(f"text={text}")
            if not elements:
                raise ValueError(f"No element with text '{text}'")
            
            await elements[0].scroll_into_view_if_needed(timeout=DEFAULT_SCROLL_TIMEOUT_MS)
            await self._human_delay(100, 400)
            await elements[0].click(timeout=DEFAULT_CLICK_TIMEOUT_MS)
            
            # 智能等待
            await self._smart_wait(page)
            
            # 记录历史
            history_entry = {"action": "click_by_text", "text": text}
            self._event_manager.add_to_history(history_entry)
            
            # 持久化状态
            if self._context is not None:
                await self._persister.persist_playwright_storage_state(self._context)

            # 捕获页面信息（仅记录，不截图/OCR）
            return await self._capture_page_info(page, None, None, capture_content=False)
        
        except PlaywrightTimeoutError as exc:
            logger.warning("Click by text timed out for %s", text)
            return self._build_error_page_info(page.url, str(exc), action="click_by_text")
        except Exception as exc:
            # 检查是否是页面关闭的错误，需要重新初始化
            error_str = str(exc)
            if "TargetClosedError" in error_str or "closed" in error_str.lower():
                logger.warning("Page was closed during click_by_text, reinitializing browser for text %s", text)
                try:
                    await self.close()
                    await self.initialize()
                    page = await self.ensure_page()
                    # 重试点击
                    elements = await page.query_selector_all(f"text={text}")
                    if not elements:
                        raise ValueError(f"No element with text '{text}'")
                    
                    await elements[0].scroll_into_view_if_needed(timeout=DEFAULT_SCROLL_TIMEOUT_MS)
                    await self._human_delay(100, 400)
                    await elements[0].click(timeout=DEFAULT_CLICK_TIMEOUT_MS)
                    await self._smart_wait(page)
                    history_entry = {"action": "click_by_text", "text": text}
                    self._event_manager.add_to_history(history_entry)
                    if self._context is not None:
                        await self._persister.persist_playwright_storage_state(self._context)
                    return await self._capture_page_info(page, None, None, capture_content=False)
                except Exception as retry_exc:
                    logger.exception("Retry click_by_text failed for text %s", text)
                    return self._build_error_page_info(page.url, str(retry_exc), action="click_by_text")
            else:
                logger.exception("Click by text failed for %s", text)
                return self._build_error_page_info(page.url, str(exc), action="click_by_text")

    async def fill(self, selector: str, value: str) -> PageInfo:
        """Fill an editable element with text."""
        page = await self.ensure_page()
        
        try:
            # 直接填充，不使用 operation 包装
            await self._scroll_into_view(page, selector)
            await page.fill(selector, value, timeout=DEFAULT_CLICK_TIMEOUT_MS)
            
            # 智能等待
            await self._smart_wait(page)
            
            # 记录历史
            history_entry = {"action": "fill", "selector": selector, "value": value}
            self._event_manager.add_to_history(history_entry)
            
            # 持久化状态
            if self._context is not None:
                await self._persister.persist_playwright_storage_state(self._context)

            # 捕获页面信息（仅记录，不截图/OCR）
            return await self._capture_page_info(page, None, None, capture_content=False)
        
        except PlaywrightTimeoutError as exc:
            logger.warning("Fill timed out for selector %s", selector)
            return self._build_error_page_info(page.url, str(exc), action="fill")
        except Exception as exc:
            # 检查是否是页面关闭的错误，需要重新初始化
            error_str = str(exc)
            if "TargetClosedError" in error_str or "closed" in error_str.lower():
                logger.warning("Page was closed during fill, reinitializing browser for selector %s", selector)
                try:
                    await self.close()
                    await self.initialize()
                    page = await self.ensure_page()
                    # 重试填充
                    await self._scroll_into_view(page, selector)
                    await page.fill(selector, value, timeout=DEFAULT_CLICK_TIMEOUT_MS)
                    await self._smart_wait(page)
                    history_entry = {"action": "fill", "selector": selector, "value": value}
                    self._event_manager.add_to_history(history_entry)
                    if self._context is not None:
                        await self._persister.persist_playwright_storage_state(self._context)
                    return await self._capture_page_info(page, None, None, capture_content=False)
                except Exception as retry_exc:
                    logger.exception("Retry fill failed for selector %s", selector)
                    return self._build_error_page_info(page.url, str(retry_exc), action="fill")
            else:
                logger.exception("Fill failed for selector %s", selector)
                return self._build_error_page_info(page.url, str(exc), action="fill")

    async def fill_by_label(self, label_text: str, value: str) -> PageInfo:
        """Fill the first input associated with a matching label."""
        page = await self.ensure_page()
        
        try:
            # 直接通过标签填充
            element = await page.query_selector(f"label:has-text('{label_text}') >> input")
            if element is None:
                raise ValueError(f"No input for label '{label_text}'")
            
            await element.scroll_into_view_if_needed(timeout=DEFAULT_SCROLL_TIMEOUT_MS)
            await element.fill(value)
            
            # 智能等待
            await self._smart_wait(page)
            
            # 记录历史
            history_entry = {"action": "fill_by_label", "label_text": label_text, "value": value}
            self._event_manager.add_to_history(history_entry)
            
            # 持久化状态
            if self._context is not None:
                await self._persister.persist_playwright_storage_state(self._context)

            # 捕获页面信息（仅记录，不截图/OCR）
            return await self._capture_page_info(page, None, None, capture_content=False)
        
        except PlaywrightTimeoutError as exc:
            logger.warning("Fill by label timed out for %s", label_text)
            return self._build_error_page_info(page.url, str(exc), action="fill_by_label")
        except Exception as exc:
            # 检查是否是页面关闭的错误，需要重新初始化
            error_str = str(exc)
            if "TargetClosedError" in error_str or "closed" in error_str.lower():
                logger.warning("Page was closed during fill_by_label, reinitializing browser for label %s", label_text)
                try:
                    await self.close()
                    await self.initialize()
                    page = await self.ensure_page()
                    # 重试填充
                    element = await page.query_selector(f"label:has-text('{label_text}') >> input")
                    if element is None:
                        raise ValueError(f"No input for label '{label_text}'")
                    
                    await element.scroll_into_view_if_needed(timeout=DEFAULT_SCROLL_TIMEOUT_MS)
                    await element.fill(value)
                    await self._smart_wait(page)
                    history_entry = {"action": "fill_by_label", "label_text": label_text, "value": value}
                    self._event_manager.add_to_history(history_entry)
                    if self._context is not None:
                        await self._persister.persist_playwright_storage_state(self._context)
                    return await self._capture_page_info(page, None, None, capture_content=False)
                except Exception as retry_exc:
                    logger.exception("Retry fill_by_label failed for label %s", label_text)
                    return self._build_error_page_info(page.url, str(retry_exc), action="fill_by_label")
            else:
                logger.exception("Fill by label failed for %s", label_text)
                return self._build_error_page_info(page.url, str(exc), action="fill_by_label")

    async def fill_human_like(self, selector: str, value: str) -> PageInfo:
        """Type into a field character-by-character to mimic human input."""
        page = await self.ensure_page()
        
        try:
            # 直接模拟人类输入
            await self._scroll_into_view(page, selector)
            await page.focus(selector)
            await page.fill(selector, "")
            
            for char in value:
                await page.keyboard.type(char, delay=random.randint(50, 150))
                await self._human_delay()
            
            # 智能等待
            await self._smart_wait(page)
            
            # 记录历史
            history_entry = {"action": "fill_human_like", "selector": selector, "value": value}
            self._event_manager.add_to_history(history_entry)
            
            # 持久化状态
            if self._context is not None:
                await self._persister.persist_playwright_storage_state(self._context)

            # 捕获页面信息（仅记录，不截图/OCR）
            return await self._capture_page_info(page, None, None, capture_content=False)
        
        except PlaywrightTimeoutError as exc:
            logger.warning("Fill human-like timed out for %s", selector)
            return self._build_error_page_info(page.url, str(exc), action="fill_human_like")
        except Exception as exc:
            # 检查是否是页面关闭的错误，需要重新初始化
            error_str = str(exc)
            if "TargetClosedError" in error_str or "closed" in error_str.lower():
                logger.warning("Page was closed during fill_human_like, reinitializing browser for selector %s", selector)
                try:
                    await self.close()
                    await self.initialize()
                    page = await self.ensure_page()
                    # 重试填充
                    await self._scroll_into_view(page, selector)
                    await page.focus(selector)
                    await page.fill(selector, "")
                    
                    for char in value:
                        await page.keyboard.type(char, delay=random.randint(50, 150))
                        await self._human_delay()
                    
                    await self._smart_wait(page)
                    history_entry = {"action": "fill_human_like", "selector": selector, "value": value}
                    self._event_manager.add_to_history(history_entry)
                    if self._context is not None:
                        await self._persister.persist_playwright_storage_state(self._context)
                    return await self._capture_page_info(page, None, None, capture_content=False)
                except Exception as retry_exc:
                    logger.exception("Retry fill_human_like failed for selector %s", selector)
                    return self._build_error_page_info(page.url, str(retry_exc), action="fill_human_like")
            else:
                logger.exception("Fill human-like failed for %s", selector)
                return self._build_error_page_info(page.url, str(exc), action="fill_human_like")

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

    def _page_summary(self, result: PageInfo | None, fallback_state: Any | None = None) -> BrowserPageState:
        return {
            "url": (result.url if result else "") or (fallback_state.url if fallback_state else ""),
            "title": (result.title if result else None) or (fallback_state.title if fallback_state else None),
            "load_state": fallback_state.load_state if fallback_state else "unknown",
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
        current_page_state = self._event_manager.get_page_runtime_state(self._page_id(self._page)) if self._page else None
        snapshot: BrowserSessionSnapshot = {
            "user_id": user_id or "default",
            "storage_dir": str(self.storage_dir),
            "history_length": len(self._event_manager.get_history()),
            "current_page": self._page_summary(last_result, current_page_state),
            "previous_page": self._page_summary(previous_result) if previous_result else None,
        }
        return snapshot

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
                # screenshot_path=screenshot_path,
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

