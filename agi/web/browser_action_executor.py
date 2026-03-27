# browser_action_executor.py
import logging
import random
import uuid
from pathlib import Path
from typing import Any, Callable
from playwright.async_api import (
    BrowserContext, Browser, Page, Response, TimeoutError as PlaywrightTimeoutError,
    async_playwright
)
from .browser_types import (
    DEFAULT_CLICK_TIMEOUT_MS, DEFAULT_SCROLL_TIMEOUT_MS, DEFAULT_SMART_WAIT_TIMEOUT_MS,
    DEFAULT_CAPTURE_DELAY_MS, MAX_FIND_RESULTS, BROWSER_OBSERVER_SCRIPT, PageInfo, QueryMatch
)

logger = logging.getLogger(__name__)

class BrowserActionExecutor:
    """
    封装所有具体的浏览器操作逻辑，如点击、导航、填充等。
    """
    def __init__(self, storage_dir: Path, timeout: int, max_content_length: int, max_retry: int):
        self.storage_dir = storage_dir
        self.timeout = timeout
        self.max_content_length = max_content_length
        self.max_retry = max_retry

        self.min_text_length = 50
        self.min_html_length = 100
        self.ocr_keywords = ["captcha", "验证", "blocked"]

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

    async def run_page_action(
        self,
        page: Page,
        action: str,
        operation: Callable[[Page], Any],
        capture_url: str | None,
        history_entry: dict[str, Any] | None = None,
        event_recorder: callable = None,
        state_persister: callable = None,
    ) -> PageInfo:
        """
        执行一个浏览器动作，并处理错误重试、智能等待和页面信息捕获。
        """
        last_error: Exception | None = None
        for attempt in range(self.max_retry + 1):
            try:
                logger.info("Browser action '%s' attempt %s/%s", action, attempt + 1, self.max_retry + 1)
                response = await operation(page)
                
                # 智能等待
                await self._smart_wait(page)
                
                # 记录历史
                if history_entry is not None:
                    # 假设 event_recorder 有一个方法来添加历史记录
                    if event_recorder:
                         event_recorder.add_to_history(history_entry)
                
                # 持久化状态
                if state_persister and hasattr(state_persister, 'persist_playwright_storage_state'):
                    await state_persister.persist_playwright_storage_state(page.context)

                # 捕获页面信息
                return await self.capture_page_info(page, capture_url or page.url, response, event_recorder)
            
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

    def _build_error_page_info(self, url: str, error: str, **metadata: Any) -> PageInfo:
        return PageInfo(
            url=url,
            title=None,
            html=None,
            text=None,
            screenshot_path=None,
            metadata={"error": error, **metadata},
        )

    async def capture_page_info(self, page: Page, url: str, response: Response | None, event_recorder: callable) -> PageInfo:
        """Capture normalized page metadata after an action completes."""
        try:
            html_repr = await self.extract_ui(page)
            page_text = await page.inner_text("body")
            page_title = await page.title()

            # take_screenshot = self._should_capture_screenshot(
            #     page_text=page_text,
            #     response=response,
            # )
            # screenshot_path: str | None = None
            # if take_screenshot:
            #     screenshot_path = str(await self._take_screenshot(page, prefix="page", full_page=True))

            # if event_recorder:
            #      event_recorder.record_event(
            #          "page_capture",
            #          page=page,
            #          metadata={
            #              "requested_url": url,
            #              "status": response.status if response is not None else 200,
            #              "has_screenshot": screenshot_path is not None,
            #          },
            #      )

            page_info = PageInfo(
                url=page.url,
                title=page_title,
                html=html_repr,
                text="",
                screenshot_path="",
                metadata={
                    "requested_url": url,
                    "status": response.status if response is not None else 200,
                    "content_length": len(html_repr),
                    "text_length": len(page_text),
                    # Assuming history length is available from the recorder
                    "history_length": len(getattr(event_recorder, '_history', [])),
                },
            )
            return page_info
        except Exception as exc:
            logger.exception("Failed to capture page info for %s", url)
            return self._build_error_page_info(url, str(exc), action="capture")

    async def extract_ui(self, page: Page):
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
        normalized_text = page_text.lower().strip()
        return (
            len(page_text.strip()) < self.min_text_length
            or (response is not None and response.status != 200)
            or any(keyword in normalized_text for keyword in self.ocr_keywords)
        )

    async def _take_screenshot(self, page: Page, *, prefix: str, full_page: bool = False) -> Path:
        file_path = self.storage_dir / f"{prefix}_{uuid.uuid4().hex[:10]}.png"
        await page.screenshot(path=str(file_path), full_page=full_page)
        logger.info("Screenshot saved to %s", file_path)
        # Assuming event_recorder is passed to handle this
        # event_recorder.record_event(...)
        return file_path

    async def _smart_wait(self, page: Page, delay: int = DEFAULT_CAPTURE_DELAY_MS) -> None:
        """Wait for network stability, then add a small human-like delay."""
        try:
            await page.wait_for_load_state("networkidle", timeout=DEFAULT_SMART_WAIT_TIMEOUT_MS)
            # event_recorder.record_event(...)
        except PlaywrightTimeoutError:
            logger.debug("networkidle wait timed out; continuing with fallback delay")
            # event_recorder.record_event(...)
        await page.wait_for_timeout(delay)

    async def _scroll_into_view(self, page: Page, selector: str) -> None:
        """Scroll a target element into the viewport when possible."""
        try:
            element = await page.query_selector(selector)
            if element is not None:
                await element.scroll_into_view_if_needed(timeout=DEFAULT_SCROLL_TIMEOUT_MS)
        except Exception:
            logger.debug("Failed to scroll selector into view: %s", selector, exc_info=True)

    async def _human_delay(self, page: Page, min_ms: int = 200, max_ms: int = 800) -> None:
        """Sleep briefly to simulate human interaction cadence."""
        await page.wait_for_timeout(random.randint(min_ms, max_ms))

    # --- Action Implementations ---

    async def navigate(self, page: Page, url: str, wait_until: str = "domcontentloaded") -> Callable[[Page], Any]:
        async def operation(p: Page) -> Response | None:
            return await p.goto(url, wait_until=wait_until, timeout=self.timeout)
        return operation

    async def click(self, page: Page, selector: str) -> Callable[[Page], Any]:
        async def operation(p: Page) -> Response | None:
            await self._scroll_into_view(p, selector)
            await self._human_delay(p, 100, 400)
            await p.click(selector, timeout=DEFAULT_CLICK_TIMEOUT_MS)
            # event_recorder.record_event(...)
            return None
        return operation

    async def click_by_text(self, page: Page, text: str) -> Callable[[Page], Any]:
        async def operation(p: Page) -> Response | None:
            elements = await p.query_selector_all(f"text={text}")
            if not elements:
                msg = f"No element with text '{text}'"
                raise ValueError(msg)
            await elements[0].scroll_into_view_if_needed(timeout=DEFAULT_SCROLL_TIMEOUT_MS)
            await self._human_delay(p, 100, 400)
            await elements[0].click(timeout=DEFAULT_CLICK_TIMEOUT_MS)
            # event_recorder.record_event(...)
            return None
        return operation

    async def fill(self, page: Page, selector: str, value: str) -> Callable[[Page], Any]:
        async def operation(p: Page) -> Response | None:
            await self._scroll_into_view(p, selector)
            await p.fill(selector, value, timeout=DEFAULT_CLICK_TIMEOUT_MS)
            # event_recorder.record_event(...)
            return None
        return operation

    async def fill_by_label(self, page: Page, label_text: str, value: str) -> Callable[[Page], Any]:
        async def operation(p: Page) -> Response | None:
            element = await p.query_selector(f"label:has-text('{label_text}') >> input")
            if element is None:
                msg = f"No input for label '{label_text}'"
                raise ValueError(msg)
            await element.scroll_into_view_if_needed(timeout=DEFAULT_SCROLL_TIMEOUT_MS)
            await element.fill(value)
            # event_recorder.record_event(...)
            return None
        return operation

    async def fill_human_like(self, page: Page, selector: str, value: str) -> Callable[[Page], Any]:
        async def operation(p: Page) -> Response | None:
            await self._scroll_into_view(p, selector)
            await p.focus(selector)
            await p.fill(selector, "")
            for char in value:
                await p.keyboard.type(char, delay=random.randint(50, 150))
                await self._human_delay(p)
            # event_recorder.record_event(...)
            return None
        return operation

    async def find_elements(self, page: Page, selector: str) -> list[QueryMatch]:
        """Return text and attributes for elements matching a CSS selector."""
        try:
            elements = await page.query_selector_all(selector)
            results: list[QueryMatch] = []
            for element in elements[:MAX_FIND_RESULTS]:
                text = await element.inner_text()
                attributes = await element.evaluate(
                    """el => {
                        const obj = {};
                        for (const attr of el.attributes)
                            obj[attr.name] = attr.value;
                        return obj;
                    }"""
                )
                results.append(QueryMatch(selector=selector, text=text, attributes=attributes or {}))
            # event_recorder.record_event(...)
            return results
        except Exception:
            logger.exception("find_elements failed for selector=%s", selector)
            return []

    async def get_screenshot(self, page: Page, *, full_page: bool = True) -> str:
        """Capture a screenshot for OCR/inspection and return the absolute file path."""
        try:
            screenshot_path = await self._take_screenshot(page, prefix="screenshot", full_page=full_page)
            return str(screenshot_path)
        except Exception:
            logger.exception("Screenshot failed")
            return ""
