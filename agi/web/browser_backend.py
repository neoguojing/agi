import logging
import random
import uuid
from asyncio import Lock
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from playwright.async_api import (
    Browser,
    BrowserContext,
    Page,
    Response,
    TimeoutError as PlaywrightTimeoutError,
    async_playwright,
)

logger = logging.getLogger(__name__)

DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)
DEFAULT_VIEWPORT = {"width": 1280, "height": 720}
DEFAULT_WAIT_UNTIL = "domcontentloaded"
DEFAULT_SMART_WAIT_TIMEOUT_MS = 5_000
DEFAULT_CLICK_TIMEOUT_MS = 5_000
DEFAULT_SCROLL_TIMEOUT_MS = 2_000
DEFAULT_CAPTURE_DELAY_MS = 300
MAX_FIND_RESULTS = 50


@dataclass(slots=True)
class PageInfo:
    url: str
    title: str | None
    html: str | None
    text: str | None
    screenshot_path: str | None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class QueryMatch:
    selector: str
    text: str
    attributes: dict[str, Any]


class StatefulBrowserBackend:
    """Stateful Playwright backend for browser automation."""

    def __init__(
        self,
        storage_dir: str,
        headless: bool = True,
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

        self._init_lock = Lock()
        self._playwright = None
        self._browser: Browser | None = None
        self._context: BrowserContext | None = None
        self._page: Page | None = None
        self._history: list[dict[str, Any]] = []

        self.min_text_length = 50
        self.min_html_length = 100
        self.ocr_keywords = ["captcha", "验证", "blocked"]

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
            self._context = await self._browser.new_context(
                viewport=DEFAULT_VIEWPORT,
                user_agent=DEFAULT_USER_AGENT,
            )
            self._page = await self._context.new_page()
            logger.info("Browser backend ready")

    async def close(self) -> None:
        """Close the browser session and release Playwright resources."""
        try:
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
            logger.info("Browser backend closed")

    async def ensure_page(self) -> Page:
        """Return the active page, initializing the backend when needed."""
        if self._page is None:
            await self.initialize()
        if self._page is None:
            msg = "Browser page is not available after initialization"
            raise RuntimeError(msg)
        return self._page

    async def navigate(self, url: str, wait_until: str = DEFAULT_WAIT_UNTIL) -> PageInfo:
        """Navigate to a page and capture the resulting page state."""

        async def operation(page: Page) -> Response | None:
            return await page.goto(url, wait_until=wait_until, timeout=self.timeout)

        return await self._run_page_action(
            action="navigate",
            operation=operation,
            capture_url=url,
            history_entry={"action": "navigate", "url": url, "wait_until": wait_until},
        )

    async def click(self, selector: str) -> PageInfo:
        """Click an element identified by CSS selector."""

        async def operation(page: Page) -> Response | None:
            await self._scroll_into_view(page, selector)
            await self._human_delay(100, 400)
            await page.click(selector, timeout=DEFAULT_CLICK_TIMEOUT_MS)
            return None

        return await self._run_page_action(
            action="click",
            operation=operation,
            capture_url=None,
            history_entry={"action": "click", "selector": selector},
        )

    async def click_by_text(self, text: str) -> PageInfo:
        """Click the first element matching visible text."""

        async def operation(page: Page) -> Response | None:
            elements = await page.query_selector_all(f"text={text}")
            if not elements:
                msg = f"No element with text '{text}'"
                raise ValueError(msg)
            await elements[0].scroll_into_view_if_needed(timeout=DEFAULT_SCROLL_TIMEOUT_MS)
            await self._human_delay(100, 400)
            await elements[0].click(timeout=DEFAULT_CLICK_TIMEOUT_MS)
            return None

        return await self._run_page_action(
            action="click_by_text",
            operation=operation,
            capture_url=None,
            history_entry={"action": "click_by_text", "text": text},
        )

    async def fill(self, selector: str, value: str) -> PageInfo:
        """Fill an editable element with text."""

        async def operation(page: Page) -> Response | None:
            await self._scroll_into_view(page, selector)
            await page.fill(selector, value, timeout=DEFAULT_CLICK_TIMEOUT_MS)
            return None

        return await self._run_page_action(
            action="fill",
            operation=operation,
            capture_url=None,
            history_entry={"action": "fill", "selector": selector, "value": value},
        )

    async def fill_by_label(self, label_text: str, value: str) -> PageInfo:
        """Fill the first input associated with a matching label."""

        async def operation(page: Page) -> Response | None:
            element = await page.query_selector(f"label:has-text('{label_text}') >> input")
            if element is None:
                msg = f"No input for label '{label_text}'"
                raise ValueError(msg)
            await element.scroll_into_view_if_needed(timeout=DEFAULT_SCROLL_TIMEOUT_MS)
            await element.fill(value)
            return None

        return await self._run_page_action(
            action="fill_by_label",
            operation=operation,
            capture_url=None,
            history_entry={"action": "fill_by_label", "label": label_text, "value": value},
        )

    async def fill_human_like(self, selector: str, value: str) -> PageInfo:
        """Type into a field character-by-character to mimic human input."""

        async def operation(page: Page) -> Response | None:
            await self._scroll_into_view(page, selector)
            await page.focus(selector)
            await page.fill(selector, "")
            for char in value:
                await page.keyboard.type(char, delay=random.randint(50, 150))
            await self._human_delay()
            return None

        return await self._run_page_action(
            action="fill_human_like",
            operation=operation,
            capture_url=None,
            history_entry={"action": "fill_human_like", "selector": selector, "value": value},
        )

    async def find_elements(self, selector: str) -> list[QueryMatch]:
        """Return text and attributes for elements matching a CSS selector."""
        page = await self.ensure_page()
        try:
            elements = await page.query_selector_all(selector)
            results: list[QueryMatch] = []
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

    async def get_screenshot(self, *, full_page: bool = False) -> str:
        """Capture a screenshot and return the absolute file path."""
        page = await self.ensure_page()
        try:
            screenshot_path = await self._take_screenshot(page, prefix="screenshot", full_page=full_page)
            return str(screenshot_path)
        except Exception:
            logger.exception("Screenshot failed")
            return ""

    async def read_screenshot_bytes(self, *, full_page: bool = False) -> tuple[str, bytes] | None:
        """Capture a screenshot and return both path and raw bytes."""
        screenshot_path = await self.get_screenshot(full_page=full_page)
        if not screenshot_path:
            return None
        return screenshot_path, Path(screenshot_path).read_bytes()

    def get_history(self) -> list[dict[str, Any]]:
        """Return a copy of the recorded browser action history."""
        return list(self._history)

    async def _run_page_action(
        self,
        action: str,
        operation: Callable[[Page], Awaitable[Response | None]],
        *,
        capture_url: str | None,
        history_entry: dict[str, Any] | None = None,
    ) -> PageInfo:
        page = await self.ensure_page()
        last_error: Exception | None = None

        for attempt in range(self.max_retry + 1):
            try:
                logger.info("Browser action '%s' attempt %s/%s", action, attempt + 1, self.max_retry + 1)
                response = await operation(page)
                await self._smart_wait(page)
                if history_entry is not None:
                    self._history.append(history_entry)
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

    def _build_error_page_info(self, url: str, error: str, **metadata: Any) -> PageInfo:
        return PageInfo(
            url=url,
            title=None,
            html=None,
            text=None,
            screenshot_path=None,
            metadata={"error": error, **metadata},
        )

    async def _capture_page_info(self, page: Page, url: str, response: Response | None) -> PageInfo:
        """Capture normalized page metadata after an action completes."""
        try:
            html = await page.content()
            if len(html) > self.max_content_length:
                html = html[: self.max_content_length] + "\n... [TRUNCATED]"

            page_text = await page.inner_text("body")
            page_title = await page.title()
            take_screenshot = self._should_capture_screenshot(
                html=html,
                page_text=page_text,
                response=response,
            )

            screenshot_path: str | None = None
            if take_screenshot:
                screenshot_path = str(await self._take_screenshot(page, prefix="page"))

            return PageInfo(
                url=page.url,
                title=page_title,
                html=html,
                text=page_text,
                screenshot_path=screenshot_path,
                metadata={
                    "requested_url": url,
                    "status": response.status if response is not None else 200,
                    "content_length": len(html),
                    "text_length": len(page_text),
                    "has_screenshot": screenshot_path is not None,
                    "history_length": len(self._history),
                },
            )
        except Exception as exc:
            logger.exception("Failed to capture page info for %s", url)
            return self._build_error_page_info(url, str(exc), action="capture")

    def _should_capture_screenshot(self, *, html: str, page_text: str, response: Response | None) -> bool:
        normalized_text = page_text.lower().strip()
        return (
            len(page_text.strip()) < self.min_text_length
            or len(html.strip()) < self.min_html_length
            or (response is not None and response.status != 200)
            or any(keyword in normalized_text for keyword in self.ocr_keywords)
        )

    async def _take_screenshot(self, page: Page, *, prefix: str, full_page: bool = False) -> Path:
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
