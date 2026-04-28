# browser_backend_core.py
"""Production-ready Playwright browser automation backend.

This module provides a stateful browser automation backend with:
- Lazy initialization and recovery from closed browser sessions
- Human-like interaction delays to avoid detection
- Error handling and retry logic for recoverable failures
- Console and request failure auditing
- State persistence across sessions
"""

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
    DEFAULT_USER_AGENT, DEFAULT_VIEWPORT,
    STATE_SNAPSHOT_FILENAME,
    PageInfo, QueryMatch, WaitUntilState, MAX_FIND_RESULTS, DEFAULT_CLICK_TIMEOUT_MS,
    DEFAULT_SCROLL_TIMEOUT_MS, DEFAULT_SMART_WAIT_TIMEOUT_MS, DEFAULT_CAPTURE_DELAY_MS,
)
from .browser_protocal import AbstractBrowserBackend
from .browser_state_persister import BrowserStatePersister

logger = logging.getLogger(__name__)


class StatefulBrowserBackend(AbstractBrowserBackend):
    """Stateful Playwright backend for browser automation.

    Design principles:
    - Action interfaces (navigate/click/fill) maintain thin encapsulation.
    - Error recovery (reconnecting after browser closure) is unified in
      `_recover_browser_session` to avoid duplicating complex try/except blocks.
    - State persistence and history logging are centralized after successful actions
      to ensure behavioral consistency.
    """

    def __init__(
        self,
        storage_dir: str,
        headless: bool = False,
        timeout: int = 60_000,
        max_content_length: int = 2_000_000,
        max_retry: int = 2,
        max_error_log_size: int = 100,
    ):
        self.headless = headless
        self.timeout = timeout
        self.max_content_length = max_content_length
        self.max_retry = max_retry
        self.max_error_log_size = max_error_log_size

        self.storage_dir = Path(storage_dir).resolve()
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        # Initialize sub-modules
        restored_snapshot = self._load_persisted_state_snapshot()
        self._persister = BrowserStatePersister(self.storage_dir, restored_snapshot)
        self._recent_console_errors: List[Dict[str, Any]] = []
        self._recent_request_failures: List[Dict[str, Any]] = []

        self._init_lock = Lock()
        self._playwright = None
        self._browser: Browser | None = None
        self._context: BrowserContext | None = None
        self._page: Page | None = None

    def _trim_error_logs(self) -> None:
        """Trim error logs to max_size to prevent unbounded memory growth."""
        if len(self._recent_console_errors) > self.max_error_log_size:
            self._recent_console_errors = self._recent_console_errors[-self.max_error_log_size:]
        if len(self._recent_request_failures) > self.max_error_log_size:
            self._recent_request_failures = self._recent_request_failures[-self.max_error_log_size:]

    def _page_is_closed(self, page: Page | None) -> bool:
        """Check if the page is closed, handling edge cases."""
        if page is None:
            return True
        try:
            return bool(page.is_closed())
        except Exception:
            return False

    def _load_persisted_state_snapshot(self) -> Dict[str, Any] | None:
        """Load persisted state snapshot from storage."""
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
                # Check browser is still alive
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

        context_kwargs: Dict[str, Any] = {
            "viewport": DEFAULT_VIEWPORT,
            "user_agent": DEFAULT_USER_AGENT,
        }
        snapshot_paths = self._persister.get_persistent_paths()
        if snapshot_paths[1].exists():
            context_kwargs["storage_state"] = str(snapshot_paths[1])

        self._context = await self._browser.new_context(**context_kwargs)
        self._page = await self._context.new_page()
        self._attach_page_audit_hooks(self._page)

        logger.info("Browser backend ready")

    async def close(self) -> None:
        """Close the browser session and release Playwright resources."""
        try:
            # Persist storage state before closing
            if self._context is not None:
                await self._persister.persist_playwright_storage_state(self._context)

            # Close in reverse order of creation
            if self._page is not None and not self._page_is_closed(self._page):
                await self._page.close()
            if self._context is not None:
                await self._context.close()
            if self._browser is not None:
                await self._browser.close()
            if self._playwright is not None:
                await self._playwright.stop()
        except Exception as exc:
            logger.error("Error closing browser session: %s", exc, exc_info=True)
        finally:
            # Clear references to allow GC
            self._page = None
            self._context = None
            self._browser = None
            self._playwright = None
            # Trim error logs on close
            self._trim_error_logs()
            logger.info("Browser backend closed")

    @property
    def is_closed(self) -> bool:
        """Check if the browser session is closed."""
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
                    self._attach_page_audit_hooks(self._page)

            if self._page is None:
                msg = "Browser page is not available after initialization"
                raise RuntimeError(msg)
        return self._page

    def _is_recoverable_browser_error(self, exc: Exception) -> bool:
        """Check if the error is recoverable by reconnecting to the browser."""
        error_str = str(exc).lower()
        return "targetclosederror" in error_str or "closed" in error_str

    async def _recover_browser_session(self, action: str) -> Page:
        """Recover browser session by closing and reinitializing."""
        logger.warning("Recovering browser session for action=%s", action)
        await self.close()
        await self.initialize()
        return await self.ensure_page()

    async def navigate(self, url: str, wait_until: WaitUntilState = "domcontentloaded") -> PageInfo:
        """Navigate to a URL and capture the resulting page state."""
        page = await self.ensure_page()

        try:
            response = await page.goto(url, wait_until=wait_until, timeout=self.timeout)
            await self._smart_wait(page)

            # Build success response
            try:
                title = await page.title()
            except Exception:
                title = None

            page_info = PageInfo(
                url=page.url,
                title=title,
                viewport=DEFAULT_VIEWPORT,
                is_loading=False,
                last_action_status="success",
                error_message=None,
            )
            return page_info
        except PlaywrightTimeoutError as exc:
            logger.warning("navigate timed out: %s", exc)
            return PageInfo(
                url=page.url or "",
                title=None,
                viewport=DEFAULT_VIEWPORT,
                is_loading=False,
                last_action_status="timeout",
                error_message=f"Page navigation timed out: {exc}",
            )
        except Exception as exc:
            can_retry = self._is_recoverable_browser_error(exc)
            if can_retry:
                logger.warning("navigate failed due to closed page, retrying once")
                page = await self._recover_browser_session("navigate")
                return await self.navigate(url, wait_until)
            logger.exception("navigate failed", exc_info=True)
            return PageInfo(
                url=page.url or "",
                title=None,
                viewport=DEFAULT_VIEWPORT,
                is_loading=False,
                last_action_status="fail",
                error_message=f"{type(exc).__name__}: {exc}",
            )

    async def click(self, selector: str) -> PageInfo:
        """Click an element identified by CSS selector."""
        page = await self.ensure_page()

        try:
            async def _operation():
                await self._scroll_into_view(page, selector)
                await self._human_delay(100, 400)
                await page.click(selector, timeout=DEFAULT_CLICK_TIMEOUT_MS)

            await _operation()
            await self._smart_wait(page)

            # Build success response
            try:
                title = await page.title()
            except Exception:
                title = None

            page_info = PageInfo(
                url=page.url,
                title=title,
                viewport=DEFAULT_VIEWPORT,
                is_loading=False,
                last_action_status="success",
                error_message=None,
            )
            return page_info
        except PlaywrightTimeoutError as exc:
            logger.warning("click timed out for selector=%s: %s", selector, exc)
            return PageInfo(
                url=page.url or "",
                title=None,
                viewport=DEFAULT_VIEWPORT,
                is_loading=False,
                last_action_status="timeout",
                error_message=f"Element click timed out for selector '{selector}': {exc}",
            )
        except Exception as exc:
            can_retry = self._is_recoverable_browser_error(exc)
            if can_retry:
                logger.warning("click failed due to closed page for selector=%s, retrying once", selector)
                page = await self._recover_browser_session("click")
                return await self.click(selector)
            logger.exception("click failed for selector=%s", selector, exc_info=True)
            return PageInfo(
                url=page.url or "",
                title=None,
                viewport=DEFAULT_VIEWPORT,
                is_loading=False,
                last_action_status="fail",
                error_message=f"Click failed for selector '{selector}': {type(exc).__name__}: {exc}",
            )

    async def fill(self, selector: str, value: str) -> PageInfo:
        """Fill input field with specified value using human-like interaction."""
        page = await self.ensure_page()

        try:
            async def _operation():
                await self._scroll_into_view(page, selector)
                await self._human_delay(100, 400)
                # Clear existing value before filling
                await page.fill(selector, value, timeout=DEFAULT_CLICK_TIMEOUT_MS)

            await _operation()
            await self._smart_wait(page)

            # Build success response
            try:
                title = await page.title()
            except Exception:
                title = None

            page_info = PageInfo(
                url=page.url,
                title=title,
                viewport=DEFAULT_VIEWPORT,
                is_loading=False,
                last_action_status="success",
                error_message=None,
            )
            return page_info
        except PlaywrightTimeoutError as exc:
            logger.warning("fill timed out for selector=%s: %s", selector, exc)
            return PageInfo(
                url=page.url or "",
                title=None,
                viewport=DEFAULT_VIEWPORT,
                is_loading=False,
                last_action_status="timeout",
                error_message=f"Fill operation timed out for selector '{selector}': {exc}",
            )
        except Exception as exc:
            can_retry = self._is_recoverable_browser_error(exc)
            if can_retry:
                logger.warning("fill failed due to closed page for selector=%s, retrying once", selector)
                page = await self._recover_browser_session("fill")
                return await self.fill(selector, value)
            logger.exception("fill failed for selector=%s", selector, exc_info=True)
            return PageInfo(
                url=page.url or "",
                title=None,
                viewport=DEFAULT_VIEWPORT,
                is_loading=False,
                last_action_status="fail",
                error_message=f"Fill failed for selector '{selector}': {type(exc).__name__}: {exc}",
            )

    async def scroll(self, direction: str = "down", distance: int = 800) -> PageInfo:
        """Scroll viewport to reveal off-screen content and trigger lazy-loading."""
        page = await self.ensure_page()

        try:
            normalized_direction = direction.lower().strip()
            signed_distance = abs(int(distance or 800))
            if normalized_direction in {"up", "backward"}:
                signed_distance = -signed_distance

            await page.evaluate(
                """({ distance }) => {
                    window.scrollBy({ top: distance, left: 0, behavior: "instant" });
                }""",
                {"distance": signed_distance},
            )

            await self._smart_wait(page)

            try:
                title = await page.title()
            except Exception:
                title = None

            page_info = PageInfo(
                url=page.url,
                title=title,
                viewport=DEFAULT_VIEWPORT,
                is_loading=False,
                last_action_status="success",
                error_message=None,
            )
            return page_info
        except PlaywrightTimeoutError as exc:
            logger.warning("scroll timed out: %s", exc)
            return PageInfo(
                url=page.url or "",
                title=None,
                viewport=DEFAULT_VIEWPORT,
                is_loading=False,
                last_action_status="timeout",
                error_message=f"Scroll operation timed out: {exc}",
            )
        except Exception as exc:
            can_retry = self._is_recoverable_browser_error(exc)
            if can_retry:
                logger.warning("scroll failed due to closed page, retrying once")
                page = await self._recover_browser_session("scroll")
                return await self.scroll(direction, distance)
            logger.exception("scroll failed", exc_info=True)
            return PageInfo(
                url=page.url or "",
                title=None,
                viewport=DEFAULT_VIEWPORT,
                is_loading=False,
                last_action_status="fail",
                error_message=f"Scroll failed: {type(exc).__name__}: {exc}",
            )

    async def find_elements(self, selector: str) -> List[QueryMatch]:
        """Return text and attributes for elements matching a CSS selector."""
        page = await self.ensure_page()
        try:
            elements = await page.query_selector_all(selector)
            results: List[QueryMatch] = []
            for element in elements[:MAX_FIND_RESULTS]:
                text = await element.inner_text()
                # Use single triple-quotes for JavaScript strings
                attributes = await element.evaluate(
                    """el => {
                        const obj = {};
                        for (const attr of el.attributes) obj[attr.name] = attr.value;
                        return obj;
                    }"""
                )
                rect = await element.bounding_box() or {"x": 0, "y": 0, "width": 0, "height": 0}
                results.append(QueryMatch(
                    id=str(uuid.uuid4()),
                    selector=selector,
                    tag_name=element.tagName,
                    text=text,
                    attributes=attributes or {},
                    rect=rect,
                    is_visible=True,
                    is_enabled=True
                ))

            return results
        except Exception as exc:
            logger.exception("find_elements failed for selector=%s", selector, exc_info=True)
            return []

    async def extract_ui(self, limit: int = 50) -> List[QueryMatch]:
        """Public UI-structure extractor for LLM planning."""
        page = await self.ensure_page()
        ui_payload = await self._extract_ui_from_page(page, limit=limit)

        results: List[QueryMatch] = []
        for el in ui_payload.get("elements", []):
            if isinstance(el, dict):
                rect = el.get("rect") or {"x": 0, "y": 0, "width": 0, "height": 0}
                results.append(QueryMatch(
                    id=str(el.get("id")),
                    selector=el.get("selector", ""),
                    tag_name=el.get("type", ""),
                    text=el.get("text", ""),
                    attributes=el.get("attributes", {}),
                    rect=rect,
                    is_visible=True,
                    is_enabled=not el.get("disabled", False)
                ))

        return results

    async def get_screenshot(self, *, full_page: bool = False) -> str:
        """Capture a screenshot for OCR/inspection and return the absolute file path."""
        page = await self.ensure_page()

        try:
            screenshot_path = await self._take_screenshot(page, prefix="screenshot", full_page=full_page)
            return str(screenshot_path)
        except Exception as exc:
            logger.exception("Screenshot failed", exc_info=True)
            raise

    async def inspect_element_property(self, selector: str, property_name: str) -> Dict[str, Any]:
        """Inspect element property/attribute for decision support."""
        page = await self.ensure_page()
        element = await page.query_selector(selector)

        if element is None:
            return {
                "error": "element_not_found",
                "selector": selector,
                "property": property_name,
            }

        try:
            value = await element.evaluate(
                """(el, propertyName) => {
                    if (propertyName in el) return el[propertyName];
                    return el.getAttribute(propertyName);
                }""",
                property_name,
            )

            return {
                "value": value,
                "selector": selector,
                "property": property_name,
            }
        except Exception as exc:
            logger.debug("Failed to inspect property %s for selector %s: %s",
                property_name, selector, str(exc))
            return {
                "error": str(exc),
                "selector": selector,
                "property": property_name,
            }

    async def get_state_snapshot(self) -> PageInfo:
        """Return current page state for middleware/LLM planning."""
        if self._page is None or self._page_is_closed(self._page):
            return PageInfo(
                url="",
                title=None,
                viewport=DEFAULT_VIEWPORT,
                is_loading=False,
                last_action_status="unknown",
                error_message="No active page",
            )

        try:
            current_url = self._page.url
            current_title = await self._page.title()
            return PageInfo(
                url=current_url,
                title=current_title,
                viewport=DEFAULT_VIEWPORT,
                is_loading=False,
                last_action_status="success",
                error_message=None,
            )
        except Exception as exc:
            logger.debug("Failed to get current page state: %s", exc)
            return PageInfo(
                url="",
                title=None,
                viewport=DEFAULT_VIEWPORT,
                is_loading=False,
                last_action_status="fail",
                error_message=str(exc),
            )

    # --- Internal Action Implementations ---

    async def _extract_ui_from_page(self, page: Page, *, limit: int = 12) -> Dict[str, Any]:
        """Extract navigation-oriented actionable UI elements from a concrete page."""
        return await page.evaluate(f""" ({{ limit }}) => {{

            function getText(el) {{
                return (
                    (el.innerText ||
                    el.value ||
                    el.getAttribute("aria-label") ||
                    el.placeholder ||
                    el.title ||
                    "")
                ).trim();
            }}

            function isVisible(el) {{
                const style = window.getComputedStyle(el);
                return (
                    el.offsetParent !== null &&
                    style.visibility !== "hidden" &&
                    style.display !== "none"
                );
            }}

            function getSelector(el) {{
                if (el.id) return `#${el.id}`;
                if (el.name) return `[name="${el.name}"]`;

                const parent = el.parentElement;
                if (parent) {{
                    const siblings = Array.from(parent.children);
                    const index = siblings.indexOf(el) + 1;
                    return `${el.tagName.toLowerCase()}:nth-child(${index})`;
                }}
                return el.tagName.toLowerCase();
            }}

            function getRole(el) {{
                const role = el.getAttribute("role") || "";
                if (role) return role;

                const tag = el.tagName.toLowerCase();

                if (tag === "a") return "link";
                if (tag === "button") return "button";
                if (tag === "input") {{
                    if (el.type === "search") return "search";
                    if (el.type === "text") return "input";
                }}

                return "";
            }}

            function isDisabled(el) {{
                return Boolean(
                    el.disabled ||
                    el.getAttribute("aria-disabled") === "true" ||
                    el.getAttribute("aria-busy") === "true"
                );
            }}

            function isSearch(el) {{
                return (
                    el.tagName === "INPUT" &&
                    (
                        el.type === "search" ||
                        (el.placeholder || "").toLowerCase().includes("search")
                    )
                );
            }}

            function isNavLike(el) {{
                const tag = el.tagName.toLowerCase();
                const role = el.getAttribute("role") || "";

                return (
                    tag === "nav" ||
                    tag === "aside" ||
                    tag === "header" ||
                    role.includes("navigation") ||
                    role === "tablist"
                );
            }}

            function isTab(el) {{
                return (
                    el.getAttribute("role") === "tab" ||
                    el.getAttribute("aria-selected") !== null
                );
            }}

            function isLink(el) {{
                return el.tagName === "A" && el.href;
            }}

            function isNavButton(el) {{
                const txt = getText(el).toLowerCase();
                return (
                    el.tagName === "BUTTON" &&
                    (
                        txt.includes("login") ||
                        txt.includes("sign") ||
                        txt.includes("menu") ||
                        txt.includes("next") ||
                        txt.includes("back")
                    )
                );
            }}

            // ---- 主逻辑 ----

            const all = Array.from(document.querySelectorAll("*"));

            const candidates = all.filter(el => {{
                if (!isVisible(el)) return false;

                return (
                    isSearch(el) ||
                    isTab(el) ||
                    isLink(el) ||
                    isNavButton(el) ||
                    isNavLike(el)
                );
            }});

            const raw = candidates.map(el => {{
                const rect = el.getBoundingClientRect();

                return {{
                    type: el.tagName.toLowerCase(),
                    role: getRole(el),
                    text: getText(el).slice(0, 60),
                    ariaLabel: (el.getAttribute("aria-label") || "").slice(0, 60),
                    placeholder: (el.placeholder || "").slice(0, 40),
                    selector: (getSelector(el) || "").slice(0, 80),
                    inputType: (el.type || "").slice(0, 20),
                    href: el.href || "",
                    disabled: isDisabled(el),
                    y: rect.top,
                    area: rect.width * rect.height
                }};
            }})
            // 过滤垃圾
            .filter(el =>
                el.selector &&
                (
                    el.text.length > 0 ||
                    el.placeholder.length > 0 ||
                    el.href
                )
            )
            // 排序：优先顶部 + 大区域（导航栏）
            .sort((a, b) => {{
                if (Math.abs(a.y - b.y) < 50) {{
                    return b.area - a.area;
                }}
                return a.y - b.y;
            }})
            .slice(0, Number(limit) || 12);

            return {{
                title: document.title,
                url: location.href,
                elements: raw.map((el, i) => ({{
                    id: i + 1,
                    type: el.type,
                    role: el.role,
                    text: el.text,
                    aria_label: el.ariaLabel,
                    placeholder: el.placeholder,
                    selector: el.selector,
                    input_type: el.inputType,
                    disabled: el.disabled,
                    href: el.href,
                    action: inferAction(el),
                    y: el.y
                }}))
            }};

            // ---- 行为推断 ----
            function inferAction(el) {{
                if (el.role === "search") return "search";
                if (el.role === "tab") return "switch_tab";
                if (el.type === "a") return "navigate";
                if (el.type === "button") return "click";
                return "interact";
            }}

        }}""", {"limit": limit})

    async def _take_screenshot(self, page: Page, *, prefix: str, full_page: bool = False) -> Path:
        """Take a screenshot and save to storage."""
        file_path = self.storage_dir / f"{prefix}_{uuid.uuid4().hex[:10]}.png"
        await page.screenshot(path=str(file_path), full_page=full_page)
        logger.info("Screenshot saved to %s", file_path)
        return file_path

    async def _smart_wait(self, page: Page, delay: int = DEFAULT_CAPTURE_DELAY_MS) -> None:
        """Wait for network stability, then add a small human-like delay."""
        await self._wait_for_network_idle(page, timeout_ms=DEFAULT_SMART_WAIT_TIMEOUT_MS)
        await page.wait_for_timeout(delay)

    async def _wait_for_network_idle(self, page: Page, *, timeout_ms: int) -> bool:
        """Wait for network to be idle, return True if successful."""
        try:
            await page.wait_for_load_state("networkidle", timeout=timeout_ms)
            return True
        except PlaywrightTimeoutError:
            logger.debug("networkidle wait timed out; continuing with fallback delay")
            return False

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
        if self._page is not None:
            await self._page.wait_for_timeout(random.randint(min_ms, max_ms))

    async def wait_for_selector(self, selector: str, timeout: int = 5000) -> Optional[QueryMatch]:
        """Wait for element to appear in DOM and become visible.

        Returns QueryMatch or None.
        """
        page = await self.ensure_page()

        try:
            # 等待元素出现并可见
            element = await page.wait_for_selector(
                selector,
                state="visible",
                timeout=timeout
            )

            if element is not None:
                text = await element.inner_text()
                attributes = await element.evaluate(
                    """el => {
                        const obj = {};
                        for (const attr of el.attributes) obj[attr.name] = attr.value;
                        return obj;
                    }"""
                )
                rect = await element.bounding_box() or {"x": 0, "y": 0, "width": 0, "height": 0}

            return QueryMatch(
                id=str(uuid.uuid4()),
                selector=selector,
                tag_name=element.tagName if element else "unknown",
                text=text if element else "",
                attributes=attributes or {},
                rect=rect,
                is_visible=True,
                is_enabled=True
            )
        except PlaywrightTimeoutError:
            logger.debug("wait_for_selector timed out for selector=%s", selector)
            return None
        except Exception as exc:
            logger.debug("wait_for_selector failed for selector=%s: %s", selector, str(exc))
            return None

    def _attach_page_audit_hooks(self, page: Page) -> None:
        """Attach console/network listeners once per page for 异常审计."""
        # 异常审计为"被动监控",在动作无响应时给 LLM 提供排障线索。
        if getattr(page, "__agi_audit_hooked__", False):
            return

        def _on_console(message: Any) -> None:
            try:
                if str(getattr(message, "type", "")) == "error":
                    self._recent_console_errors.append(
                        {"type": "console_error", "text": str(getattr(message, "text", "")), "url": page.url}
                    )
            except Exception:
                logger.debug("console hook parse failed", exc_info=True)

        def _on_request_failed(request: Any) -> None:
            try:
                self._recent_request_failures.append(
                    {"type": "request_failed", "url": str(getattr(request, "url", "")), "method": str(getattr(request, "method", ""))}
                )
            except Exception:
                logger.debug("requestfailed hook parse failed", exc_info=True)

        page.on("console", _on_console)
        page.on("requestfailed", _on_request_failed)
        setattr(page, "__agi_audit_hooked__", True)
