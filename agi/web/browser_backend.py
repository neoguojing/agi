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
    DEFAULT_SCROLL_TIMEOUT_MS, DEFAULT_SMART_WAIT_TIMEOUT_MS, DEFAULT_CAPTURE_DELAY_MS,
    DEFAULT_NETWORK_IDLE_TIMEOUT_MS,
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
        self._recent_console_errors: list[dict[str, Any]] = []
        self._recent_request_failures: list[dict[str, Any]] = []

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
        self._attach_page_audit_hooks(self._page)
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
                    self._attach_page_audit_hooks(self._page)
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
                previous_url = page.url
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
                    action=action,
                    previous_url=previous_url,
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

    async def scroll(self, direction: str = "down", distance: int = 800) -> PageInfo:
        """Scroll viewport to reveal off-screen content and trigger lazy-loading."""
        # 统一滚动参数：仅接受方向+距离，避免上层传坐标导致跨页面不稳定。
        normalized_direction = direction.lower().strip()
        signed_distance = abs(int(distance or 800))
        if normalized_direction in {"up", "backward"}:
            signed_distance = -signed_distance

        async def _operation(page: Page) -> None:
            await page.evaluate(
                """({ distance }) => {
                    window.scrollBy({ top: distance, left: 0, behavior: "instant" });
                }""",
                {"distance": signed_distance},
            )
            return None

        return await self._run_page_action(
            action="scroll",
            operation=_operation,
            history_entry={"action": "scroll", "direction": normalized_direction, "distance": signed_distance},
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

    async def inspect_element_property(self, selector: str, property_name: str) -> Dict[str, Any]:
        """Inspect element property/attribute for decision support."""
        # 属性探测原子：用于在动作前判断可交互性（如 disabled/aria-busy）。
        page = await self.ensure_page()
        element = await page.query_selector(selector)
        if element is None:
            return {"ok": False, "selector": selector, "property": property_name, "error": "element_not_found"}
        value = await element.evaluate(
            """(el, propertyName) => {
                if (propertyName in el) return el[propertyName];
                return el.getAttribute(propertyName);
            }""",
            property_name,
        )
        return {"ok": True, "selector": selector, "property": property_name, "value": value}

    async def get_environment_status(self) -> Dict[str, Any]:
        """Get URL/title and network-idle validation for closed-loop checks."""
        # 环境校验原子：显式返回 network_idle，避免“点击成功=页面已稳定”的误判。
        page = await self.ensure_page()
        url_before = page.url
        network_idle = await self._wait_for_network_idle(page, timeout_ms=DEFAULT_NETWORK_IDLE_TIMEOUT_MS)
        current_url = page.url
        return {
            "url": current_url,
            "title": await page.title(),
            "network_idle": network_idle,
            "url_changed": current_url != url_before,
            "console_errors": list(self._recent_console_errors[-10:]),
            "request_failures": list(self._recent_request_failures[-10:]),
        }

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
        metadata = dict(source.metadata)
        compact_metadata = {
            "load_state": metadata.get("load_state"),
            "history_length": metadata.get("history_length"),
            "error": metadata.get("error"),
            "empty_page": metadata.get("empty_page", False),
            "text_length": metadata.get("text_length", 0),
            "html_length": metadata.get("html_length", 0),
            "has_screenshot": bool(source.screenshot_path),
        }
        return {
            "url": source.url,
            "title": source.title,
            # Snapshot for middleware should stay compact; keep raw page content in last_result only.
            "html": None,
            "text": None,
            "screenshot_path": source.screenshot_path,
            "status": source.status,
            "action": source.action,
            "actionable_elements": list(source.actionable_elements),
            "environment": dict(source.environment),
            "diagnostics": dict(source.diagnostics),
            "metadata": compact_metadata,
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

    # --- Internal Action Implementations ---

    async def _capture_page_info(
        self,
        page: Page,
        url: str,
        response: Response | None,
        capture_content: bool = True,
        action: str | None = None,
        previous_url: str | None = None,
    ) -> PageInfo:
        """Capture normalized page metadata after an action completes."""
        try:
            # 语义感知：输出精简后的可操作元素，而不是完整 DOM。
            html_repr = await self.extract_ui(page, limit=8)
            actionable_elements = self._normalize_actionable_elements(
                html_repr.get("elements", []) if isinstance(html_repr, dict) else []
            )
            
            page_text = await page.inner_text("body")
            page_title = await page.title()
            normalized_text = page_text.strip()
            normalized_html = html_repr if isinstance(html_repr, str) else json.dumps(html_repr, ensure_ascii=False)
            text_is_empty = len(normalized_text) == 0
            html_is_empty = len(normalized_html.strip()) == 0

            # 视觉捕获：动作结束后优先记录截图路径，供多模态对齐/回放。
            screenshot_path = str(await self._take_screenshot(page, prefix="page", full_page=True)) if capture_content else None
            # 闭环反馈：每次动作后都附带 URL/title/network_idle。
            env_status = await self._capture_environment_feedback(page, action=action, previous_url=previous_url)

            page_info = PageInfo(
                url=page.url,
                title=page_title,
                html=normalized_html[: self.max_content_length],
                # text=normalized_text[: self.max_content_length],
                text="",
                screenshot_path=screenshot_path,
                status=response.status if response is not None else 200,
                action=action,
                actionable_elements=actionable_elements,
                environment=env_status,
                diagnostics={
                    "console_errors": list(self._recent_console_errors[-10:]),
                    "request_failures": list(self._recent_request_failures[-10:]),
                },
                metadata={
                    "html_length": len(normalized_html),
                    "text_length": len(normalized_text),
                    "has_screenshot": screenshot_path is not None,
                    "history_length": len(self._history),
                    "empty_page": text_is_empty and html_is_empty,
                    "text_truncated": len(normalized_text) > self.max_content_length,
                    "html_truncated": len(normalized_html) > self.max_content_length,
                }
            )
            self._page_runtime_state[self._page_id(page)] = page_info
            logger.info("Captured page info: url=%s title=%s", page_info.url, page_info.title)

            return page_info
        except Exception as exc:
            logger.exception("Failed to capture page info for %s", url)
            return self._build_error_page_info(url, str(exc), action="capture")

    async def extract_ui(self, page: Page, *, limit: int = 12):
        """Extract navigation-oriented actionable UI elements."""
        return await page.evaluate(""" ({ limit }) => {

            function getText(el) {
                return (
                    el.innerText ||
                    el.value ||
                    el.getAttribute("aria-label") ||
                    el.placeholder ||
                    el.title ||
                    ""
                ).trim();
            }

            function isVisible(el) {
                const style = window.getComputedStyle(el);
                return (
                    el.offsetParent !== null &&
                    style.visibility !== "hidden" &&
                    style.display !== "none"
                );
            }

            function getSelector(el) {
                if (el.id) return `#${el.id}`;
                if (el.name) return `[name="${el.name}"]`;

                const parent = el.parentElement;
                if (parent) {
                    const siblings = Array.from(parent.children);
                    const index = siblings.indexOf(el) + 1;
                    return `${el.tagName.toLowerCase()}:nth-child(${index})`;
                }
                return el.tagName.toLowerCase();
            }

            function getRole(el) {
                const role = el.getAttribute("role") || "";
                if (role) return role;

                const tag = el.tagName.toLowerCase();

                if (tag === "a") return "link";
                if (tag === "button") return "button";
                if (tag === "input") {
                    if (el.type === "search") return "search";
                    if (el.type === "text") return "input";
                }

                return "";
            }

            function isSearch(el) {
                return (
                    el.tagName === "INPUT" &&
                    (
                        el.type === "search" ||
                        (el.placeholder || "").toLowerCase().includes("search")
                    )
                );
            }

            function isNavLike(el) {
                const tag = el.tagName.toLowerCase();
                const role = el.getAttribute("role") || "";

                return (
                    tag === "nav" ||
                    tag === "aside" ||
                    tag === "header" ||
                    role.includes("navigation") ||
                    role === "tablist"
                );
            }

            function isTab(el) {
                return (
                    el.getAttribute("role") === "tab" ||
                    el.getAttribute("aria-selected") !== null
                );
            }

            function isLink(el) {
                return el.tagName === "A" && el.href;
            }

            function isNavButton(el) {
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
            }

            // ---- 主逻辑 ----

            const all = Array.from(document.querySelectorAll("*"));

            const candidates = all.filter(el => {
                if (!isVisible(el)) return false;

                return (
                    isSearch(el) ||
                    isTab(el) ||
                    isLink(el) ||
                    isNavButton(el) ||
                    isNavLike(el)
                );
            });

            const raw = candidates.map(el => {
                const rect = el.getBoundingClientRect();

                return {
                    type: el.tagName.toLowerCase(),
                    role: getRole(el),
                    text: getText(el).slice(0, 60),
                    placeholder: (el.placeholder || "").slice(0, 40),
                    selector: (getSelector(el) || "").slice(0, 80),
                    href: el.href || "",
                    y: rect.top,
                    area: rect.width * rect.height
                };
            })
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
            .sort((a, b) => {
                if (Math.abs(a.y - b.y) < 50) {
                    return b.area - a.area;
                }
                return a.y - b.y;
            })
            .slice(0, Number(limit) || 12);

            return {
                title: document.title,
                url: location.href,
                elements: raw.map((el, i) => ({
                    id: i + 1,
                    type: el.type,
                    role: el.role,
                    text: el.text,
                    placeholder: el.placeholder,
                    selector: el.selector,
                    href: el.href,
                    action: inferAction(el),
                    y: el.y
                }))
            };

            // ---- 行为推断 ----
            function inferAction(el) {
                if (el.role === "search") return "search";
                if (el.role === "tab") return "switch_tab";
                if (el.type === "a") return "navigate";
                if (el.type === "button") return "click";
                return "interact";
            }

        } """, {"limit": limit})

    def _normalize_actionable_elements(self, elements: Any) -> list[dict[str, Any]]:
        """Normalize actionable elements from JS payload into a stable schema."""
        if not isinstance(elements, list):
            return []
        normalized: list[dict[str, Any]] = []
        for item in elements:
            if not isinstance(item, dict):
                continue
            selector = str(item.get("selector") or item.get("sel") or "").strip()
            text = str(item.get("text") or item.get("c") or "").strip()
            kind = str(item.get("type") or item.get("t") or "").strip()
            placeholder = str(item.get("placeholder") or "").strip()
            if not any((selector, text, kind, placeholder)):
                continue
            normalized.append(
                {
                    "type": kind,
                    "text": text,
                    "placeholder": placeholder,
                    "selector": selector,
                    "action": str(item.get("action") or ""),
                }
            )
        return normalized

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
        # click/fill 等动作完成后必须等待“网络空闲”确认，而不是立即判定结束。
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
            status=None,
            metadata={"error": error, **metadata},
        )

    async def _capture_environment_feedback(self, page: Page, *, action: str | None, previous_url: str | None) -> dict[str, Any]:
        # 统一动作反馈结构：供 middleware 直接透出给 LLM。
        network_idle = await self._wait_for_network_idle(page, timeout_ms=DEFAULT_NETWORK_IDLE_TIMEOUT_MS)
        return {
            "action": action or "unknown",
            "network_idle": network_idle,
            "url_changed": bool(previous_url) and previous_url != page.url,
        }

    def _attach_page_audit_hooks(self, page: Page) -> None:
        """Attach console/network listeners once per page for异常审计."""
        # 异常审计为“被动监控”，在动作无响应时给 LLM 提供排障线索。
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
