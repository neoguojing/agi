import json
import logging
import random
import uuid
from asyncio import Lock, Queue, Task
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager, suppress
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

# 默认浏览器配置：尽量模拟真实桌面浏览器，减少被网站识别为自动化环境的概率。
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
MAX_FIND_RESULTS = 5
MAX_BROWSER_EVENTS = 1
MAX_STATE_MESSAGES = 1
# 持久化的两个核心文件：一个保存 agent 可消费的状态摘要，一个保存 Playwright 的 cookies/localStorage。
STATE_SNAPSHOT_FILENAME = "browser_session_state.json"
PLAYWRIGHT_STORAGE_STATE_FILENAME = "playwright_storage_state.json"
# 用户直接与页面交互产生的事件类型。
USER_EVENT_TYPES = {"dom_click", "dom_input", "dom_change", "dom_submit", "page_hashchange", "page_popstate", "page_focusin"}
# 注入到页面中的监听脚本：把用户在浏览器里的真实操作同步回 Python 后端。
BROWSER_OBSERVER_SCRIPT = """
(() => {
  if (window.__agiBrowserObserverInstalled) return;
  window.__agiBrowserObserverInstalled = true;

  const buildTarget = (target) => {
    if (!(target instanceof Element)) {
      return { tag: null, id: null, classes: [], text: "" };
    }
    return {
      tag: target.tagName ? target.tagName.toLowerCase() : null,
      id: target.id || null,
      name: target.getAttribute?.("name") || null,
      type: target.getAttribute?.("type") || null,
      classes: Array.from(target.classList || []),
      text: (target.innerText || target.textContent || "").trim().slice(0, 120),
      value: ["INPUT", "TEXTAREA", "SELECT"].includes(target.tagName) ? String(target.value || "").slice(0, 120) : null,
    };
  };

  const emit = (type, extra = {}) => {
    const payload = {
      type,
      url: window.location.href,
      title: document.title,
      timestamp: new Date().toISOString(),
      ...extra,
    };
    if (window.__agiRecordBrowserEvent) {
      window.__agiRecordBrowserEvent(payload).catch(() => undefined);
    }
  };

  document.addEventListener("click", (event) => emit("dom_click", { target: buildTarget(event.target) }), true);
  document.addEventListener("input", (event) => emit("dom_input", { target: buildTarget(event.target) }), true);
  document.addEventListener("change", (event) => emit("dom_change", { target: buildTarget(event.target) }), true);
  document.addEventListener("submit", (event) => emit("dom_submit", { target: buildTarget(event.target) }), true);
  document.addEventListener("focusin", (event) => emit("page_focusin", { target: buildTarget(event.target) }), true);
  window.addEventListener("hashchange", () => emit("page_hashchange"), true);
  window.addEventListener("popstate", () => emit("page_popstate"), true);
})();
"""


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


@dataclass(slots=True)
class UserBrowserSession:
    """Browser session state scoped to a single user."""

    # 这里保存的是“会话级”信息：当前用户对应的 backend、最近一次页面结果、活跃操作数等。

    user_id: str
    backend: "StatefulBrowserBackend"
    last_result: PageInfo | None = None
    active_operations: int = 0
    last_used_at: float = field(default_factory=lambda: 0.0)
    idle_task: Task[None] | None = None
    operation_lock: Lock = field(default_factory=Lock)


class BrowserBackendPool:
    """Manage one browser backend per user with idle-time eviction."""

    # middleware 只需要关心“给我某个 user_id 的会话”；具体创建/复用/空闲回收由这里统一管理。

    def __init__(
        self,
        storage_dir: str,
        *,
        idle_timeout_seconds: float = 60.0*30.0,
        headless: bool = False,
        timeout: int = 30_000,
        max_content_length: int = 2_000_000,
        max_retry: int = 2,
    ) -> None:
        self.storage_dir = Path(storage_dir).resolve()
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        # 每个用户都有独立目录，用于保存截图、状态快照、storage_state。
        self.idle_timeout_seconds = idle_timeout_seconds
        self.headless = headless
        self.timeout = timeout
        self.max_content_length = max_content_length
        self.max_retry = max_retry
        self._lock = Lock()
        self._sessions: dict[str, UserBrowserSession] = {}

    @asynccontextmanager
    async def session(self, user_id: str) -> AsyncIterator[UserBrowserSession]:
        """Yield the active session for a user and refresh its idle timer."""
        session = await self._acquire_session(user_id)
        async with session.operation_lock:
            try:
                yield session
            finally:
                await self._release_session(user_id)

    def get_existing_session(self, user_id: str) -> UserBrowserSession | None:
        """Return the current session for a user without creating a new browser."""
        return self._sessions.get(user_id)

    async def close_all(self) -> None:
        """Close every managed browser session."""
        async with self._lock:
            sessions = list(self._sessions.items())
            self._sessions.clear()

        for _user_id, session in sessions:
            if session.idle_task is not None:
                session.idle_task.cancel()
                with suppress(BaseException):
                    await session.idle_task
            await session.backend.close()

    async def _acquire_session(self, user_id: str) -> UserBrowserSession:
        async with self._lock:
            session = self._sessions.get(user_id)
            if session is None:
                session = UserBrowserSession(
                    user_id=user_id,
                    backend=StatefulBrowserBackend(
                        storage_dir=str(self.storage_dir / self._sanitize_user_id(user_id)),
                        headless=self.headless,
                        timeout=self.timeout,
                        max_content_length=self.max_content_length,
                        max_retry=self.max_retry,
                    ),
                )
                self._sessions[user_id] = session
                logger.info("Created browser session for user_id=%s", user_id)

            if session.idle_task is not None:
                session.idle_task.cancel()
                session.idle_task = None

            loop = session.backend._get_running_loop()
            session.active_operations += 1
            session.last_used_at = loop.time()
            return session

    async def _release_session(self, user_id: str) -> None:
        async with self._lock:
            session = self._sessions.get(user_id)
            if session is None:
                return

            session.active_operations = max(0, session.active_operations - 1)
            loop = session.backend._get_running_loop()
            session.last_used_at = loop.time()
            if session.active_operations == 0:
                session.idle_task = loop.create_task(self._close_when_idle(user_id))

    async def _close_when_idle(self, user_id: str) -> None:
        try:
            await self._sleep(self.idle_timeout_seconds)
            async with self._lock:
                session = self._sessions.get(user_id)
                if session is None or session.active_operations > 0:
                    return

                loop = session.backend._get_running_loop()
                if loop.time() - session.last_used_at < self.idle_timeout_seconds:
                    session.idle_task = loop.create_task(self._close_when_idle(user_id))
                    return

                self._sessions.pop(user_id, None)
            logger.info("Closing idle browser session for user_id=%s", user_id)
            await session.backend.close()
        except Exception:
            logger.exception("Failed to close idle browser session for user_id=%s", user_id)

    async def _sleep(self, seconds: float) -> None:
        import asyncio

        await asyncio.sleep(seconds)

    def _sanitize_user_id(self, user_id: str) -> str:
        return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in user_id) or "default"


class StatefulBrowserBackend:
    """Stateful Playwright backend for browser automation."""

    # backend 负责两件事：
    # 1) 提供 navigate / click / fill / screenshot 等原子操作；
    # 2) 监听并汇总浏览器、context、page、用户交互产生的状态变化。

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
        # 每个用户都有独立目录，用于保存截图、状态快照、storage_state。
        self._state_snapshot_path = self.storage_dir / STATE_SNAPSHOT_FILENAME
        self._playwright_storage_state_path = self.storage_dir / PLAYWRIGHT_STORAGE_STATE_FILENAME

        self._init_lock = Lock()
        self._playwright = None
        self._browser: Browser | None = None
        self._context: BrowserContext | None = None
        self._page: Page | None = None
        self._history: list[dict[str, Any]] = []
        self._events: list[dict[str, Any]] = []
        self._state_message_queue: Queue[dict[str, Any]] = Queue()
        self._state_messages: list[dict[str, Any]] = []
        self._event_seq = 0
        self._active_page_id: str | None = None
        self._page_titles: dict[str, str | None] = {}
        self._page_runtime_state: dict[str, dict[str, Any]] = {}
        self._instrumented_pages: set[str] = set()
        self._binding_registered = False
        self._restored_state_snapshot = self._load_persisted_state_snapshot()

        self.min_text_length = 50
        self.min_html_length = 100
        self.ocr_keywords = ["captcha", "验证", "blocked"]

    def _get_running_loop(self):
        import asyncio

        return asyncio.get_running_loop()

    async def initialize(self) -> None:
        """Initialize the shared browser session lazily."""
        # 懒初始化：只有真正第一次用到浏览器时才启动 Playwright。
        async with self._init_lock:
            if self._browser is not None:
                return

            logger.info("Launching browser backend")
            self._playwright = await async_playwright().start()
            self._browser = await self._playwright.chromium.launch(
                headless=self.headless,
                args=["--no-sandbox", "--disable-setuid-sandbox"],
                executable_path="/usr/bin/google-chrome",
            )
            # 如果之前持久化过 storage_state，则在新建 context 时恢复 cookies/localStorage。
            context_kwargs: dict[str, Any] = {
                "viewport": DEFAULT_VIEWPORT,
                "user_agent": DEFAULT_USER_AGENT,
            }
            if self._playwright_storage_state_path.exists():
                context_kwargs["storage_state"] = str(self._playwright_storage_state_path)
            self._context = await self._browser.new_context(**context_kwargs)
            await self._register_context_instrumentation(self._context)
            self._page = await self._context.new_page()
            self._set_active_page(self._page)
            await self._instrument_page(self._page, source="initialize")
            logger.info("Browser backend ready")

    async def close(self) -> None:
        """Close the browser session and release Playwright resources."""
        self._record_event(
            "browser_closed",
            page=self._page,
            metadata={"storage_dir": str(self.storage_dir)},
        )
        try:
            await self._persist_playwright_storage_state()
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
            self._active_page_id = None
            self._binding_registered = False
            self._page_runtime_state.clear()
            self._state_messages.clear()
            self._instrumented_pages.clear()
            logger.info("Browser backend closed")

    @property
    def is_closed(self) -> bool:
        return self._browser is None or self._context is None or self._page is None or self._page_is_closed(self._page)

    async def ensure_page(self) -> Page:
        """Return the active page, initializing the backend when needed."""
        if self._page is None or self._page_is_closed(self._page):
            await self.initialize()
            if self._context is not None:
                live_pages = [page for page in self._context_pages() if not self._page_is_closed(page)]
                if live_pages:
                    self._page = live_pages[-1]
                    self._set_active_page(self._page)
                    await self._instrument_page(self._page, source="ensure_page")
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
            self._record_event(
                "action_click",
                page=page,
                metadata={"selector": selector},
            )
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
            self._record_event(
                "action_click_text",
                page=page,
                metadata={"text": text},
            )
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
            self._record_event(
                "action_fill",
                page=page,
                metadata={"selector": selector, "value": value},
            )
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
            self._record_event(
                "action_fill_label",
                page=page,
                metadata={"label": label_text, "value": value},
            )
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
            self._record_event(
                "action_fill_human_like",
                page=page,
                metadata={"selector": selector, "value": value},
            )
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
        self._set_active_page(page)
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
            self._record_event(
                "query_selector_all",
                page=page,
                metadata={"selector": selector, "count": len(results)},
            )
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

    def get_history(self) -> list[dict[str, Any]]:
        """Return a copy of the recorded browser action history."""
        return list(self._history)

    def peek_state_messages(self, limit: int = 1) -> list[dict[str, Any]]:
        """Return recently published state messages without consuming them."""
        if limit <= 0:
            return []
        return [dict(message) for message in self._state_messages[-limit:]]

    def drain_state_messages(self, limit: int = 1) -> list[dict[str, Any]]:
        """Drain pending state messages for downstream synchronizers."""
        messages: list[dict[str, Any]] = []
        while limit > 0 and not self._state_message_queue.empty():
            messages.append(self._state_message_queue.get_nowait())
            limit -= 1
        return messages

    def get_recent_events(self, limit: int = 5) -> list[dict[str, Any]]:
        """Return a copy of the most recent browser/page events."""
        if limit <= 0:
            return []
        return [dict(event) for event in self._events[-limit:]]

    def get_state_snapshot_full(self, *, user_id: str | None = None, last_result: PageInfo | None = None) -> dict[str, Any]:
        """Return the current browser/context/page state for agent decision making."""
        # 这里返回的是 agent 侧真正需要消费的统一状态视图：
        # - browser: 浏览器是否打开/是否从持久化恢复
        # - context: 当前 tab 列表、激活页、最近用户事件
        # - page: 当前页面 URL / title / load 状态 / 最近交互
        restored = self._restored_state_snapshot or {}
        restored_context = restored.get("context", {}) if isinstance(restored.get("context"), dict) else {}
        restored_page = restored.get("page", {}) if isinstance(restored.get("page"), dict) else {}
        restored_browser = restored.get("browser", {}) if isinstance(restored.get("browser"), dict) else {}

        pages = self._context_pages()
        active_page = self._page
        active_page_id = self._page_id(active_page) if active_page is not None else restored_context.get("active_page_id") or self._active_page_id
        last_event = self._events[-1] if self._events else restored.get("last_event")
        last_page_result = last_result or None

        active_runtime = self._page_runtime_state.get(active_page_id or "", {})
        recent_events = self.get_recent_events() or list(restored.get("recent_events", []))
        recent_user_events = [event for event in recent_events if event.get("type") in USER_EVENT_TYPES][-10:]
        state_messages = self.peek_state_messages() or list(restored.get("state_messages", []))
        context_pages = [self._page_snapshot(page) for page in pages] or list(restored_context.get("pages", []))

        snapshot = {
            "user_id": user_id or restored.get("user_id"),
            "storage_dir": str(self.storage_dir),
            "browser": {
                "is_open": self._browser is not None and not self.is_closed,
                "is_closed": self.is_closed,
                "headless": self.headless,
                "timeout_ms": self.timeout,
                "was_restored": bool(restored),
                "last_persisted_open": restored_browser.get("is_open"),
            },
            "context": {
                "is_initialized": self._context is not None,
                "page_count": len(context_pages),
                "active_page_id": active_page_id,
                "pages": context_pages,
                "recent_user_events": recent_user_events,
            },
            "page": {
                "active_page_id": active_page_id,
                "url": getattr(active_page, "url", None) or restored_page.get("url"),
                "title": self._page_titles.get(active_page_id) if active_page_id and self._page_titles.get(active_page_id) is not None else restored_page.get("title"),
                "is_closed": self._page_is_closed(active_page) if active_page is not None else restored_page.get("is_closed", True),
                "load_state": self._infer_load_state(last_event),
                "last_result_url": last_page_result.url if last_page_result else restored_page.get("last_result_url"),
                "last_result_title": last_page_result.title if last_page_result else restored_page.get("last_result_title"),
                "has_screenshot": bool(last_page_result and last_page_result.screenshot_path) or restored_page.get("has_screenshot", False),
                "last_interaction": active_runtime.get("last_interaction") or restored_page.get("last_interaction"),
                "last_user_event": active_runtime.get("last_user_event") or restored_page.get("last_user_event"),
                "user_interaction_count": active_runtime.get("user_interaction_count", restored_page.get("user_interaction_count", 0)),
                "observed_title": active_runtime.get("title") or restored_page.get("observed_title"),
                "observed_url": active_runtime.get("url") or restored_page.get("observed_url"),
            },
            "history_length": len(self._history) or restored.get("history_length", 0),
            "recent_events": recent_events,
            "last_event": dict(last_event) if isinstance(last_event, dict) else None,
            "event_version": self._event_seq or restored.get("event_version", 0),
            "state_messages": state_messages,
        }
        
        return snapshot
    
    def get_state_snapshot(self, *, user_id: str | None = None, last_result: PageInfo | None = None) -> dict[str, Any]:
        """Return the current browser/context/page state for agent decision making."""
        restored = self._restored_state_snapshot or {}
        restored_context = restored.get("context", {}) if isinstance(restored.get("context"), dict) else {}
        restored_page = restored.get("page", {}) if isinstance(restored.get("page"), dict) else {}

        pages = self._context_pages()
        active_page = self._page
        active_page_id = self._page_id(active_page) if active_page is not None else restored_context.get("active_page_id") or self._active_page_id
        last_event = self._events[-1] if self._events else restored.get("last_event")
        last_page_result = last_result or None

        active_runtime = self._page_runtime_state.get(active_page_id or "", {})
        recent_events = self.get_recent_events() or list(restored.get("recent_events", []))

        # ✅ 只保留最近事件（防止上下文爆炸）
        recent_events = recent_events[-1:]

        snapshot = {
            "user_id": user_id or restored.get("user_id"),
            "storage_dir": str(self.storage_dir),
            "browser": {
                # ✅ 只保留可用性判断
                "is_open": self._browser is not None and not self.is_closed,
            },

            "context": {
                # ✅ 仅保留必要上下文
                "page_count": len(pages),
                "active_page_id": active_page_id,
            },

            "page": {
                "url": getattr(active_page, "url", None) or restored_page.get("url"),
                "title": (
                    self._page_titles.get(active_page_id)
                    if active_page_id and self._page_titles.get(active_page_id) is not None
                    else restored_page.get("title")
                ),
                "is_closed": (
                    self._page_is_closed(active_page)
                    if active_page is not None
                    else restored_page.get("is_closed", True)
                ),
                "load_state": self._infer_load_state(last_event),
                # AttributeError: 'tuple' object has no attribute 'url'
                # "last_result_url": last_page_result.url if last_page_result else restored_page.get("last_result_url"),
                # ✅ 行为信息（核心）
                "last_interaction": active_runtime.get("last_interaction") or restored_page.get("last_interaction"),
                "last_user_event": active_runtime.get("last_user_event") or restored_page.get("last_user_event"),
            },

            # ✅ 事件（核心）
            "recent_events": recent_events,
            "last_event": dict(last_event) if isinstance(last_event, dict) else None,

            # ✅ 显式变化（非常关键）
            # AttributeError: 'tuple' object has no attribute 'url'
            # "changes": {
            #     "url_changed": (
            #         (getattr(active_page, "url", None) or restored_page.get("url"))
            #         != (last_page_result.url if last_page_result else restored_page.get("last_result_url"))
            #     )
            # },
        }

        return snapshot

    async def _run_page_action(
        self,
        action: str,
        operation: Callable[[Page], Awaitable[Response | None]],
        *,
        capture_url: str | None,
        history_entry: dict[str, Any] | None = None,
    ) -> PageInfo:
        page = await self.ensure_page()
        self._set_active_page(page)
        last_error: Exception | None = None

        for attempt in range(self.max_retry + 1):
            try:
                logger.info("Browser action '%s' attempt %s/%s", action, attempt + 1, self.max_retry + 1)
                response = await operation(page)
                await self._smart_wait(page)
                await self._instrument_page(page, source=action)
                if history_entry is not None:
                    self._history.append(history_entry)
                await self._persist_playwright_storage_state()
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
        self._record_event(
            "page_error",
            page=self._page,
            metadata={"url": url, "error": error, **metadata},
        )
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
            # html = await page.content()
            # if len(html) > self.max_content_length:
            #     html = html[: self.max_content_length] + "\n... [TRUNCATED]"
            html_repr = await self.extract_ui(page)
            
            page_text = await page.inner_text("body")
            page_title = await page.title()
            self._page_titles[self._page_id(page)] = page_title
            take_screenshot = self._should_capture_screenshot(
                page_text=page_text,
                response=response,
            )

            screenshot_path: str | None = None
            if take_screenshot:
                screenshot_path = str(await self._take_screenshot(page, prefix="page", full_page=True))

            self._record_event(
                "page_capture",
                page=page,
                metadata={
                    "requested_url": url,
                    "status": response.status if response is not None else 200,
                    "has_screenshot": screenshot_path is not None,
                },
            )

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
                    "history_length": len(self._history),
                    # "browser_state": self.get_state_snapshot(last_result=None),
                }
            )

            return page_info
        except Exception as exc:
            logger.exception("Failed to capture page info for %s", url)
            return self._build_error_page_info(url, str(exc), action="capture")

    async def extract_ui(self,page: Page):
        return await page.evaluate("""
        () => {
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
        }
        """)

    def _should_capture_screenshot(self, *,  page_text: str, response: Response | None) -> bool:
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
        self._record_event(
            "screenshot_captured",
            page=page,
            metadata={"path": str(file_path), "full_page": full_page},
        )
        return file_path

    async def _smart_wait(self, page: Page, delay: int = DEFAULT_CAPTURE_DELAY_MS) -> None:
        """Wait for network stability, then add a small human-like delay."""
        try:
            await page.wait_for_load_state("networkidle", timeout=DEFAULT_SMART_WAIT_TIMEOUT_MS)
            self._record_event("page_load_state", page=page, metadata={"state": "networkidle"})
        except PlaywrightTimeoutError:
            logger.debug("networkidle wait timed out; continuing with fallback delay")
            self._record_event("page_load_state_timeout", page=page, metadata={"state": "networkidle"})
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

    async def _register_context_instrumentation(self, context: BrowserContext) -> None:
        # context 级监听：新 tab / popup 会从这里进入。
        if not hasattr(context, "on"):
            return
        context.on("page", self._handle_new_page)
        if not self._binding_registered and hasattr(context, "expose_binding"):
            try:
                await context.expose_binding("__agiRecordBrowserEvent", self._record_page_dom_event)
                self._binding_registered = True
            except Exception:
                logger.debug("Failed to register browser event binding", exc_info=True)
        if hasattr(context, "add_init_script"):
            try:
                await context.add_init_script(BROWSER_OBSERVER_SCRIPT)
            except Exception:
                logger.debug("Failed to install browser observer script", exc_info=True)

    async def _instrument_page(self, page: Page, *, source: str) -> None:
        page_id = self._page_id(page)
        if page_id in self._instrumented_pages:
            return
        self._instrumented_pages.add(page_id)
        self._page_titles.setdefault(page_id, None)
        self._set_active_page(page)
        self._attach_page_listeners(page)
        if hasattr(page, "add_init_script"):
            try:
                await page.add_init_script(BROWSER_OBSERVER_SCRIPT)
            except Exception:
                logger.debug("Failed to install page observer script", exc_info=True)
        self._record_event("page_instrumented", page=page, metadata={"source": source})

    def _attach_page_listeners(self, page: Page) -> None:
        if not hasattr(page, "on"):
            return
        page.on("domcontentloaded", lambda: self._record_event("page_domcontentloaded", page=page))
        page.on("load", lambda: self._record_event("page_load", page=page))
        page.on("close", lambda: self._record_event("page_closed", page=page))
        page.on("framenavigated", lambda frame: self._handle_frame_navigated(page, frame))

    async def _handle_new_page(self, page: Page) -> None:
        self._set_active_page(page)
        await self._instrument_page(page, source="context_page")
        self._record_event("context_page_created", page=page, metadata={"page_count": len(self._context_pages())})

    def _handle_frame_navigated(self, page: Page, frame: Any) -> None:
        frame_url = getattr(frame, "url", None)
        if callable(frame_url):
            try:
                frame_url = frame_url()
            except Exception:
                frame_url = None
        is_main_frame = True
        main_frame = getattr(page, "main_frame", None)
        if main_frame is not None and frame is not None:
            try:
                is_main_frame = frame == main_frame
            except Exception:
                is_main_frame = True
        if is_main_frame:
            self._set_active_page(page)
            self._record_event("page_navigated", page=page, metadata={"url": frame_url or getattr(page, 'url', None)})

    async def _record_page_dom_event(self, _source: Any, payload: dict[str, Any]) -> None:
        # 页面注入脚本把 DOM 事件回传到这里，再映射到具体 page 并进入统一事件流。
        if not isinstance(payload, dict):
            return
        url = payload.get("url")
        page = self._resolve_page_by_url(url)
        page_id = self._page_id(page)
        if payload.get("title"):
            self._page_titles[page_id] = str(payload["title"])
        self._record_event(payload.get("type", "dom_event"), page=page, metadata=payload)
        await self._persist_playwright_storage_state()

    def _record_event(self, event_type: str, *, page: Page | None = None, metadata: dict[str, Any] | None = None) -> None:
        # 所有状态变化最终都汇聚到这里：更新 recent events、runtime state、状态消息队列、磁盘快照。
        self._event_seq += 1
        page_id = self._page_id(page) if page is not None else self._active_page_id
        url = getattr(page, "url", None) if page is not None else None
        event = {
            "seq": self._event_seq,
            "type": event_type,
            "page_id": page_id,
            "url": url,
            "metadata": dict(metadata or {}),
        }
        self._events.append(event)
        self._update_page_runtime_state(event)
        self._publish_state_message(event)
        if len(self._events) > MAX_BROWSER_EVENTS:
            self._events = self._events[-MAX_BROWSER_EVENTS:]
        self._persist_state_snapshot()


    def _update_page_runtime_state(self, event: dict[str, Any]) -> None:
        page_id = event.get("page_id")
        if not page_id or page_id == "page:none":
            return

        runtime_state = self._page_runtime_state.setdefault(
            page_id,
            {
                "load_state": "idle",
                "last_event_type": None,
                "last_user_event": None,
                "last_interaction": None,
                "user_interaction_count": 0,
                "title": self._page_titles.get(page_id),
                "url": event.get("url"),
            },
        )
        event_type = str(event.get("type"))
        metadata = event.get("metadata", {}) if isinstance(event.get("metadata"), dict) else {}
        runtime_state["last_event_type"] = event_type
        runtime_state["url"] = metadata.get("url") or event.get("url") or runtime_state.get("url")
        if metadata.get("title"):
            runtime_state["title"] = metadata.get("title")

        if event_type in {"page_domcontentloaded", "page_navigated", "page_hashchange", "page_popstate"}:
            runtime_state["load_state"] = "loading"
        elif event_type in {"page_load", "page_load_state", "page_capture"}:
            runtime_state["load_state"] = "loaded"
        elif event_type in {"page_closed", "browser_closed"}:
            runtime_state["load_state"] = "closed"

        if event_type in USER_EVENT_TYPES or event_type.startswith("action_"):
            runtime_state["last_interaction"] = {
                "type": event_type,
                "url": runtime_state.get("url"),
                "target": metadata.get("target"),
                "timestamp": metadata.get("timestamp"),
            }
        if event_type in USER_EVENT_TYPES:
            runtime_state["last_user_event"] = {
                "type": event_type,
                "url": runtime_state.get("url"),
                "target": metadata.get("target"),
                "timestamp": metadata.get("timestamp"),
            }
            runtime_state["user_interaction_count"] = int(runtime_state.get("user_interaction_count", 0)) + 1

    async def _persist_playwright_storage_state(self) -> None:
        # storage_state 用于恢复 cookies/localStorage；它不是完整浏览器进程快照，但足以恢复很多登录态。
        if self._context is None or not hasattr(self._context, "storage_state"):
            return
        try:
            await self._context.storage_state(path=str(self._playwright_storage_state_path))
        except Exception:
            logger.debug("Failed to persist Playwright storage state", exc_info=True)

    def _publish_state_message(self, event: dict[str, Any]) -> None:
        message = {
            "kind": "browser_state",
            "event": dict(event),
            "event_version": self._event_seq,
        }
        self._state_messages.append(message)
        if len(self._state_messages) > MAX_STATE_MESSAGES:
            self._state_messages = self._state_messages[-MAX_STATE_MESSAGES:]
        self._state_message_queue.put_nowait(message)

    def _persist_state_snapshot(self) -> None:
        snapshot = self.get_state_snapshot()
        try:
            self._state_snapshot_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2))
            self._restored_state_snapshot = snapshot
        except Exception:
            logger.debug("Failed to persist browser state snapshot", exc_info=True)

    def _load_persisted_state_snapshot(self) -> dict[str, Any] | None:
        if not self._state_snapshot_path.exists():
            return None
        try:
            data = json.loads(self._state_snapshot_path.read_text())
            return data if isinstance(data, dict) else None
        except Exception:
            logger.debug("Failed to load persisted browser state snapshot", exc_info=True)
            return None

    def _set_active_page(self, page: Page | None) -> None:
        if page is None:
            return
        self._page = page
        self._active_page_id = self._page_id(page)
        self._record_event("active_page_changed", page=page, metadata={"page_count": len(self._context_pages())})

    def _context_pages(self) -> list[Page]:
        pages = getattr(self._context, "pages", None)
        if pages is None:
            return [self._page] if self._page is not None else []
        return list(pages)

    def _page_snapshot(self, page: Page) -> dict[str, Any]:
        page_id = self._page_id(page)
        runtime_state = self._page_runtime_state.get(page_id, {})
        return {
            "page_id": page_id,
            "url": getattr(page, "url", None),
            "title": self._page_titles.get(page_id),
            "is_active": page_id == self._active_page_id,
            "is_closed": self._page_is_closed(page),
            "load_state": runtime_state.get("load_state", "idle"),
            "last_event_type": runtime_state.get("last_event_type"),
            "last_user_event": runtime_state.get("last_user_event"),
            "last_interaction": runtime_state.get("last_interaction"),
            "user_interaction_count": runtime_state.get("user_interaction_count", 0),
        }

    def _infer_load_state(self, last_event: dict[str, Any] | None) -> str:
        if not last_event:
            return "idle"
        event_type = last_event.get("type")
        if event_type in {"page_load", "page_load_state", "page_capture"}:
            return "loaded"
        if event_type in {"page_domcontentloaded", "page_navigated"}:
            return "loading"
        if event_type in {"page_closed", "browser_closed"}:
            return "closed"
        return "idle"

    def _resolve_page_by_url(self, url: str | None) -> Page | None:
        if not url:
            return self._page
        for page in self._context_pages():
            if getattr(page, "url", None) == url:
                return page
        return self._page

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
