import base64
import logging
from collections.abc import Awaitable, Callable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any, NotRequired, Optional

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ContextT,
    ModelRequest,
    ModelResponse,
    ResponseT,
)
from langchain.tools import ToolRuntime
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.messages.content import create_image_block
from langchain_core.tools import BaseTool, StructuredTool
from langchain.tools.tool_node import ToolCallRequest
from langgraph.types import Command
from agi.config import BROWSER_STORAGE_PATH
from agi.utils.common import append_to_system_message
from agi.web.browser_backend import StatefulBrowserBackend
from agi.web.browser_types import (
    PageInfo,
    DEFAULT_VIEWPORT,
)
from agi.agent.prompt import get_middleware_prompt

logger = logging.getLogger(__name__)


BROWSER_NAVIGATE_TOOL_DESCRIPTION = """
Navigates the browser to a specific URL.

Key Behavior:
- Updates the current browser session (cookies, local storage, history).
- Waits for `domcontentloaded` and page stabilization.
- May trigger navigation events.

Returns:
- Page summary (URL, title, preview text)
- Recent browser events (if any)

Guidelines:
- Always use this before interacting with a new website.
- If recent events already indicate navigation to the target page, avoid redundant navigation.
"""

BROWSER_CLICK_TOOL_DESCRIPTION = """
Clicks an element on the current page using a CSS selector.

Key Behavior:
- Triggers DOM updates and possibly navigation.
- Waits for page stabilization after the click.

Returns:
- Updated page state
- Any resulting navigation or DOM change events

Guidelines:
- Use `browser_find` first if the selector is uncertain.
- If recent events already indicate the intended action occurred, do not click again.
- If a click does not cause visible changes, consider extracting or inspecting the page.
"""

BROWSER_FILL_TOOL_DESCRIPTION = """
Fills a text input field on the current page.

Key Behavior:
- Clears existing content before entering new text.
- May trigger dynamic UI updates (e.g., suggestions, validation).

Returns:
- Updated page state
- Any DOM change events

Guidelines:
- Ensure the selector targets an input-capable element.
- Avoid re-filling if the desired value is already present.
"""

BROWSER_EXTRACT_TOOL_DESCRIPTION = """
Extracts page content from the current page, prioritizing OCR.

Key Behavior:
- Captures a full-page screenshot and applies OCR.
- Falls back to DOM text and HTML if needed.

Returns:
- Extracted text content
- Metadata including truncation and storage info
- Recent browser events

Guidelines:
- Use this as the primary method to understand page content.
- If the page has recently changed (navigation or DOM update), call this again.
- Prefer this over relying on previous observations.
"""

BROWSER_FIND_TOOL_DESCRIPTION = """
Finds elements by a known CSS selector (precision lookup).

Key Behavior:
- Returns text and attributes of up to 10 matches.

Guidelines:
- Use for:
  - verifying a selector already obtained from browser_extract_ui
  - checking attribute values before click/fill
- If selector is unknown, use browser_extract_ui first.
"""

BROWSER_STATUS_TOOL_DESCRIPTION = """
Gets the current browser session status without performing any page interaction.

Key Behavior:
- Returns canonical browser session state for the active user/session.
- Does not navigate, click, or mutate page content.

Returns:
- Browser exists/open flags
- Current page summary (URL/title/viewport/status/error)
- No historical page state is retained

Guidelines:
- Use this when you need to confirm browser state before deciding next action.
- If no page is loaded yet, navigate first.
"""

BROWSER_SCROLL_TOOL_DESCRIPTION = """
Scrolls the viewport to reveal off-screen content and trigger lazy-loaded sections.
Input: direction (up/down), distance (pixels).
Returns: unified action feedback (URL/title/network-idle/screenshot/actionable elements).
"""

BROWSER_PROBE_TOOL_DESCRIPTION = """
Checks a runtime element property/attribute for decision making.
Input: selector + property_name (e.g., disabled, aria-busy).
"""

BROWSER_EXTRACT_UI_TOOL_DESCRIPTION = """
Extracts a compact actionable UI structure (AOM-style) from current page.
Input: limit of returned elements.
Returns: title/url + ranked actionable elements (id/selector/text/role/disabled) for planning click/fill.
"""


class BrowserMiddlewareState(AgentState):
    """Single middleware state schema.

    Middleware only relies on one structured browser field:
    - browser_session_state: compact snapshot from PageInfo
    """

    browser_session_state: NotRequired[Optional[dict[str, Any]]]


class BrowserMiddleware(AgentMiddleware):
    """Stateful browser middleware with filesystem-style tool wrappers.

    This middleware provides browser automation tools with:
    - Structured ToolMessage responses (identity, action, result)
    - Proper tool_call_id tracking for each tool invocation
    - Error handling and recovery
    - State persistence across sessions
    """

    state_schema = BrowserMiddlewareState

    def __init__(
        self,
        storage_dir: str = BROWSER_STORAGE_PATH,
        ocr_engine: Any | None = None,
        enable_ocr_fallback: bool = True,
        content_token_limit: int = 15_000,
        eviction_handler: Callable[[str], str] | None = None,
        system_prompt: str | None = None,
    ):
        super().__init__()
        self._storage_root = Path(storage_dir).resolve()
        self._storage_root.mkdir(parents=True, exist_ok=True)
        self._session_backends: dict[str, StatefulBrowserBackend] = {}
        self.ocr = ocr_engine
        self.enable_ocr = enable_ocr_fallback
        self.content_limit = content_token_limit
        self.eviction_handler = eviction_handler
        self._custom_system_prompt = system_prompt
        self.tools = self.get_tools()

    def get_tools(self) -> list[BaseTool]:
        """Return the browser tools exposed by this middleware."""
        return [
            self._create_navigate_tool(),
            self._create_click_tool(),
            self._create_fill_tool(),
            self._create_scroll_tool(),
            self._create_extract_tool(),
            self._create_extract_ui_tool(),
            self._create_find_tool(),
            self._create_probe_tool(),
            self._create_status_tool(),
        ]

    def wrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], ModelResponse[ResponseT]],
    ) -> ModelResponse[ResponseT]:
        system_prompt = self._custom_system_prompt or get_middleware_prompt("browser")
        browser_state_prompt = self._format_browser_state_prompt(self._extract_request_browser_state(request))
        combined_prompt = "\n\n".join(part for part in [system_prompt, browser_state_prompt] if part)
        if combined_prompt:
            request = request.override(system_message=append_to_system_message(request.system_message, combined_prompt))
        return handler(request)

    async def awrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], Awaitable[ModelResponse[ResponseT]]],
    ) -> ModelResponse[ResponseT]:
        system_prompt = self._custom_system_prompt or get_middleware_prompt("browser")
        browser_state_prompt = self._format_browser_state_prompt(self._extract_request_browser_state(request))
        combined_prompt = "\n\n".join(part for part in [system_prompt, browser_state_prompt] if part)
        if combined_prompt:
            request = request.override(system_message=append_to_system_message(request.system_message, combined_prompt))
        return await handler(request)

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        """Check the size of the tool call result and evict to filesystem if too large."""
        return handler(request)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        """(async)Check the size of the tool call result and evict to filesystem if too large."""
        return await handler(request)


    def _resolve_user_id(self, runtime: ToolRuntime[None, BrowserMiddlewareState] | None = None) -> str:
        if runtime is not None:
            context = getattr(runtime, "context", None)
            if getattr(context, "user_id", None):
                return str(context.user_id)
            config = getattr(runtime, "config", {}) or {}
            configurable = config.get("configurable", {})
            if configurable.get("user_id"):
                return str(configurable["user_id"])
        logger.warning("Browser middleware could not resolve user_id; falling back to shared 'default' user")
        return "default"

    def _resolve_runtime_key(self, runtime: ToolRuntime[None, BrowserMiddlewareState] | None = None) -> tuple[str, str]:
        """Resolve the single browser instance key for a user.

        Product semantics are intentionally one browser per user: different
        tasks from the same user share the same backend and active page.
        """
        user_id = self._resolve_user_id(runtime)
        return user_id, user_id

    def _backend_for_runtime(self, runtime: ToolRuntime[None, BrowserMiddlewareState] | None = None) -> tuple[StatefulBrowserBackend, str]:
        """Get or create a backend for the given runtime, creating storage dir as needed."""
        user_id, runtime_key = self._resolve_runtime_key(runtime)
        backend = self._session_backends.get(runtime_key)
        if backend is not None:
            return backend, runtime_key

        session_root = self._storage_root / user_id
        session_root.mkdir(parents=True, exist_ok=True)
        backend = StatefulBrowserBackend(storage_dir=str(session_root))
        self._session_backends[runtime_key] = backend
        logger.debug("Created new browser backend for user_id=%s", user_id)
        return backend, runtime_key

    def _utc_now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _extract_request_browser_state(self, request: ModelRequest[ContextT]) -> dict[str, Any] | None:
        state = getattr(request, "state", None)
        if isinstance(state, Mapping):
            browser_state = state.get("browser_session_state")
            return browser_state if isinstance(browser_state, dict) else None
        return None

    def _format_browser_state_prompt(self, state: dict[str, Any] | None) -> str:
        if not state:
            return (
                "Current browser state for this user: unknown. "
                "Call browser_status before assuming a browser instance or active page exists."
            )

        return (
            "Current browser state for this user:\n"
            f"- browser_exists: {bool(state.get('browser_exists'))}\n"
            f"- browser_open: {bool(state.get('browser_open'))}\n"
            f"- page_exists: {bool(state.get('page_exists'))}\n"
            f"- url: {state.get('url') or ''}\n"
            f"- title: {state.get('title') or ''}\n"
            f"- last_action_status: {state.get('last_action_status') or 'unknown'}\n"
            f"- error_message: {state.get('error_message') or ''}\n"
            "Use browser_status if you need to refresh this state before acting."
        )

    def _empty_browser_state(
        self,
        user_id: str,
        *,
        browser_exists: bool = False,
        browser_open: bool = False,
        last_action_status: str = "unknown",
        error_message: str | None = None,
    ) -> dict[str, Any]:
        return {
            "user_id": user_id,
            "browser_exists": browser_exists,
            "browser_open": browser_open,
            "page_exists": False,
            "url": "",
            "title": None,
            "viewport": DEFAULT_VIEWPORT,
            "is_loading": False,
            "last_action_status": last_action_status,
            "error_message": error_message,
            "updated_at": self._utc_now(),
        }

    def _state_from_page_info(
        self,
        user_id: str,
        backend: StatefulBrowserBackend,
        page_info: PageInfo,
        *,
        last_action_status: str | None = None,
        error_message: str | None = None,
    ) -> dict[str, Any]:
        status = last_action_status or page_info.last_action_status
        resolved_error = error_message if error_message is not None else page_info.error_message
        return {
            "user_id": user_id,
            "browser_exists": True,
            "browser_open": not backend.is_closed,
            "page_exists": bool(page_info.url),
            "url": page_info.url,
            "title": page_info.title,
            "viewport": page_info.viewport or DEFAULT_VIEWPORT,
            "is_loading": page_info.is_loading,
            "last_action_status": status,
            "error_message": resolved_error,
            "updated_at": self._utc_now(),
        }

    async def _state_from_backend(
        self,
        user_id: str,
        backend: StatefulBrowserBackend,
        *,
        last_action_status: str | None = None,
        error_message: str | None = None,
    ) -> dict[str, Any]:
        try:
            page_info = await backend.get_state_snapshot()
            return self._state_from_page_info(
                user_id,
                backend,
                page_info,
                last_action_status=last_action_status,
                error_message=error_message,
            )
        except Exception as exc:
            logger.debug("Failed to build browser state from backend: %s", exc)
            return self._empty_browser_state(
                user_id,
                browser_exists=True,
                browser_open=False,
                last_action_status=last_action_status or "fail",
                error_message=error_message or str(exc),
            )

    def _tool_command(
        self,
        runtime: ToolRuntime[None, BrowserMiddlewareState],
        *,
        content: str,
        state: dict[str, Any],
        name: str = "browser_action",
    ) -> Command:
        tool_message = ToolMessage(content=content, name=name, tool_call_id=runtime.tool_call_id)
        return Command(update={"browser_session_state": state, "messages": [tool_message]})

    def _truncate_content(self, content: str) -> tuple[str, bool]:
        if self.content_limit <= 0 or len(content) <= self.content_limit:
            return content, False
        return content[: self.content_limit], True

    async def _run_page_action(
        self,
        runtime: ToolRuntime[None, BrowserMiddlewareState],
        *,
        tool_name: str,
        action: str,
        operation: Callable[[StatefulBrowserBackend], Awaitable[PageInfo]],
        success_result: Callable[[PageInfo], str],
        fail_result: Callable[[PageInfo], str],
        timeout_result: Callable[[PageInfo], str],
        exception_result: Callable[[Exception], str],
    ) -> Command:
        """Run a browser action that returns PageInfo and refresh minimal state.

        This keeps mutate-style tools consistent: every success, failure,
        timeout, and unexpected exception returns a Command with the current
        browser_session_state.
        """
        user_id, _ = self._resolve_runtime_key(runtime)
        backend, _ = self._backend_for_runtime(runtime)

        try:
            page_info = await operation(backend)
        except Exception as exc:
            logger.exception("%s failed with exception", tool_name, exc_info=True)
            result = exception_result(exc)
            state = await self._state_from_backend(user_id, backend, last_action_status="fail", error_message=result)
            return self._tool_command(
                runtime,
                content=f"""Tool Identity: {tool_name}
Action: {action}
Result: Error - {result}""",
                state=state,
            )

        state = self._state_from_page_info(user_id, backend, page_info)
        if page_info.last_action_status == "fail":
            result = fail_result(page_info)
            return self._tool_command(
                runtime,
                content=f"""Tool Identity: {tool_name}
Action: {action}
Result: Error - {result}""",
                state=state,
            )

        if page_info.last_action_status == "timeout":
            result = timeout_result(page_info)
            return self._tool_command(
                runtime,
                content=f"""Tool Identity: {tool_name}
Action: {action}
Result: Timeout - {result}""",
                state=state,
            )

        return self._tool_command(
            runtime,
            content=f"""Tool Identity: {tool_name}
Action: {action}
Result: Success - {success_result(page_info)}""",
            state=state,
        )

    def _create_navigate_tool(self) -> BaseTool:
        """Create the navigate tool and refresh minimal browser state on every path."""
        async def async_navigate(
            url: Annotated[str, "URL to navigate to in the current browser session."],
            runtime: ToolRuntime[None, BrowserMiddlewareState],
        ) -> Command:
            return await self._run_page_action(
                runtime,
                tool_name="browser_navigate",
                action=f"Navigated to URL '{url}'",
                operation=lambda backend: backend.navigate(url),
                success_result=lambda page: (
                    f"Page loaded at '{page.url}' with title '{page.title}'. Browser session updated."
                ),
                fail_result=lambda page: (
                    f"{page.error_message or 'Unknown error'}. Please check if the URL is correct and accessible."
                ),
                timeout_result=lambda page: (
                    f"Navigation timed out after waiting for network idle. URL: {url}. "
                    "The page may be loading slowly or is unreachable."
                ),
                exception_result=lambda exc: (
                    f"Error navigating to {url}: Unexpected error occurred. "
                    "Please check if the URL is correct and accessible."
                ),
            )

        return StructuredTool.from_function(
            name="browser_navigate",
            description=BROWSER_NAVIGATE_TOOL_DESCRIPTION,
            coroutine=async_navigate,
        )

    def _create_click_tool(self) -> BaseTool:
        """Create the click tool and refresh minimal browser state on every path."""
        async def async_click(
            selector: Annotated[str, "CSS selector to click on the current page."],
            runtime: ToolRuntime[None, BrowserMiddlewareState],
        ) -> Command:
            return await self._run_page_action(
                runtime,
                tool_name="browser_click",
                action=f"Clicked element with CSS selector '{selector}'",
                operation=lambda backend: backend.click(selector),
                success_result=lambda page: (
                    f"Page updated at '{page.url}' with title '{page.title}'. DOM state changed."
                ),
                fail_result=lambda page: (
                    f"{page.error_message or 'Unknown error'}. The element may not be visible or clickable."
                ),
                timeout_result=lambda page: (
                    f"Click timed out. Element with selector '{selector}' may not be visible or interactive."
                ),
                exception_result=lambda exc: (
                    f"Error clicking element with selector '{selector}': Unexpected error occurred. "
                    "The element may not be visible or clickable."
                ),
            )

        return StructuredTool.from_function(
            name="browser_click",
            description=BROWSER_CLICK_TOOL_DESCRIPTION,
            coroutine=async_click,
        )

    def _create_fill_tool(self) -> BaseTool:
        """Create the fill tool and refresh minimal browser state on every path."""
        async def async_fill(
            selector: Annotated[str, "CSS selector for the input field to fill."],
            text: Annotated[str, "Text to enter into the selected field."],
            runtime: ToolRuntime[None, BrowserMiddlewareState],
        ) -> Command:
            return await self._run_page_action(
                runtime,
                tool_name="browser_fill",
                action=f"Filled input field with CSS selector '{selector}' with text '{text}'",
                operation=lambda backend: backend.fill(selector, text),
                success_result=lambda page: (
                    f"Field populated. Page updated at '{page.url}' with title '{page.title}'. DOM state changed."
                ),
                fail_result=lambda page: (
                    f"{page.error_message or 'Unknown error'}. The element may not be an input field."
                ),
                timeout_result=lambda page: (
                    f"Fill timed out. Element with selector '{selector}' may not be editable."
                ),
                exception_result=lambda exc: (
                    f"Error filling field with selector '{selector}': Unexpected error occurred. "
                    "The element may not be an input field."
                ),
            )

        return StructuredTool.from_function(
            name="browser_fill",
            description=BROWSER_FILL_TOOL_DESCRIPTION,
            coroutine=async_fill,
        )

    def _create_scroll_tool(self) -> BaseTool:
        """Create the scroll tool and refresh minimal browser state on every path."""
        async def async_scroll(
            direction: Annotated[str, "Scroll direction (up/down/left/right)."],
            distance: Annotated[int, "Scroll distance in pixels."],
            runtime: ToolRuntime[None, BrowserMiddlewareState],
        ) -> Command:
            scroll_direction = "up" if direction.lower() in ["up", "backward"] else "down"
            return await self._run_page_action(
                runtime,
                tool_name="browser_scroll",
                action=f"Scrolled viewport {scroll_direction} by {abs(distance)} pixels",
                operation=lambda backend: backend.scroll(direction, distance),
                success_result=lambda page: (
                    f"Viewport scrolled. Page updated at '{page.url}' with title '{page.title}'. DOM state changed."
                ),
                fail_result=lambda page: f"{page.error_message or 'Unknown error'}.",
                timeout_result=lambda page: "Scroll timed out.",
                exception_result=lambda exc: "Error scrolling: Unexpected error occurred.",
            )

        return StructuredTool.from_function(
            name="browser_scroll",
            description=BROWSER_SCROLL_TOOL_DESCRIPTION,
            coroutine=async_scroll,
        )

    def _create_extract_tool(self) -> BaseTool:
        """Create the extract tool and refresh minimal browser state without storing page content."""
        async def async_extract(runtime: ToolRuntime[None, BrowserMiddlewareState]) -> Command:
            user_id, _ = self._resolve_runtime_key(runtime)
            backend, _ = self._backend_for_runtime(runtime)

            try:
                page = await backend.ensure_page()
            except Exception:
                logger.exception("extract failed while ensuring page", exc_info=True)
                error = "Unable to read page content. No active page is available."
                state = await self._state_from_backend(user_id, backend, last_action_status="fail", error_message=error)
                return self._tool_command(
                    runtime,
                    content=f"""Tool Identity: browser_extract
Action: Attempted to extract page content
Result: Error - {error}""",
                    state=state,
                )

            if self.enable_ocr and self.ocr is not None:
                try:
                    screenshot_path = await backend.get_screenshot()
                    if screenshot_path:
                        image_path = Path(screenshot_path)
                        if image_path.exists():
                            image_bytes = image_path.read_bytes()
                            image_b64 = base64.b64encode(image_bytes).decode("utf-8")
                            ocr_text = await self.ocr.ainvoke([
                                HumanMessage(
                                    content="extract the image content",
                                    content_blocks=[create_image_block(base64=image_b64, mime_type="image/png")],
                                )
                            ])
                            if ocr_text:
                                text, truncated = self._truncate_content(str(ocr_text))
                                state = await self._state_from_backend(user_id, backend, last_action_status="success")
                                suffix = "\n[Content truncated]" if truncated else ""
                                return self._tool_command(
                                    runtime,
                                    content=f"""Tool Identity: browser_extract
Action: Extracted page content via OCR from screenshot at '{screenshot_path}'
Result: Success - {text}{suffix}""",
                                    state=state,
                                )
                except Exception as e:
                    logger.debug("OCR extraction failed: %s", e)

            try:
                try:
                    content = await page.inner_text("body")
                except Exception:
                    content = await page.content()
                text, truncated = self._truncate_content(content)
                state = await self._state_from_backend(user_id, backend, last_action_status="success")
                suffix = "\n[Content truncated]" if truncated else ""
                return self._tool_command(
                    runtime,
                    content=f"""Tool Identity: browser_extract
Action: Extracted page content from DOM
Result: Success - Page URL: '{page.url}'. Extracted content:\n{text}{suffix}""",
                    state=state,
                )
            except Exception as e:
                logger.debug("DOM extraction failed: %s", e)
                error = "Unable to read page content. Check if page is loaded and accessible."
                state = await self._state_from_backend(user_id, backend, last_action_status="fail", error_message=error)
                return self._tool_command(
                    runtime,
                    content=f"""Tool Identity: browser_extract
Action: Attempted to extract page content
Result: Error - {error}""",
                    state=state,
                )

        return StructuredTool.from_function(
            name="browser_extract",
            description=BROWSER_EXTRACT_TOOL_DESCRIPTION,
            coroutine=async_extract,
        )

    def _create_extract_ui_tool(self) -> BaseTool:
        """Create the extract UI tool and refresh minimal browser state without storing UI history."""
        async def async_extract_ui(
            runtime: ToolRuntime[None, BrowserMiddlewareState],
            limit: Annotated[int, "Maximum number of actionable elements to return (1-50)."] = 12,
        ) -> Command:
            user_id, _ = self._resolve_runtime_key(runtime)
            backend, _ = self._backend_for_runtime(runtime)

            try:
                normalized_limit = max(1, min(int(limit or 12), 50))
                result = await backend.extract_ui(normalized_limit)
            except Exception:
                logger.error("extract_ui failed", exc_info=True)
                error = "Unable to extract UI elements. Page may be empty or have no interactive elements."
                state = await self._state_from_backend(user_id, backend, last_action_status="fail", error_message=error)
                return self._tool_command(
                    runtime,
                    content=f"""Tool Identity: browser_extract_ui
Action: Attempted to extract UI structure from page
Result: Error - {error}""",
                    state=state,
                )

            state = await self._state_from_backend(user_id, backend, last_action_status="success")
            if not result:
                return self._tool_command(
                    runtime,
                    content="""Tool Identity: browser_extract_ui
Action: Extracted UI structure from page
Result: No actionable elements found. Page may be empty or have no interactive elements.""",
                    state=state,
                )

            try:
                lines = []
                for i, el in enumerate(result[:normalized_limit]):
                    attrs = el.attributes or {}
                    role = attrs.get("role") or attrs.get("aria-role") or ""
                    disabled = attrs.get("disabled") or attrs.get("aria-disabled") or (not el.is_enabled)
                    href = attrs.get("href") or ""
                    placeholder = attrs.get("placeholder") or ""
                    lines.append(
                        f"- Element {i + 1}: selector='{el.selector}', type='{el.tag_name}', "
                        f"role='{role}', text='{el.text[:80]}', placeholder='{placeholder}', "
                        f"disabled='{disabled}', href='{href}'"
                    )
                return self._tool_command(
                    runtime,
                    content=(
                        f"""Tool Identity: browser_extract_ui
Action: Extracted UI structure from page with limit={normalized_limit}
Result: Success - Found {len(result)} actionable elements. Current URL: '{state.get('url')}', Title: '{state.get('title')}'. Elements listed below.\n"""
                        + "\n".join(lines)
                    ),
                    state=state,
                )
            except Exception:
                logger.error("Failed to format extract_ui result", exc_info=True)
                error = "Failed to format extracted elements. Try browser_extract instead."
                state = await self._state_from_backend(user_id, backend, last_action_status="fail", error_message=error)
                return self._tool_command(
                    runtime,
                    content=f"""Tool Identity: browser_extract_ui
Action: Extracted UI structure from page
Result: Error - {error}""",
                    state=state,
                )

        return StructuredTool.from_function(
            name="browser_extract_ui",
            description=BROWSER_EXTRACT_UI_TOOL_DESCRIPTION,
            coroutine=async_extract_ui,
        )

    def _create_find_tool(self) -> BaseTool:
        """Create the find tool and refresh minimal browser state without storing matches."""
        async def async_find(
            selector: Annotated[str, "CSS selector to query on the current page."],
            runtime: ToolRuntime[None, BrowserMiddlewareState],
        ) -> Command:
            user_id, _ = self._resolve_runtime_key(runtime)
            backend, _ = self._backend_for_runtime(runtime)

            try:
                matches = await backend.find_elements(selector)
            except Exception:
                logger.exception("find failed with exception", exc_info=True)
                error = f"Unexpected error occurred while searching for selector '{selector}'."
                state = await self._state_from_backend(user_id, backend, last_action_status="fail", error_message=error)
                return self._tool_command(
                    runtime,
                    content=f"""Tool Identity: browser_find
Action: Searched for elements with CSS selector '{selector}'
Result: Error - {error}""",
                    state=state,
                )

            state = await self._state_from_backend(user_id, backend, last_action_status="success")
            if not matches:
                return self._tool_command(
                    runtime,
                    content=f"""Tool Identity: browser_find
Action: Searched for elements with CSS selector '{selector}'
Result: No elements found. Verify the selector is correct or use browser_extract_ui to discover available selectors.""",
                    state=state,
                )

            lines = []
            for i, match in enumerate(matches):
                lines.append(
                    f"- Match {i + 1}: selector='{match.selector}', type='{match.tag_name}', "
                    f"text='{match.text[:100]}', attributes={match.attributes}, "
                    f"rect={match.rect}, visible={match.is_visible}, enabled={match.is_enabled}"
                )
            return self._tool_command(
                runtime,
                content=(
                    f"""Tool Identity: browser_find
Action: Found elements matching CSS selector '{selector}'
Result: Success - Found {len(matches)} elements. Element details:\n"""
                    + "\n".join(lines)
                ),
                state=state,
            )

        return StructuredTool.from_function(
            name="browser_find",
            description=BROWSER_FIND_TOOL_DESCRIPTION,
            coroutine=async_find,
        )

    def _create_status_tool(self) -> BaseTool:
        """Create the status tool without creating a browser just to inspect state."""
        async def async_status(runtime: ToolRuntime[None, BrowserMiddlewareState]) -> Command:
            user_id, runtime_key = self._resolve_runtime_key(runtime)
            backend = self._session_backends.get(runtime_key)

            if backend is None:
                state = self._empty_browser_state(user_id)
                return self._tool_command(
                    runtime,
                    content="""Tool Identity: browser_status
Action: Checked browser session status
Result: No browser instance exists for this user. Use browser_navigate to create one and load a page.""",
                    state=state,
                )

            try:
                page_info = await backend.get_state_snapshot()
                state = self._state_from_page_info(user_id, backend, page_info)
                if not page_info.url:
                    return self._tool_command(
                        runtime,
                        content="""Tool Identity: browser_status
Action: Checked browser session status
Result: Browser instance exists, but no active page is loaded. Please navigate first using browser_navigate.""",
                        state=state,
                    )

                title = page_info.title or "N/A"
                return self._tool_command(
                    runtime,
                    content=f"""Tool Identity: browser_status
Action: Retrieved current browser session status
Result: Success - Browser is active. Current URL: '{page_info.url}', Title: '{title}'. Ready for interaction.""",
                    state=state,
                )
            except Exception as e:
                logger.debug("Failed to get browser state snapshot: %s", e)
                state = self._empty_browser_state(
                    user_id,
                    browser_exists=True,
                    browser_open=False,
                    last_action_status="fail",
                    error_message=str(e),
                )
                return self._tool_command(
                    runtime,
                    content="""Tool Identity: browser_status
Action: Checked browser session status
Result: Error - Browser instance exists, but current page state could not be read.""",
                    state=state,
                )

        return StructuredTool.from_function(
            name="browser_status",
            description=BROWSER_STATUS_TOOL_DESCRIPTION,
            coroutine=async_status,
        )

    def _create_probe_tool(self) -> BaseTool:
        """Create the probe tool and refresh minimal browser state without storing probe results."""
        async def async_probe(
            selector: Annotated[str, "CSS selector."],
            property_name: Annotated[str, "DOM property/attribute name to inspect."],
            runtime: ToolRuntime[None, BrowserMiddlewareState],
        ) -> Command:
            user_id, _ = self._resolve_runtime_key(runtime)
            backend, _ = self._backend_for_runtime(runtime)

            try:
                result = await backend.inspect_element_property(selector, property_name)
            except Exception:
                logger.exception("probe failed with exception", exc_info=True)
                error = "Unexpected error occurred while inspecting element."
                state = await self._state_from_backend(user_id, backend, last_action_status="fail", error_message=error)
                return self._tool_command(
                    runtime,
                    content=f"""Tool Identity: browser_probe
Action: Inspected element property '{property_name}' for selector '{selector}'
Result: Error - {error}""",
                    state=state,
                )

            if result.get("error"):
                error = f"{result['error']}. The element may not exist or the property is not accessible."
                state = await self._state_from_backend(user_id, backend, last_action_status="fail", error_message=error)
                return self._tool_command(
                    runtime,
                    content=f"""Tool Identity: browser_probe
Action: Inspected element property '{property_name}' for selector '{selector}'
Result: Error - {error}""",
                    state=state,
                )

            value = result.get("value")
            state = await self._state_from_backend(user_id, backend, last_action_status="success")
            if value is None:
                return self._tool_command(
                    runtime,
                    content=f"""Tool Identity: browser_probe
Action: Inspected element property '{property_name}' for selector '{selector}'
Result: Property not found. Verify the selector and property name.""",
                    state=state,
                )

            return self._tool_command(
                runtime,
                content=f"""Tool Identity: browser_probe
Action: Inspected element property '{property_name}' for selector '{selector}'
Result: Success - Property value is '{str(value)}'. Element is accessible. Use this information to determine if element is enabled, disabled, or has specific attributes.""",
                state=state,
            )

        return StructuredTool.from_function(
            name="browser_probe",
            description=BROWSER_PROBE_TOOL_DESCRIPTION,
            coroutine=async_probe,
        )
