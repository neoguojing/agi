import base64
import logging
from collections.abc import Awaitable, Callable
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
from typing_extensions import TypedDict

from agi.config import BROWSER_STORAGE_PATH
from agi.utils.common import append_to_system_message
from agi.web.browser_backend import StatefulBrowserBackend
from agi.web.browser_types import (
    PageInfo,
    QueryMatch,
    Rect,
    DEFAULT_VIEWPORT,
)
from agi.agent.prompt import get_middleware_prompt

logger = logging.getLogger(__name__)


class SessionRuntime(TypedDict):
    """In-memory runtime record for a browser session."""

    last_result: PageInfo | None
    previous_result: PageInfo | None


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
- Browser open/closed flags
- Current page summary (URL/title/screenshot path when available)
- Previous page summary (if available)

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


class MiddlewareBrowserState(AgentState):
    """Single middleware state schema.

    Middleware only relies on one structured browser field:
    - browser_session_state: compact snapshot from PageInfo
    """

    browser_session_state: Annotated[
        NotRequired[Optional[dict[str, Any]]],
    ]


class BrowserMiddleware(AgentMiddleware):
    """Stateful browser middleware with filesystem-style tool wrappers."""

    state_schema = MiddlewareBrowserState

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
        if system_prompt:
            request = request.override(system_message=append_to_system_message(request.system_message, system_prompt))
        return handler(request)

    async def awrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], Awaitable[ModelResponse[ResponseT]]],
    ) -> ModelResponse[ResponseT]:
        system_prompt = self._custom_system_prompt or get_middleware_prompt("browser")
        if system_prompt:
            request = request.override(system_message=append_to_system_message(request.system_message, system_prompt))
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

    def _resolve_user_id(self, runtime: ToolRuntime[None, MiddlewareBrowserState] | None = None) -> str:
        if runtime is not None:
            context = getattr(runtime, "context", None)
            if getattr(context, "user_id", None):
                return str(context.user_id)
            config = getattr(runtime, "config", {}) or {}
            configurable = config.get("configurable", {})
            if configurable.get("user_id"):
                return str(configurable["user_id"])
        return "default"

    def _resolve_runtime_key(self, runtime: ToolRuntime[None, MiddlewareBrowserState] | None = None) -> tuple[str, str]:
        user_id = self._resolve_user_id(runtime)
        return user_id, user_id

    def _backend_for_runtime(self, runtime: ToolRuntime[None, MiddlewareBrowserState] | None = None) -> tuple[StatefulBrowserBackend, str]:
        user_id, runtime_key = self._resolve_runtime_key(runtime)
        backend = self._session_backends.get(runtime_key)
        if backend is not None:
            return backend, runtime_key

        session_root = self._storage_root / user_id
        session_root.mkdir(parents=True, exist_ok=True)
        backend = StatefulBrowserBackend(storage_dir=str(session_root))
        self._session_backends[runtime_key] = backend
        return backend, runtime_key

    def _create_navigate_tool(self) -> BaseTool:
        """Create the navigate tool."""
        async def async_navigate(
            url: Annotated[str, "URL to open in the current browser session."],
            runtime: ToolRuntime[None, MiddlewareBrowserState],
        ) -> str:
            backend, runtime_key = self._backend_for_runtime(runtime)
            result = await backend.navigate(url)

            if result.last_action_status == "fail":
                return f"Error navigating to {url}: {result.error_message or 'Unknown error'}"

            if result.last_action_status == "timeout":
                return f"Navigation timed out after waiting for network idle. URL: {result.url}"

            return f"Successfully navigated to {result.url} - Title: {result.title or 'N/A'}"

        return StructuredTool.from_function(
            name="browser_navigate",
            description=BROWSER_NAVIGATE_TOOL_DESCRIPTION,
            coroutine=async_navigate,
        )

    def _create_click_tool(self) -> BaseTool:
        """Create the click tool."""
        async def async_click(
            selector: Annotated[str, "CSS selector to click on the current page."],
            runtime: ToolRuntime[None, MiddlewareBrowserState],
        ) -> str:
            backend, runtime_key = self._backend_for_runtime(runtime)
            result = await backend.click(selector)

            if result.last_action_status == "fail":
                return f"Error clicking element with selector '{selector}': {result.error_message or 'Unknown error'}"

            if result.last_action_status == "timeout":
                return f"Click timed out. Element may not be visible or clickable."

            return f"Successfully clicked element with selector: {selector}"

        return StructuredTool.from_function(
            name="browser_click",
            description=BROWSER_CLICK_TOOL_DESCRIPTION,
            coroutine=async_click,
        )

    def _create_fill_tool(self) -> BaseTool:
        """Create the fill tool."""
        async def async_fill(
            selector: Annotated[str, "CSS selector for the field to fill."],
            text: Annotated[str, "Text to enter into the selected field."],
            runtime: ToolRuntime[None, MiddlewareBrowserState],
        ) -> str:
            backend, runtime_key = self._backend_for_runtime(runtime)
            result = await backend.fill(selector, text)

            if result.last_action_status == "fail":
                return f"Error filling field with selector '{selector}': {result.error_message or 'Unknown error'}"

            if result.last_action_status == "timeout":
                return f"Fill timed out. Element may not be visible or editable."

            return f"Successfully filled field with selector: {selector} - Text: {text}"

        return StructuredTool.from_function(
            name="browser_fill",
            description=BROWSER_FILL_TOOL_DESCRIPTION,
            coroutine=async_fill,
        )

    def _create_scroll_tool(self) -> BaseTool:
        """Create the scroll tool."""
        async def async_scroll(
            direction: Annotated[str, "Scroll direction: up/down."],
            distance: Annotated[int, "Scroll distance in px."],
            runtime: ToolRuntime[None, MiddlewareBrowserState],
        ) -> str:
            backend, runtime_key = self._backend_for_runtime(runtime)
            result = await backend.scroll(direction, distance)

            if result.last_action_status == "fail":
                return f"Error scrolling: {result.error_message or 'Unknown error'}"

            if result.last_action_status == "timeout":
                return f"Scroll timed out."

            return f"Successfully scrolled {direction} by {distance}px - URL: {result.url}"

        return StructuredTool.from_function(
            name="browser_scroll",
            description=BROWSER_SCROLL_TOOL_DESCRIPTION,
            coroutine=async_scroll,
        )

    def _create_extract_tool(self) -> BaseTool:
        """Create the extract tool."""
        async def async_extract(runtime: ToolRuntime[None, MiddlewareBrowserState]) -> str:
            backend, runtime_key = self._backend_for_runtime(runtime)
            page = await backend.ensure_page()

            # Try OCR first if enabled
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
                                return f"Extracted {len(str(ocr_text).strip())} characters via OCR from screenshot at {screenshot_path}"
                except Exception as e:
                    logger.debug("OCR extraction failed: %s", e)

            # Fallback to DOM text extraction
            try:
                content = await page.content()
                return f"Extracted {len(content)} characters from page DOM. URL: {page.url}"
            except Exception as e:
                logger.debug("DOM extraction failed: %s", e)
                return "Error extracting page content: Unable to read page content"

        return StructuredTool.from_function(
            name="browser_extract",
            description=BROWSER_EXTRACT_TOOL_DESCRIPTION,
            coroutine=async_extract,
        )

    def _create_extract_ui_tool(self) -> BaseTool:
        """Create the extract UI tool."""
        async def async_extract_ui(
            runtime: ToolRuntime[None, MiddlewareBrowserState],
            limit: Annotated[int, "Maximum number of actionable elements to return."] = 12,
        ) -> str:
            backend, runtime_key = self._backend_for_runtime(runtime)
            result = await backend.extract_ui(max(1, min(int(limit or 12), 50)))

            if not result:
                return "No actionable elements found on the current page"

            return f"Extracted {len(result)} actionable elements from {result[0].url if result else 'current page'}"

        return StructuredTool.from_function(
            name="browser_extract_ui",
            description=BROWSER_EXTRACT_UI_TOOL_DESCRIPTION,
            coroutine=async_extract_ui,
        )

    def _create_find_tool(self) -> BaseTool:
        """Create the find tool - merged from _tool_find."""
        async def async_find(
            selector: Annotated[str, "CSS selector to query on the current page."],
            runtime: ToolRuntime[None, MiddlewareBrowserState],
        ) -> str:
            backend, runtime_key = self._backend_for_runtime(runtime)
            matches = await backend.find_elements(selector)

            if not matches:
                return f"No elements found matching selector: {selector}"

            return f"Found {len(matches)} elements matching selector: {selector}"

        return StructuredTool.from_function(
            name="browser_find",
            description=BROWSER_FIND_TOOL_DESCRIPTION,
            coroutine=async_find,
        )

    def _create_status_tool(self) -> BaseTool:
        """Create the status tool - calls get_state_snapshot from backend."""
        async def async_status(runtime: ToolRuntime[None, MiddlewareBrowserState]) -> str:
            backend, runtime_key = self._backend_for_runtime(runtime)

            try:
                page_info = await backend.get_state_snapshot()

                if not page_info.url:
                    return "No active browser session. Please navigate first."

                return f"Browser Status - URL: {page_info.url}, Title: {page_info.title or 'N/A'}"
            except Exception as e:
                logger.debug("Failed to get browser state snapshot: %s", e)
                return "No active browser session. Please navigate first."

        return StructuredTool.from_function(
            name="browser_status",
            description=BROWSER_STATUS_TOOL_DESCRIPTION,
            coroutine=async_status,
        )

    def _create_probe_tool(self) -> BaseTool:
        """Create the probe tool."""
        async def async_probe(
            selector: Annotated[str, "CSS selector."],
            property_name: Annotated[str, "DOM property/attribute name to inspect."],
            runtime: ToolRuntime[None, MiddlewareBrowserState],
        ) -> str:
            backend, runtime_key = self._backend_for_runtime(runtime)
            result = await backend.inspect_element_property(selector, property_name)

            if result.get("error"):
                return f"Error inspecting property '{property_name}': {result['error']}"

            value = result.get("value")
            if value is None:
                return f"Property '{property_name}' not found on element with selector: {selector}"

            return f"Property '{property_name}' value: {str(value)}"

        return StructuredTool.from_function(
            name="browser_probe",
            description=BROWSER_PROBE_TOOL_DESCRIPTION,
            coroutine=async_probe,
        )

    def _resolve_user_id(self, runtime: ToolRuntime[None, MiddlewareBrowserState] | None = None) -> str:
        if runtime is not None:
            context = getattr(runtime, "context", None)
            if getattr(context, "user_id", None):
                return str(context.user_id)
            config = getattr(runtime, "config", {}) or {}
            configurable = config.get("configurable", {})
            if configurable.get("user_id"):
                return str(configurable["user_id"])
        return "default"
