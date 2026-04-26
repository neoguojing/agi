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

BROWSER_SCREENSHOT_TOOL_DESCRIPTION = """
Captures a screenshot of the current page.

Key Behavior:
- Produces an image for OCR or visual inspection.

Returns:
- Image content (for model inspection)
- Metadata including URL and file path

Guidelines:
- Use when:
  - OCR is needed explicitly
  - layout or visual elements matter
  - debugging is required
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
            self._create_screenshot_tool(),
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

    def _resolve_session_id(self, runtime: ToolRuntime[None, MiddlewareBrowserState] | None = None) -> str:
        if runtime is not None:
            context = getattr(runtime, "context", None)
            if getattr(context, "conversation_id", None):
                return str(context.conversation_id)
            config = getattr(runtime, "config", {}) or {}
            configurable = config.get("configurable", {}) if isinstance(config, dict) else {}
            if configurable.get("conversation_id"):
                return str(configurable["conversation_id"])
        return "default"

    def _resolve_runtime_key(self, runtime: ToolRuntime[None, MiddlewareBrowserState] | None = None) -> tuple[str, str, str]:
        user_id = self._resolve_user_id(runtime)
        session_id = self._resolve_session_id(runtime)
        return user_id, session_id, f"{user_id}_{session_id}"

    def _backend_for_runtime(self, runtime: ToolRuntime[None, MiddlewareBrowserState] | None = None) -> tuple[StatefulBrowserBackend, str]:
        user_id, session_id, runtime_key = self._resolve_runtime_key(runtime)
        backend = self._session_backends.get(runtime_key)
        if backend is not None:
            return backend, runtime_key

        session_root = self._storage_root / user_id / session_id
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

            return f"Navigate to {result.url} - Title: {result.title}"

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

            return f"Clicked element with selector: {selector} - URL: {result.url}"

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

            return f"Filled field with selector: {selector} - Text: {text}"

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

            return f"Scrolled {direction} by {distance}px - URL: {result.url}"

        return StructuredTool.from_function(
            name="browser_scroll",
            description=BROWSER_SCROLL_TOOL_DESCRIPTION,
            coroutine=async_scroll,
        )

    def _create_extract_tool(self) -> BaseTool:
        """Create the extract tool."""
        async def async_extract(runtime: ToolRuntime[None, MiddlewareBrowserState]) -> str:
            artifact = await self._tool_extract(runtime)
            return artifact.get("content_preview", "")

        return StructuredTool.from_function(
            name="browser_extract",
            description=BROWSER_EXTRACT_TOOL_DESCRIPTION,
            coroutine=async_extract,
        )

    def _create_screenshot_tool(self) -> BaseTool:
        """Create the screenshot tool - merged from _tool_screenshot."""
        async def async_screenshot(runtime: ToolRuntime[None, MiddlewareBrowserState]) -> ToolMessage | str:
            backend, runtime_key = self._backend_for_runtime(runtime)
            result = await backend.get_screenshot()

            if result.status == "error":
                return f"Error: {result.error or 'Failed to take screenshot'}"

            screenshot_path = result.metadata.get("screenshot_path")
            if not screenshot_path:
                return "Error: Screenshot path missing in result"

            page = await backend.ensure_page()
            current_url = page.url
            text = f"Screenshot captured for {current_url}" if current_url else "Screenshot captured"

            return ToolMessage(
                content=text,
                content_blocks=[create_image_block(file_id=screenshot_path, mime_type="image/png")],
                name="browser_screenshot",
                tool_call_id=runtime.tool_call_id,
            )

        return StructuredTool.from_function(
            name="browser_screenshot",
            description=BROWSER_SCREENSHOT_TOOL_DESCRIPTION,
            coroutine=async_screenshot,
        )

    def _create_extract_ui_tool(self) -> BaseTool:
        """Create the extract UI tool."""
        async def async_extract_ui(
            runtime: ToolRuntime[None, MiddlewareBrowserState],
            limit: Annotated[int, "Maximum number of actionable elements to return."] = 12,
        ) -> str:
            backend, runtime_key = self._backend_for_runtime(runtime)
            result = await backend.extract_ui(max(1, min(int(limit or 12), 50)))

            return f"Extracted {len(result)} actionable elements from {result.url}"

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

            return f"Found {len(matches)} elements matching selector: {selector}"

        return StructuredTool.from_function(
            name="browser_find",
            description=BROWSER_FIND_TOOL_DESCRIPTION,
            coroutine=async_find,
        )

    def _create_status_tool(self) -> BaseTool:
        """Create the status tool."""
        async def async_status(runtime: ToolRuntime[None, MiddlewareBrowserState]) -> str:
            _, runtime_key = self._backend_for_runtime(runtime)
            session_state = await self._get_canonical_session_state(runtime_key)

            if session_state is None:
                return "No active browser session. Please navigate first."

            current_page = session_state.get("current_page", {}) if isinstance(session_state, dict) else {}

            return f"Browser Status - URL: {current_page.get('url', 'N/A')}, Title: {current_page.get('title', 'N/A')}"

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
                return f"Error inspecting property {property_name}: {result['error']}"

            return f"Property '{property_name}' value: {result.get('value', 'N/A')}"

        return StructuredTool.from_function(
            name="browser_probe",
            description=BROWSER_PROBE_TOOL_DESCRIPTION,
            coroutine=async_probe,
        )

    async def _tool_extract(self, runtime: ToolRuntime[None, MiddlewareBrowserState]) -> dict[str, Any]:
        """Extract page content from the last successfully loaded page, prioritizing OCR."""
        backend, runtime_key = self._backend_for_runtime(runtime)
        page = await backend.ensure_page()

        dom_snapshot = ""
        page_text = ""

        ocr_text, screenshot_path = await self._extract_content_with_ocr(runtime, runtime_key)

        if not ocr_text and not dom_snapshot and not page_text:
            return {
                "status": "error",
                "content_preview": "Page content is empty and OCR extraction was unavailable.",
                "metadata": {},
                "screenshot_path": screenshot_path,
            }

        primary_content = ocr_text or page_text
        artifact: dict[str, Any] = {
            "status": "success",
            "metadata": {
                "ocr_priority": True,
                "ocr_applied": bool(ocr_text),
            },
            "content_preview": self._build_preview(primary_content),
        }
        if screenshot_path:
            artifact["screenshot_path"] = screenshot_path
        if ocr_text:
            artifact["ocr_text_preview"] = self._build_preview(ocr_text, limit=self.content_limit)

        if len(dom_snapshot) > self.content_limit:
            artifact["dom_snapshot_preview"] = self._create_browser_content_preview(dom_snapshot)
            artifact["page_text_preview"] = self._create_browser_content_preview(page_text)
            artifact["is_truncated"] = True
            if self.eviction_handler is not None:
                file_path = self.eviction_handler(dom_snapshot)
                artifact["full_content_path"] = file_path
                artifact["metadata"] = {
                    **artifact["metadata"],
                    "evicted": True,
                    "full_content_path": file_path,
                }
        else:
            artifact["dom_snapshot_preview"] = dom_snapshot
            artifact["page_text_preview"] = page_text
            artifact["is_truncated"] = False

        return artifact

    async def _extract_content_with_ocr(
        self,
        runtime: ToolRuntime[None, MiddlewareBrowserState],
        runtime_key: str,
    ) -> tuple[str, str | None]:
        """Capture a full-page screenshot and use OCR as the primary extraction path."""
        if not self.enable_ocr or self.ocr is None:
            return "", ""

        backend, _ = self._backend_for_runtime(runtime)
        screenshot_path = await backend.get_screenshot()

        if not screenshot_path:
            return "", ""

        image_path = Path(screenshot_path)
        if not image_path.exists():
            logger.warning("Screenshot file not found: %s", screenshot_path)
            return "", screenshot_path

        try:
            image_bytes = image_path.read_bytes()
        except Exception as e:
            logger.exception("Failed to read screenshot bytes from %s", screenshot_path, exc_info=True)
            return "", screenshot_path

        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
        try:
            ocr_text = await self.ocr.ainvoke([
                HumanMessage(
                    content="extract the image content",
                    content_blocks=[create_image_block(base64=image_b64, mime_type="image/png")],
                )
            ])
        except Exception:
            logger.exception("OCR extraction failed for %s", screenshot_path)
            return "", screenshot_path

        normalized_text = str(ocr_text).strip()
        return normalized_text, screenshot_path

    def _build_preview(self, content: str, *, limit: int = 500) -> str:
        if not content:
            return ""
        if len(content) <= limit:
            return content
        return content[:limit] + "..."

    def _create_browser_content_preview(self, content: str, *, head_lines: int = 10, tail_lines: int = 10) -> str:
        """Create a preview of content showing head and tail with truncation marker."""
        if not content:
            return ""
        lines = content.splitlines()
        if len(lines) <= head_lines + tail_lines:
            return "\n".join(lines)

        head = lines[:head_lines]
        tail = lines[-tail_lines:]

        head_text = "\n".join(head)
        tail_text = "\n".join(tail)
        truncation_notice = f"\n... [{len(lines) - head_lines - tail_lines} lines truncated] ...\n"

        return head_text + truncation_notice + tail_text

    async def _get_canonical_session_state(self, runtime_key: str) -> dict[str, Any] | None:
        """Get the canonical session state for a runtime key."""
        backend, _ = self._backend_for_runtime(None)  # Use default backend if no runtime provided
        page = await backend.ensure_page()

        return {
            "browser": {"open": True},
            "current_page": {
                "url": page.url,
                "title": await page.title(),
                "screenshot_path": None,
            },
            "previous_page": None,
        }

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
