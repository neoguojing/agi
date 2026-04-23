import base64
import logging
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Annotated, Any, NotRequired

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
    BrowserSessionSnapshot,
    PageInfo,
    build_browser_runtime_key,
    normalize_browser_session_snapshot,
)
from agi.agent.prompt import get_middleware_prompt

logger = logging.getLogger(__name__)


class SessionRuntime(TypedDict):
    """In-memory runtime record for a browser session."""

    last_result: PageInfo | None
    previous_result: PageInfo | None


def _browser_session_state_reducer(
    left: BrowserSessionSnapshot | None,
    right: BrowserSessionSnapshot | None,
) -> BrowserSessionSnapshot | None:
    """Keep the latest browser session snapshot from middleware updates."""
    _ = left
    return right


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
    - browser_session_state: compact snapshot from agi.web.browser_types.BrowserSessionSnapshot
    """

    browser_session_state: Annotated[
        NotRequired[BrowserSessionSnapshot | None],
        _browser_session_state_reducer,
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
            self._create_screenshot_tool(),
            self._create_find_tool(),
            self._create_probe_tool(),
            self._create_status_tool(),
        ]

    def wrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], ModelResponse[ResponseT]],
    ) -> ModelResponse[ResponseT]:
        # 同步模型调用入口
        system_prompt = self._custom_system_prompt or get_middleware_prompt("browser")
        if system_prompt:
            request = request.override(system_message=append_to_system_message(request.system_message, system_prompt))
        return handler(request)

    # 异步模型调用入口：逻辑和同步版本一致，只是适配异步 handler。
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
        tool_result = await handler(request)
        return tool_result

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
        return user_id, session_id, build_browser_runtime_key(user_id, session_id)

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
        async def async_navigate(
            url: Annotated[str, "URL to open in the current browser session."],
            runtime: ToolRuntime[None, MiddlewareBrowserState],
        ) -> Command:
            backend, runtime_key = self._backend_for_runtime(runtime)
            result = await backend.navigate(url)

            # Update session state using backend's canonical snapshot
            session_state = backend.get_state_snapshot(last_result=result)

            return self._command_for_result(
                "browser_navigate",
                runtime.tool_call_id,
                self._format_page_result(result),
                session_state=session_state,
            )

        return StructuredTool.from_function(
            name="browser_navigate",
            description=BROWSER_NAVIGATE_TOOL_DESCRIPTION,
            coroutine=async_navigate,
        )

    def _create_click_tool(self) -> BaseTool:
        async def async_click(
            selector: Annotated[str, "CSS selector to click on the current page."],
            runtime: ToolRuntime[None, MiddlewareBrowserState],
        ) -> Command:
            backend, runtime_key = self._backend_for_runtime(runtime)
            result = await backend.click(selector)

            # Update session state using backend's canonical snapshot
            session_state = backend.get_state_snapshot(last_result=result)

            return self._command_for_result(
                "browser_click",
                runtime.tool_call_id,
                self._format_page_result(result),
                session_state=session_state,
            )

        return StructuredTool.from_function(
            name="browser_click",
            description=BROWSER_CLICK_TOOL_DESCRIPTION,
            coroutine=async_click,
        )

    def _create_fill_tool(self) -> BaseTool:
        async def async_fill(
            selector: Annotated[str, "CSS selector for the field to fill."],
            text: Annotated[str, "Text to enter into the selected field."],
            runtime: ToolRuntime[None, MiddlewareBrowserState],
        ) -> Command:
            backend, runtime_key = self._backend_for_runtime(runtime)
            result = await backend.fill(selector, text)

            # Update session state using backend's canonical snapshot
            session_state = backend.get_state_snapshot(last_result=result)

            return self._command_for_result(
                "browser_fill",
                runtime.tool_call_id,
                self._format_page_result(result),
                session_state=session_state,
            )

        return StructuredTool.from_function(
            name="browser_fill",
            description=BROWSER_FILL_TOOL_DESCRIPTION,
            coroutine=async_fill,
        )

    def _create_extract_tool(self) -> BaseTool:
        async def async_extract(runtime: ToolRuntime[None, MiddlewareBrowserState]) -> Command:
            artifact = await self._tool_extract(runtime)
            return self._command_for_result(
                "browser_extract",
                runtime.tool_call_id,
                artifact,
                session_state=self._extract_state_from_artifact(artifact),
            )

        return StructuredTool.from_function(
            name="browser_extract",
            description=BROWSER_EXTRACT_TOOL_DESCRIPTION,
            coroutine=async_extract,
        )

    def _create_scroll_tool(self) -> BaseTool:
        async def async_scroll(
            direction: Annotated[str, "Scroll direction: up/down."],
            distance: Annotated[int, "Scroll distance in px."],
            runtime: ToolRuntime[None, MiddlewareBrowserState],
        ) -> Command:
            backend, runtime_key = self._backend_for_runtime(runtime)
            result = await backend.scroll(direction, distance)

            # Update session state using backend's canonical snapshot
            session_state = backend.get_state_snapshot(last_result=result)

            return self._command_for_result(
                "browser_scroll",
                runtime.tool_call_id,
                self._format_page_result(result),
                session_state=session_state,
            )

        return StructuredTool.from_function(
            name="browser_scroll",
            description=BROWSER_SCROLL_TOOL_DESCRIPTION,
            coroutine=async_scroll,
        )

    def _create_screenshot_tool(self) -> BaseTool:
        async def async_screenshot(runtime: ToolRuntime[None, MiddlewareBrowserState]) -> ToolMessage | Command:
            return await self._tool_screenshot(runtime, runtime.tool_call_id)

        return StructuredTool.from_function(
            name="browser_screenshot",
            description=BROWSER_SCREENSHOT_TOOL_DESCRIPTION,
            coroutine=async_screenshot,
        )

    def _create_extract_ui_tool(self) -> BaseTool:
        async def async_extract_ui(
            runtime: ToolRuntime[None, MiddlewareBrowserState],
            limit: Annotated[int, "Maximum number of actionable elements to return."] = 12,
        ) -> Command:
            backend, runtime_key = self._backend_for_runtime(runtime)
            result = await backend.extract_ui(max(1, min(int(limit or 12), 50)))
            artifact = await self._artifact_with_state(
                {
                    "status": result.status,
                    "metadata": {"ui": result.metadata.get("ui")},
                    "content_preview": result.content,
                },
                runtime_key,
            )
            return self._command_for_result(
                "browser_extract_ui",
                runtime.tool_call_id,
                artifact,
                session_state=self._extract_state_from_artifact(artifact),
            )

        return StructuredTool.from_function(
            name="browser_extract_ui",
            description=BROWSER_EXTRACT_UI_TOOL_DESCRIPTION,
            coroutine=async_extract_ui,
        )

    def _create_find_tool(self) -> BaseTool:
        async def async_find(
            selector: Annotated[str, "CSS selector to query on the current page."],
            runtime: ToolRuntime[None, MiddlewareBrowserState],
        ) -> Command:
            backend, runtime_key = self._backend_for_runtime(runtime)
            result = await backend.find_elements(selector)

            # Get current page info for the artifact
            page = await backend.ensure_page()
            last_result = await backend._capture_page_info(page, page.url, None, capture_content=False, action="find")

            artifact = await self._artifact_with_state(
                {
                    "status": result.status,
                    "metadata": {
                        "selector": selector,
                        "count": len(result.metadata.get("matches", [])),
                        "matches": [{"text": m.text, "attrs": m.attributes} for m in result.metadata.get("matches", [])],
                        "empty_result": len(result.metadata.get("matches", [])) == 0,
                    },
                    "content_preview": result.content,
                    "url": last_result.url,
                    "title": last_result.title,
                },
                runtime_key,
            )
            return self._command_for_result(
                "browser_find",
                runtime.tool_call_id,
                artifact,
                session_state=self._extract_state_from_artifact(artifact),
            )

        return StructuredTool.from_function(
            name="browser_find",
            description=BROWSER_FIND_TOOL_DESCRIPTION,
            coroutine=async_find,
        )

    def _create_status_tool(self) -> BaseTool:
        async def async_status(runtime: ToolRuntime[None, MiddlewareBrowserState]) -> Command:
            _, runtime_key = self._backend_for_runtime(runtime)
            session_state = await self._get_canonical_session_state(runtime_key)
            if session_state is None:
                artifact = self._error_artifact("No active browser session. Please navigate first.")
                return self._command_for_result(
                    "browser_status",
                    runtime.tool_call_id,
                    artifact,
                    session_state=None,
                )

            browser = session_state.get("browser", {}) if isinstance(session_state, dict) else {}
            current_page = session_state.get("current_page", {}) if isinstance(session_state, dict) else {}
            previous_page = session_state.get("previous_page") if isinstance(session_state.get("previous_page"), dict) else None
            artifact: dict[str, Any] = {
                "status": "success",
                "content_preview": "Fetched current browser session status.",
                "metadata": {"browser_session_state": session_state},
                "current_page": dict(current_page) if isinstance(current_page, dict) else {},
            }
            if isinstance(previous_page, dict):
                artifact["previous_page"] = dict(previous_page)
            artifact["browser"] = dict(browser) if isinstance(browser, dict) else {}
            return self._command_for_result(
                "browser_status",
                runtime.tool_call_id,
                artifact,
                session_state=session_state,
            )

        return StructuredTool.from_function(
            name="browser_status",
            description=BROWSER_STATUS_TOOL_DESCRIPTION,
            coroutine=async_status,
        )

    def _create_probe_tool(self) -> BaseTool:
        # 属性探测原子：在 click 前确认按钮是否 disabled/隐藏/忙碌。
        async def async_probe(
            selector: Annotated[str, "CSS selector."],
            property_name: Annotated[str, "DOM property/attribute name to inspect."],
            runtime: ToolRuntime[None, MiddlewareBrowserState],
        ) -> Command:
            backend, runtime_key = self._backend_for_runtime(runtime)
            result = await backend.inspect_element_property(selector, property_name)
            artifact = await self._artifact_with_state(
                {
                    "status": result.status,
                    "metadata": result.metadata,
                    "content_preview": result.content,
                    "error": result.error,
                },
                runtime_key,
            )
            return self._command_for_result(
                "browser_probe",
                runtime.tool_call_id,
                artifact,
                session_state=self._extract_state_from_artifact(artifact),
            )

        return StructuredTool.from_function(
            name="browser_probe",
            description=BROWSER_PROBE_TOOL_DESCRIPTION,
            coroutine=async_probe,
        )

    async def _tool_extract(self, runtime: ToolRuntime[None, MiddlewareBrowserState]) -> dict[str, Any]:
        """Extract page content from the last successfully loaded page, prioritizing OCR."""

        backend, runtime_key = self._backend_for_runtime(runtime)
        page = await backend.ensure_page()

        # Re-capture PageInfo to ensure we have the full content
        last_result_full = await backend._capture_page_info(
            page, page.url, None, capture_content=True, action="extract"
        )

        dom_snapshot = last_result_full.dom_snapshot or ""
        page_text = last_result_full.page_text or ""
        ocr_text, screenshot_path = await self._extract_content_with_ocr(runtime, runtime_key, last_result_full)

        if not ocr_text and not dom_snapshot and not page_text:
            return await self._artifact_with_state(
                self._error_artifact(
                    "Page content is empty and OCR extraction was unavailable.",
                    url=last_result_full.url,
                    metadata=last_result_full.metadata,
                    screenshot_path=screenshot_path,
                ),
                runtime_key,
            )

        primary_content = ocr_text or page_text
        artifact: dict[str, Any] = {
            "status": "success",
            "metadata": {
                **dict(last_result_full.metadata),
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
            # Implement eviction similar to FilesystemMiddleware
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

        return await self._artifact_with_state(artifact, runtime_key)

    async def _tool_find(self, runtime: ToolRuntime[None, MiddlewareBrowserState], selector: str) -> dict[str, Any]:
        """Find candidate elements on the current page."""
        backend, runtime_key = self._backend_for_runtime(runtime)
        matches = await backend.find_elements(selector)

        # Get current page info for the artifact
        page = await backend.ensure_page()
        last_result = await backend._capture_page_info(page, page.url, None, capture_content=False, action="find")

        metadata = {
            "selector": selector,
            "count": len(matches),
            "matches": [{"text": match.text, "attrs": match.attributes} for match in matches[:10]],
            "empty_result": len(matches) == 0,
        }
        return await self._artifact_with_state(
            {
                "status": "success",
                "url": last_result.url,
                "title": last_result.title,
                "metadata": metadata,
                "content_preview": f"Found {len(matches)} element(s) for selector: {selector}",
            },
            runtime_key,
        )

    async def _extract_content_with_ocr(
        self,
        runtime: ToolRuntime[None, MiddlewareBrowserState],
        runtime_key: str,
        last_result: PageInfo,
    ) -> tuple[str, str | None]:
        """Capture a full-page screenshot and use OCR as the primary extraction path."""
        if not self.enable_ocr or self.ocr is None:
            return "", last_result.screenshot_path

        backend, _ = self._backend_for_runtime(runtime)
        screenshot_path = await backend.get_screenshot()
        
        if not screenshot_path:
            return "", last_result.screenshot_path

        # 单独读取文件内容获取 image_bytes
        from pathlib import Path
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
            logger.exception("OCR extraction failed for %s", last_result.url or "current page")
            return "", screenshot_path

        normalized_text = str(ocr_text).strip()
        return normalized_text, screenshot_path

    async def _tool_screenshot(self, runtime: ToolRuntime[None, MiddlewareBrowserState], tool_call_id: str) -> ToolMessage | Command:
        """Capture a screenshot and return a multimodal tool response."""
        backend, runtime_key = self._backend_for_runtime(runtime)
        result = await backend.get_screenshot()

        if result.status == "error":
            artifact = await self._artifact_with_state(self._error_artifact(result.error or "Failed to take screenshot"), runtime_key)
            return self._command_for_result(
                "browser_screenshot",
                tool_call_id,
                artifact,
                session_state=self._extract_state_from_artifact(artifact),
            )

        screenshot_path = result.metadata.get("screenshot_path")
        if not screenshot_path:
            artifact = await self._artifact_with_state(self._error_artifact("Screenshot path missing in result"), runtime_key)
            return self._command_for_result(
                "browser_screenshot",
                tool_call_id,
                artifact,
                session_state=self._extract_state_from_artifact(artifact),
            )

        page = await backend.ensure_page()
        last_result = await backend._capture_page_info(page, page.url, None, capture_content=False, action="screenshot")
        current_url = last_result.url
        text = f"Screenshot captured for {current_url}" if current_url else "Screenshot captured"
        artifact: dict[str, Any] = {
            "status": "success",
            "url": current_url,
            "title": last_result.title,
            "metadata": {"screenshot_path": screenshot_path},
            "content_preview": text,
            "screenshot_path": screenshot_path,
        }
        artifact = await self._artifact_with_state(artifact, runtime_key)
        session_state = self._extract_state_from_artifact(artifact)
        return Command(
            update={
                "browser_session_state": session_state,
                "messages": [
                    ToolMessage(
                        content=text,
                        content_blocks=[create_image_block(file_id=screenshot_path, mime_type="image/png")],
                        name="browser_screenshot",
                        tool_call_id=tool_call_id,
                        additional_kwargs={"artifact": artifact},
                    )
                ],
            }
        )

    def _format_page_result(self, result: PageInfo) -> dict[str, Any]:
        if result.metadata.get("error"):
            return self._error_artifact(
                str(result.metadata["error"]),
                url=result.url,
                title=result.title,
                metadata=result.metadata,
                screenshot_path=result.screenshot_path,
            )

        result.metadata.pop("elements", None)
        current_page = {
            "url": result.url,
            "title": result.title,
            "dom_snapshot": result.dom_snapshot,
            "page_text": result.page_text,
            "screenshot_path": result.screenshot_path,
            "response_status": result.response_status,
            "last_action": result.last_action,
            "actionable_elements": list(result.actionable_elements),
            "network_idle": result.network_idle,
            "url_changed": result.url_changed,
            "metadata": dict(result.metadata),
        }
        artifact: dict[str, Any] = {
            "status": "success",
            "metadata": dict(result.metadata),
            "content_preview": self._build_preview(result.page_text or ""),
            "current_page": current_page,
        }
        logger.info("Formatted tool artifact with current_page for url=%s", result.url)
        return artifact

    def _error_artifact(
        self,
        error: str,
        *,
        url: str = "",
        title: str | None = None,
        metadata: dict[str, Any] | None = None,
        screenshot_path: str | None = None,
    ) -> dict[str, Any]:
        current_page = {
            "url": url,
            "title": title,
            "dom_snapshot": None,
            "page_text": None,
            "screenshot_path": screenshot_path,
            "metadata": dict(metadata or {}),
        }
        artifact: dict[str, Any] = {
            "status": "error",
            "metadata": dict(metadata or {}),
            "content_preview": error,
            "error": error,
            "current_page": current_page,
        }
        return artifact

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

    # --- Removed functions for state-to-prompt splicing ---

    async def _artifact_with_state(self, artifact: dict[str, Any], runtime_key: str) -> dict[str, Any]:
        normalized_state = await self._get_canonical_session_state(runtime_key)
        if normalized_state is None:
            logger.debug("No live browser state for runtime_key=%s when building artifact", runtime_key)
            return artifact
        current_page = normalized_state.get("current_page")
        artifact["metadata"] = {
            **dict(artifact.get("metadata", {})),
            "browser_session_state": normalized_state,
        }
        if isinstance(current_page, dict):
            artifact["current_page"] = dict(current_page)
        previous_page = normalized_state.get("previous_page")
        if isinstance(previous_page, dict):
            artifact["previous_page"] = dict(previous_page)
        for legacy_key in ("url", "title", "screenshot_path", "element", "history_length", "page_info"):
            artifact.pop(legacy_key, None)
        logger.debug("Attached browser state/current_page to artifact for runtime_key=%s", runtime_key)
        return artifact

    def _extract_state_from_artifact(self, artifact: dict[str, Any]) -> BrowserSessionSnapshot | None:
        metadata = artifact.get("metadata", {})
        state = metadata.get("browser_session_state") if isinstance(metadata, dict) else None
        return self._normalize_llm_state(state) if isinstance(state, dict) else None

    def _command_for_result(
        # 把浏览器工具结果统一包装成 LangGraph Command，便于更新 agent state。
        self,
        tool_name: str,
        tool_call_id: str,
        artifact: dict[str, Any],
        session_state: BrowserSessionSnapshot | None = None,
    ) -> Command:
        text = self._artifact_to_text(tool_name, artifact)
        update: dict[str, Any] = {
            "messages": [
                ToolMessage(
                    content=text,
                    name=tool_name,
                    tool_call_id=tool_call_id,
                    additional_kwargs={"artifact": artifact},
                )
            ],
        }
        if session_state is not None:
            update["browser_session_state"] = session_state
        return Command(update=update)

    def _artifact_to_text(self, tool_name: str, artifact: dict[str, Any]) -> str:
        """Convert a tool artifact into a human-readable string for the LLM."""
        compact_current_page = self._compact_page_for_text(artifact.get("current_page"))
        compact_previous_page = self._compact_page_for_text(artifact.get("previous_page"))

        lines = [f"status: {artifact.get('status', 'unknown')}"]

        if compact_current_page.get("url"):
            lines.append(f"url: {compact_current_page['url']}")
        if compact_current_page.get("title"):
            lines.append(f"title: {compact_current_page['title']}")

        # Handle core content preview
        content_preview = artifact.get("content_preview")
        if content_preview:
            #- Navigate doesn't need a preview usually, it's the page state itself
            if tool_name != "browser_navigate":
                preview_limit = 800 if tool_name == "browser_extract" else 240
                lines.append(f"preview: {self._build_preview(str(content_preview), limit=preview_limit)}")

        if artifact.get("error"):
            lines.append(f"error: {artifact['error']}")

        if compact_current_page.get("screenshot_path"):
            lines.append(f"screenshot_path: {compact_current_page['screenshot_path']}")

        if artifact.get("ocr_text_preview"):
            lines.append(f"ocr_text_preview: {self._build_preview(str(artifact['ocr_text_preview']), limit=400)}")

        if artifact.get("full_content_path"):
            lines.append(f"full_content_path: {artifact['full_content_path']}")

        # Tool-specific metadata formatting
        metadata = artifact.get("metadata", {})
        if isinstance(metadata, dict):
            if tool_name == "browser_find":
                lines.append(f"matches_count: {metadata.get('count', 0)}")
            elif tool_name in {"browser_navigate", "browser_click", "browser_fill", "browser_scroll"}:
                lines.extend(self._format_actionable_elements(artifact.get("current_page")))
                current_page = artifact.get("current_page", {}) if isinstance(artifact.get("current_page"), dict) else {}
                lines.append(f"network_idle: {current_page.get('network_idle')}")
                lines.append(f"url_changed: {current_page.get('url_changed')}")
            elif tool_name == "browser_extract":
                lines.append(f"is_truncated: {artifact.get('is_truncated', False)}")
                lines.append(f"evicted: {bool(metadata.get('evicted'))}")
            elif tool_name == "browser_screenshot" and metadata.get("screenshot_path"):
                lines.append(f"metadata_screenshot_path: {metadata['screenshot_path']}")
            elif tool_name == "browser_extract_ui":
                ui = metadata.get("ui")
                if isinstance(ui, dict):
                    elements = ui.get("elements")
                    lines.append(f"ui_elements_count: {len(elements) if isinstance(elements, list) else 0}")

        if compact_current_page:
            lines.append(f"current_page: {compact_current_page}")
        if compact_previous_page:
            lines.append(f"previous_page: {compact_previous_page}")

        return "\n".join(lines)

    def _compact_page_for_text(self, page: Any) -> dict[str, Any]:
        if not isinstance(page, dict):
            return {}
        return {
            "url": page.get("url"),
            "title": page.get("title"),
            "screenshot_path": page.get("screenshot_path"),
        }

# Remove _next_step_hint as it's no longer needed for an autonomous LLM.

    def _format_actionable_elements(self, metadata: dict[str, Any] | Any, *, limit: int = 5) -> list[str]:
        if not isinstance(metadata, dict):
            return ["actionable_elements: []"]
        actionable = metadata.get("actionable_elements")
        if not isinstance(actionable, list) or not actionable:
            return ["actionable_elements: []"]
        valid_items = []
        for item in actionable:
            if not isinstance(item, dict):
                continue
            selector = str(item.get("selector") or item.get("sel") or "").strip()
            text = str(item.get("text") or item.get("c") or "").strip()
            kind = str(item.get("type") or item.get("t") or "").strip()
            placeholder = str(item.get("placeholder", "")).strip()
            disabled = bool(item.get("disabled", False))
            if not selector and not text and not kind and not placeholder:
                continue
            valid_items.append(
                {
                    "type": kind,
                    "selector": selector,
                    "text": text,
                    "placeholder": placeholder,
                    "disabled": disabled,
                }
            )
        if not valid_items:
            return ["actionable_elements: []"]

        lines = [f"actionable_count: {len(valid_items)}"]
        for idx, item in enumerate(valid_items[:limit], start=1):
            if not isinstance(item, dict):
                continue
            lines.append(
                "actionable_{idx}: type={type} selector={selector} text={text} placeholder={placeholder} disabled={disabled}".format(
                    idx=idx,
                    type=item.get("type", "-"),
                    selector=item.get("selector", "-"),
                    text=item.get("text", "-"),
                    placeholder=item.get("placeholder", "-"),
                    disabled=item.get("disabled", False),
                )
            )
        if len(valid_items) > limit:
            lines.append(f"actionable_more: {len(valid_items) - limit}")
        return lines

    def _count_valid_actionable_elements(self, metadata: dict[str, Any] | Any) -> int:
        if not isinstance(metadata, dict):
            return 0
        actionable = metadata.get("actionable_elements")
        if not isinstance(actionable, list):
            return 0
        count = 0
        for item in actionable:
            if not isinstance(item, dict):
                continue
            if any(
                (
                    str(item.get("selector") or item.get("sel") or "").strip(),
                    str(item.get("text") or item.get("c") or "").strip(),
                    str(item.get("type") or item.get("t") or "").strip(),
                    str(item.get("placeholder") or "").strip(),
                )
            ):
                count += 1
        return count

    def _build_preview(self, content: str, *, limit: int = 500) -> str:
        if not content:
            return ""
        if len(content) <= limit:
            return content
        return content[:limit] + "..."

    def _create_browser_content_preview(self, content: str, *, head_lines: int = 10, tail_lines: int = 10) -> str:
        """Create a preview of content showing head and tail with truncation marker.

        Similar to FilesystemMiddleware's _create_content_preview.
        """
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
