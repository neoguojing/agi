import asyncio
import base64
import logging
import random
import time
from collections.abc import Awaitable, Callable
from typing import Annotated, Any

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ContextT,
    ModelRequest,
    ModelResponse,
    ResponseT,
)
from langchain.tools import ToolRuntime
from langchain_core.messages import ToolMessage,HumanMessage
from langchain_core.messages.content import create_image_block
from langchain_core.tools import BaseTool, StructuredTool
from langchain.tools.tool_node import ToolCallRequest
from langgraph.types import Command

from agi.config import BROWSER_STORAGE_PATH
from agi.utils.common import append_to_system_message
from agi.web.session_manager import BrowserSessionManager
from agi.web.browser_backend import StatefulBrowserBackend
from agi.web.browser_types import BrowserSessionSnapshot, PageInfo, normalize_browser_session_snapshot
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
Finds elements on the current page matching a CSS selector.

Key Behavior:
- Returns text and attributes of up to 10 matches.

Guidelines:
- Use for:
  - discovering selectors
  - verifying element presence
- Prefer this before click/fill when uncertain.
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
Returns: title/url + actionable elements list for planning click/fill.
"""


class MiddlewareBrowserState(AgentState):
    """Single middleware state schema.

    Middleware only relies on one structured browser field:
    - browser_session_state: compact snapshot from agi.web.browser_types.BrowserSessionSnapshot
    """

    browser_session_state: BrowserSessionSnapshot | None


class BrowserMiddleware(AgentMiddleware):
    """Stateful browser middleware with filesystem-style tool wrappers."""

    # 这里不直接管理 Playwright 细节，而是通过 BrowserBackendPool 获取"某个用户的当前浏览器会话"。

    def __init__(
        self,
        storage_dir: str = BROWSER_STORAGE_PATH,
        ocr_engine: Any | None = None,
        max_retries: int = 3,
        enable_ocr_fallback: bool = True,
        content_token_limit: int = 15_000,
        eviction_handler: Callable[[str], str] | None = None,
        system_prompt: str | None = None,
        session_ttl: int = 1800,  # 30分钟，与 BrowserSessionManager 保持一致
        cleanup_interval: int = 60,  # 清理间隔，与 BrowserSessionManager 保持一致
    ):
        super().__init__()
        self._session_manager = BrowserSessionManager(
            backend_factory=lambda: StatefulBrowserBackend(storage_dir=storage_dir),
            session_ttl=session_ttl,
            cleanup_interval=cleanup_interval
        )
        self.ocr = ocr_engine
        self.max_retries = max_retries
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

    # 同步模型调用入口：在真正调用模型前，把当前浏览器状态摘要拼到 system prompt。
    def wrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], ModelResponse[ResponseT]],
    ) -> ModelResponse[ResponseT]:
        # Sync hook cannot await live browser state safely; use static guidance.
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
        system_prompt = await self._build_model_system_prompt(request)
        if system_prompt:
            request = request.override(system_message=append_to_system_message(request.system_message, system_prompt))
        return await handler(request)

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        """Check the size of the tool call result and evict to filesystem if too large.

        Args:
            request: The tool call request being processed.
            handler: The handler function to call with the modified request.

        Returns:
            The raw ToolMessage, or a pseudo tool message with the ToolResult in state.
        """
        return handler(request)


    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        """(async)Check the size of the tool call result and evict to filesystem if too large.

        Args:
            request: The tool call request being processed.
            handler: The handler function to call with the modified request.

        Returns:
            The raw ToolMessage, or a pseudo tool message with the ToolResult in state.
        """
        tool_result = await handler(request)
        return tool_result

    
    def _create_navigate_tool(self) -> BaseTool:
        async def async_navigate(
            url: Annotated[str, "URL to open in the current browser session."],
            runtime: ToolRuntime[None, MiddlewareBrowserState],
        ) -> Command:
            result = await self._execute_with_retry(runtime, "navigate", url=url)
            return self._command_for_result(
                "browser_navigate",
                runtime.tool_call_id,
                self._format_page_result(result),
                session_state=self._extract_state_from_result(result),
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
            result = await self._execute_with_retry(runtime, "click", selector=selector)
            return self._command_for_result(
                "browser_click",
                runtime.tool_call_id,
                self._format_page_result(result),
                session_state=self._extract_state_from_result(result),
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
            result = await self._execute_with_retry(runtime, "fill", selector=selector, text=text)
            return self._command_for_result(
                "browser_fill",
                runtime.tool_call_id,
                self._format_page_result(result),
                session_state=self._extract_state_from_result(result),
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
        # 视口操纵原子：用于触发懒加载并暴露屏外元素。
        async def async_scroll(
            direction: Annotated[str, "Scroll direction: up/down."],
            distance: Annotated[int, "Scroll distance in px."],
            runtime: ToolRuntime[None, MiddlewareBrowserState],
        ) -> Command:
            result = await self._execute_with_retry(runtime, "scroll", direction=direction, distance=distance)
            return self._command_for_result(
                "browser_scroll",
                runtime.tool_call_id,
                self._format_page_result(result),
                session_state=self._extract_state_from_result(result),
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
            user_id = self._resolve_user_id(runtime)
            ui_payload = await self._session_manager.extract_ui(user_id, max(1, min(int(limit or 12), 50)))
            artifact = await self._artifact_with_state(
                {
                    "status": "success",
                    "metadata": {"ui": ui_payload},
                    "content_preview": f"Extracted {len(ui_payload.get('elements', [])) if isinstance(ui_payload, dict) else 0} actionable UI element(s).",
                },
                user_id,
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
            artifact = await self._tool_find(runtime, selector)
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
            user_id = self._resolve_user_id(runtime)
            session_state = await self._get_canonical_session_state(user_id)
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
            user_id = self._resolve_user_id(runtime)
            probe = await self._session_manager.inspect_element_property(user_id, selector, property_name)
            artifact = await self._artifact_with_state(
                {
                    "status": "success" if probe.get("ok") else "error",
                    "metadata": {"probe": probe},
                    "content_preview": str(probe),
                },
                user_id,
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
        user_id = self._resolve_user_id(runtime)
        runtime_context = await self._session_manager.get_runtime_context(user_id)
        last_result = runtime_context.get("last_result")
        if not last_result:
            return await self._artifact_with_state(
                self._error_artifact("No page loaded. Please navigate first."),
                user_id,
            )

        html = last_result.html or ""
        text = last_result.text or ""
        ocr_text, screenshot_path = await self._extract_content_with_ocr(user_id, last_result)
        if not ocr_text and not html and not text:
            return await self._artifact_with_state(
                self._error_artifact(
                    "Page content is empty and OCR extraction was unavailable.",
                    url=last_result.url,
                    metadata=last_result.metadata,
                    screenshot_path=screenshot_path,
                ),
                user_id,
            )

        primary_content = ocr_text or text
        artifact: dict[str, Any] = {
            "status": "success",
            "url": last_result.url,
            "title": last_result.title,
            "metadata": {
                **dict(last_result.metadata),
                "ocr_priority": True,
                "ocr_applied": bool(ocr_text),
            },
            "content_preview": self._build_preview(primary_content),
        }
        if screenshot_path:
            artifact["screenshot_path"] = screenshot_path
        if ocr_text:
            artifact["ocr_text_preview"] = self._build_preview(ocr_text, limit=self.content_limit)

        if len(html) > self.content_limit:
            artifact["html_preview"] = self._build_preview(html, limit=self.content_limit)
            artifact["text_preview"] = self._build_preview(text, limit=self.content_limit)
            artifact["is_truncated"] = True
            if self.eviction_handler is not None:
                file_path = self.eviction_handler(html)
                artifact["full_content_path"] = file_path
                artifact["metadata"] = {
                    **artifact["metadata"],
                    "evicted": True,
                    "full_content_path": file_path,
                }
        else:
            artifact["html_preview"] = html
            artifact["text_preview"] = text
            artifact["is_truncated"] = False

        return await self._artifact_with_state(artifact, user_id)

    async def _tool_find(self, runtime: ToolRuntime[None, MiddlewareBrowserState], selector: str) -> dict[str, Any]:
        """Find candidate elements on the current page."""
        user_id = self._resolve_user_id(runtime)
        
        # 使用 session manager 的统一 API
        matches = await self._session_manager.find_elements(user_id, selector)
        runtime_context = await self._session_manager.get_runtime_context(user_id)
        state = runtime_context.get("state")
        last_result = runtime_context.get("last_result")
        
        metadata = {
            "selector": selector,
            "count": len(matches),
            "matches": [{"text": match.text, "attrs": match.attributes} for match in matches[:10]],
            "empty_result": len(matches) == 0,
        }
        return await self._artifact_with_state(
            {
                "status": "success",
                "url": last_result.url if last_result else "",
                "title": last_result.title if last_result else None,
                "metadata": metadata,
                "content_preview": f"Found {len(matches)} element(s) for selector: {selector}",
                "history_length": int(state.get("history_length", 0)) if state else 0,
            },
            user_id,
        )

    async def _extract_content_with_ocr(self, user_id: str, last_result: PageInfo) -> tuple[str, str | None]:
        """Capture a full-page screenshot and use OCR as the primary extraction path."""
        if not self.enable_ocr or self.ocr is None:
            return "", last_result.screenshot_path

        # 使用 session manager 的统一 API 获取截图路径（不是元组）
        screenshot_path = await self._session_manager.screenshot(user_id)
        
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
        if normalized_text:
            await self._session_manager.apply_ocr_result(
                user_id,
                text=normalized_text,
                screenshot_path=screenshot_path,
                metadata_update={
                    "ocr_applied": True,
                    "ocr_text_length": len(normalized_text),
                    "ocr_screenshot_path": screenshot_path,
                },
            )
        return normalized_text, screenshot_path

    async def _tool_screenshot(self, runtime: ToolRuntime[None, MiddlewareBrowserState], tool_call_id: str) -> ToolMessage | Command:
        """Capture a screenshot and return a multimodal tool response."""
        user_id = self._resolve_user_id(runtime)
        
        # 使用 session manager 的统一 API 获取截图路径（不是元组）
        screenshot_path = await self._session_manager.screenshot(user_id)
        
        if not screenshot_path:
            artifact = await self._artifact_with_state(self._error_artifact("Failed to take screenshot"), user_id)
            return self._command_for_result(
                "browser_screenshot",
                tool_call_id,
                artifact,
                session_state=self._extract_state_from_artifact(artifact),
            )

        # 单独读取文件内容获取 image_bytes（如果需要）
        from pathlib import Path
        image_path = Path(screenshot_path)
        if not image_path.exists():
            logger.warning("Screenshot file not found: %s", screenshot_path)
            artifact = await self._artifact_with_state(self._error_artifact(f"Screenshot file not found: {screenshot_path}"), user_id)
            return self._command_for_result(
                "browser_screenshot",
                tool_call_id,
                artifact,
                session_state=self._extract_state_from_artifact(artifact),
            )

        # image_b64 = base64.b64encode(image_bytes).decode("utf-8")
        runtime_context = await self._session_manager.get_runtime_context(user_id)
        last_result = runtime_context.get("last_result")
        current_url = last_result.url if last_result else ""
        text = f"Screenshot captured for {current_url}" if current_url else "Screenshot captured"
        artifact: dict[str, Any] = {
            "status": "success",
            "url": current_url,
            "title": last_result.title if last_result else None,
            "metadata": {"screenshot_path": screenshot_path},
            "content_preview": text,
            "screenshot_path": screenshot_path,
        }
        artifact = await self._artifact_with_state(artifact, user_id)
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

    async def _execute_with_retry(
        self,
        runtime: ToolRuntime[None, MiddlewareBrowserState],
        action: str,
        **kwargs: Any
    ) -> PageInfo:
        last_error: Exception | None = None
        user_id = self._resolve_user_id(runtime)

        for attempt in range(self.max_retries):
            try:
                if attempt > 0:
                    delay = random.uniform(1.0, 3.0)
                    await asyncio.sleep(delay)

                started_at = time.perf_counter()

                # 使用 session manager 的统一 API
                if action == "navigate":
                    result = await self._session_manager.navigate(user_id, kwargs["url"])
                elif action == "click":
                    result = await self._session_manager.click(user_id, kwargs["selector"])
                elif action == "fill":
                    result = await self._session_manager.fill(user_id, kwargs["selector"], kwargs["text"])
                elif action == "scroll":
                    result = await self._session_manager.scroll(user_id, kwargs["direction"], kwargs["distance"])
                else:
                    msg = f"Unknown action: {action}"
                    raise ValueError(msg)

                logger.info(
                    "Browser action '%s' for user_id=%s completed in %.2fs",
                    action,
                    user_id,
                    time.perf_counter() - started_at,
                )

                if result.metadata.get("error"):
                    raise RuntimeError(str(result.metadata["error"]))

                canonical_state = await self._get_canonical_session_state(user_id)
                # 把实时会话快照塞回每次动作结果，确保“动作-反馈一体化”。
                result.metadata = {
                    **result.metadata,
                    "browser_session_state": canonical_state or {"browser": {"is_open": False, "is_closed": True}, "current_page": {}, "previous_page": None},
                }

                return result

            except Exception as exc:
                last_error = exc
                logger.warning(
                    "Browser action '%s' failed (%s/%s): %s",
                    action,
                    attempt + 1,
                    self.max_retries,
                    exc,
                )

                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2**attempt)

        return PageInfo(
            url=kwargs.get("url", "unknown"),
            title=None,
            html=None,
            text=None,
            screenshot_path=None,
            metadata={"error": f"Failed after {self.max_retries}: {last_error}"},
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
            "html": result.html,
            "text": result.text,
            "screenshot_path": result.screenshot_path,
            "status": result.status,
            "action": result.action,
            "actionable_elements": list(result.actionable_elements),
            "network_idle": result.network_idle,
            "url_changed": result.url_changed,
            "diagnostics": dict(result.diagnostics),
            "metadata": dict(result.metadata),
        }
        artifact: dict[str, Any] = {
            "status": "success",
            "metadata": dict(result.metadata),
            "content_preview": self._build_preview(result.text),
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
            "html": None,
            "text": None,
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

    async def _build_model_system_prompt(self, request: ModelRequest[ContextT]) -> str:
        # system prompt = 浏览器工具说明 + 当前 live browser state 摘要。
        system_prompt = self._custom_system_prompt or get_middleware_prompt("browser")
        session_state = await self._resolve_session_state_for_request(request)
        if not session_state:
            return system_prompt
        return f"{system_prompt}\n\n{self._format_browser_state_for_prompt(session_state)}"

    async def _resolve_session_state_for_request(self, request: ModelRequest[ContextT]) -> BrowserSessionSnapshot | None:
        """Resolve canonical browser session state for LLM context."""
        state = getattr(request, "state", None) or {}
        if isinstance(state, dict):
            session_state = state.get("browser_session_state")
            if isinstance(session_state, dict):
                user_id = await self._resolve_user_id_from_request(request)
                live_state = await self._get_canonical_session_state(user_id=user_id) if user_id else None
                if live_state:
                    return live_state
                return self._normalize_llm_state(session_state)

        user_id = await self._resolve_user_id_from_request(request)
        if not user_id:
            return None
        return await self._get_canonical_session_state(user_id=user_id)

    async def _resolve_user_id_from_request(self, request: ModelRequest[ContextT]) -> str | None:
        state = getattr(request, "state", None) or {}
        if isinstance(state, dict) and state.get("user_id"):
            return str(state["user_id"])

        runtime = getattr(request, "runtime", None)
        if runtime is not None:
            return self._resolve_user_id(runtime)

        context = getattr(request, "context", None)
        if getattr(context, "user_id", None):
            return str(context.user_id)

        config = getattr(request, "config", {}) or {}
        configurable = config.get("configurable", {}) if isinstance(config, dict) else {}
        if configurable.get("user_id"):
            return str(configurable["user_id"])
        return None

    async def _get_canonical_session_state(self, user_id: str | None) -> BrowserSessionSnapshot | None:
        """Single source of truth for middleware/browser state normalization."""
        if not user_id:
            return None
        runtime_context = await self._session_manager.get_runtime_context(user_id)
        state = runtime_context.get("state")
        if not state:
            return None
        return self._normalize_llm_state(state)

    def _format_browser_state_for_prompt(self, session_state: BrowserSessionSnapshot) -> str:
        """Generate compact, task-relevant LLM-facing browser state."""
        browser = session_state.get("browser", {}) or {}
        current_page = session_state.get("current_page", {}) or {}
        previous_page = session_state.get("previous_page") if isinstance(session_state.get("previous_page"), dict) else {}

        def _compact_page(page: dict[str, Any]) -> dict[str, Any]:
            return {
                "url": page.get("url"),
                "title": page.get("title"),
                "screenshot_path": page.get("screenshot_path"),
            }

        current_page_compact = _compact_page(current_page) if isinstance(current_page, dict) else {}
        previous_page_compact = _compact_page(previous_page) if isinstance(previous_page, dict) else None
        return "\n".join(
            [
                "## Current Browser Session State",
                f"browser: is_open={browser.get('is_open')} is_closed={browser.get('is_closed')}",
                f"current_page: {current_page_compact}",
                f"previous_page: {previous_page_compact or '<none>'}",
                f"last_action: {current_page.get('metadata', {}).get('action') if isinstance(current_page.get('metadata'), dict) else None}",
                "Decide next action only from this summary: status, navigate, find, click, fill, extract, screenshot.",
            ]
        )

    def _normalize_llm_state(self, state: dict[str, Any] | None) -> BrowserSessionSnapshot:
        """Normalize arbitrary/raw state using shared browser_types schema helper."""
        normalized = normalize_browser_session_snapshot(state)
        logger.debug(
            "Normalizing browser state via shared helper: current_page_keys=%s previous_page_type=%s",
            list(normalized.get("current_page", {}).keys()),
            type(normalized.get("previous_page")).__name__,
        )
        return normalized

    async def _artifact_with_state(self, artifact: dict[str, Any], user_id: str) -> dict[str, Any]:
        normalized_state = await self._get_canonical_session_state(user_id)
        if normalized_state is None:
            logger.debug("No live browser state for user_id=%s when building artifact", user_id)
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
        logger.debug("Attached browser state/current_page to artifact for user_id=%s", user_id)
        return artifact

    def _extract_state_from_result(self, result: PageInfo) -> BrowserSessionSnapshot | None:
        state = result.metadata.get("browser_session_state")
        return self._normalize_llm_state(state) if isinstance(state, dict) else None

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
        compact_current_page = self._compact_page_for_text(artifact.get("current_page"))
        compact_previous_page = self._compact_page_for_text(artifact.get("previous_page"))
        lines = [f"status: {artifact['status']}"]
        if compact_current_page.get("url"):
            lines.append(f"url: {compact_current_page['url']}")
        if compact_current_page.get("title"):
            lines.append(f"title: {compact_current_page['title']}")

        if artifact.get("content_preview") and tool_name != "browser_navigate":
            preview_limit = 800 if tool_name == "browser_extract" else 240
            lines.append(f"preview: {self._build_preview(str(artifact['content_preview']), limit=preview_limit)}")
        if artifact.get("error"):
            lines.append(f"error: {artifact['error']}")
        if compact_current_page.get("screenshot_path"):
            lines.append(f"screenshot_path: {compact_current_page['screenshot_path']}")
        if artifact.get("ocr_text_preview"):
            lines.append(f"ocr_text_preview: {self._build_preview(str(artifact['ocr_text_preview']), limit=400)}")
        if artifact.get("full_content_path"):
            lines.append(f"full_content_path: {artifact['full_content_path']}")

        metadata = artifact.get("metadata", {})
        if isinstance(metadata, dict):
            if tool_name == "browser_find":
                lines.append(f"matches_count: {metadata.get('count', 0)}")
            elif tool_name in {"browser_navigate", "browser_click", "browser_fill", "browser_scroll"}:
                # 非 navigate 动作也回传可交互元素，减少模型盲点。
                lines.extend(self._format_actionable_elements(artifact.get("current_page")))
                current_page = artifact.get("current_page", {}) if isinstance(artifact.get("current_page"), dict) else {}
                # 显式透出网络空闲/URL变化，避免重复点击未加载完成的元素。
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
        hint = self._next_step_hint(tool_name, artifact)
        if hint:
            lines.append(f"next_step_hint: {hint}")
        return "\n".join(lines)

    def _compact_page_for_text(self, page: Any) -> dict[str, Any]:
        if not isinstance(page, dict):
            return {}
        return {
            "url": page.get("url"),
            "title": page.get("title"),
            "screenshot_path": page.get("screenshot_path"),
        }

    def _next_step_hint(self, tool_name: str, artifact: dict[str, Any]) -> str:
        """Provide concise, scenario-specific guidance for the next LLM action."""
        status = artifact.get("status")
        if status == "error":
            if tool_name in {"browser_click", "browser_fill"}:
                return "Run browser_find to verify selector before retry."
            if tool_name == "browser_extract":
                return "Try browser_screenshot or browser_navigate to recover content."
            return "Check page state and retry with a safer action."

        metadata = artifact.get("metadata", {}) if isinstance(artifact.get("metadata"), dict) else {}
        if tool_name == "browser_find" and metadata.get("count", 0) == 0:
            return "No matching element found; try broader selector or browser_extract first."
        if tool_name == "browser_extract" and not artifact.get("content_preview"):
            return "Extracted content is empty; try browser_screenshot or navigate/reload."
        if tool_name == "browser_navigate":
            current_page = artifact.get("current_page", {}) if isinstance(artifact.get("current_page"), dict) else {}
            if self._count_valid_actionable_elements(current_page) > 0:
                return "Use suggested selectors from actionable_elements for click/fill."
            return "No clear controls found; run browser_extract then browser_find with broader selectors."
        if tool_name == "browser_screenshot":
            return "Use browser_extract to read text after visual confirmation."
        return ""

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
            if not selector and not text and not kind and not placeholder:
                continue
            valid_items.append(
                {
                    "type": kind,
                    "selector": selector,
                    "text": text,
                    "placeholder": placeholder,
                }
            )
        if not valid_items:
            return ["actionable_elements: []"]

        lines = [f"actionable_count: {len(valid_items)}"]
        for idx, item in enumerate(valid_items[:limit], start=1):
            if not isinstance(item, dict):
                continue
            lines.append(
                "actionable_{idx}: type={type} selector={selector} text={text} placeholder={placeholder}".format(
                    idx=idx,
                    type=item.get("type", "-"),
                    selector=item.get("selector", "-"),
                    text=item.get("text", "-"),
                    placeholder=item.get("placeholder", "-"),
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
        if len(content) <= limit:
            return content
        return content[:limit] + "..."
