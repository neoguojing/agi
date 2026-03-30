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
from agi.web.browser_types import BrowserSessionSnapshot, PageInfo

logger = logging.getLogger(__name__)
# middleware 的定位：
# - 对外暴露浏览器工具集；
# - 在模型调用前，把当前浏览器状态同步到 system prompt；
# - 把 backend 的状态结果包装成 agent state / tool messages。

BROWSER_SYSTEM_PROMPT = """## Browser Tools

You have access to a stateful, real-time browser session.

### Core Principles
- **State Awareness**: Always check the current URL and recent events (navigation, DOM updates, user interaction) before acting. Adapt your plan if the page context has changed; do not repeat actions that have already succeeded.
- **Tool Strategy**:
  - Call `browser_navigate` before interacting with a new site.
  - Use `browser_find` to locate selectors before clicking or filling.
  - Use `browser_extract` (OCR prioritized) as the primary way to read content; use `browser_screenshot` for visual layout or debugging.
- **Efficiency**: Avoid redundant clicks or inputs. If the page appears unchanged, re-extract content or verify via screenshot.
- **Large Content**: HTML may be truncated. Rely on extraction tools rather than raw HTML dumps.
"""

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
            self._create_extract_tool(),
            self._create_screenshot_tool(),
            self._create_find_tool(),
        ]

    # 同步模型调用入口：在真正调用模型前，把当前浏览器状态摘要拼到 system prompt。
    def wrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], ModelResponse[ResponseT]],
    ) -> ModelResponse[ResponseT]:
        # Sync hook cannot await live browser state safely; use static guidance.
        system_prompt = self._custom_system_prompt or BROWSER_SYSTEM_PROMPT
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

    def _create_screenshot_tool(self) -> BaseTool:
        async def async_screenshot(runtime: ToolRuntime[None, MiddlewareBrowserState]) -> ToolMessage | Command:
            return await self._tool_screenshot(runtime, runtime.tool_call_id)

        return StructuredTool.from_function(
            name="browser_screenshot",
            description=BROWSER_SCREENSHOT_TOOL_DESCRIPTION,
            coroutine=async_screenshot,
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

    async def _tool_extract(self, runtime: ToolRuntime[None, MiddlewareBrowserState]) -> dict[str, Any]:
        """Extract page content from the last successfully loaded page, prioritizing OCR."""
        user_id = self._resolve_user_id(runtime)
        last_result = await self._session_manager.get_last_result(user_id)
        if not last_result:
            return self._artifact_with_state(
                self._error_artifact("No page loaded. Please navigate first."),
                user_id,
            )

        html = last_result.html or ""
        text = last_result.text or ""
        ocr_text, screenshot_path = await self._extract_content_with_ocr(user_id, last_result)
        if not ocr_text and not html and not text:
            return self._artifact_with_state(
                self._error_artifact(
                    "Page content is empty and OCR extraction was unavailable.",
                    url=last_result.url,
                    metadata=last_result.metadata,
                    screenshot_path=screenshot_path,
                ),
                user_id,
            )

        primary_content = ocr_text or text or html
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

        return self._artifact_with_state(artifact, user_id)

    async def _tool_find(self, runtime: ToolRuntime[None, MiddlewareBrowserState], selector: str) -> dict[str, Any]:
        """Find candidate elements on the current page."""
        user_id = self._resolve_user_id(runtime)
        
        # 使用 session manager 的统一 API
        matches = await self._session_manager.find_elements(user_id, selector)
        state = await self._session_manager.get_state(user_id)
        last_result = await self._session_manager.get_last_result(user_id)
        
        metadata = {
            "selector": selector,
            "count": len(matches),
            "matches": [{"text": match.text, "attrs": match.attributes} for match in matches[:10]],
        }
        return self._artifact_with_state(
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
            artifact = self._artifact_with_state(self._error_artifact("Failed to take screenshot"), user_id)
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
            artifact = self._artifact_with_state(self._error_artifact(f"Screenshot file not found: {screenshot_path}"), user_id)
            return self._command_for_result(
                "browser_screenshot",
                tool_call_id,
                artifact,
                session_state=self._extract_state_from_artifact(artifact),
            )

        # image_b64 = base64.b64encode(image_bytes).decode("utf-8")
        last_result = await self._session_manager.get_last_result(user_id)
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
        artifact = self._artifact_with_state(artifact, user_id)
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

        result.metadata.pop('elements', None)
        current_page = {
            "url": result.url,
            "title": result.title,
            "html": result.html,
            "text": result.text,
            "screenshot_path": result.screenshot_path,
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
        system_prompt = self._custom_system_prompt or BROWSER_SYSTEM_PROMPT
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

    async def _get_live_session_state(self, user_id: str) -> BrowserSessionSnapshot | None:
        return await self._get_canonical_session_state(user_id=user_id)

    async def _get_canonical_session_state(self, user_id: str | None) -> BrowserSessionSnapshot | None:
        """Single source of truth for middleware/browser state normalization."""
        if not user_id:
            return None
        state = await self._session_manager.get_state(user_id)
        if not state:
            return None
        return self._normalize_llm_state(state)

    def _format_browser_state_for_prompt(self, session_state: BrowserSessionSnapshot) -> str:
        """Generate a compact LLM-facing state summary.

        Keep only what helps action planning:
        0) browser state
        1) current page
        2) previous page (if available)
        """
        browser = session_state.get("browser", {}) or {}
        current_page = session_state.get("current_page", {}) or {}
        previous_page = session_state.get("previous_page")
        previous_line = f"previous_page: {previous_page if isinstance(previous_page, dict) else '<none>'}"
        return "\n".join(
            [
                "## Current Browser Session State",
                f"browser: is_open={browser.get('is_open')} is_closed={browser.get('is_closed')}",
                f"current_page: {current_page}",
                previous_line,
                "Use only this state to decide next step: navigate, find, click, fill, extract, screenshot.",
            ]
        )

    def _normalize_llm_state(self, state: dict[str, Any] | None) -> BrowserSessionSnapshot:
        """Normalize arbitrary/raw state to the single LLM-facing schema."""
        source = state or {}
        current_page = source.get("current_page", {})
        previous_page = source.get("previous_page")
        logger.debug(
            "Normalizing browser state: current_page_keys=%s previous_page_type=%s",
            list(current_page.keys()) if isinstance(current_page, dict) else [],
            type(previous_page).__name__,
        )
        return {
            "browser": dict(source.get("browser", {"is_open": False, "is_closed": True})),
            "current_page": dict(current_page) if isinstance(current_page, dict) else {},
            "previous_page": dict(previous_page) if isinstance(previous_page, dict) else previous_page,
        }

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
        text = self._artifact_to_text(artifact)
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

    def _artifact_to_text(self, artifact: dict[str, Any]) -> str:
        lines = [f"status: {artifact['status']}"]
        current_page = artifact.get("current_page", {})
        if isinstance(current_page, dict) and current_page.get("url"):
            lines.append(f"url: {current_page['url']}")
        if isinstance(current_page, dict) and current_page.get("title"):
            lines.append(f"title: {current_page['title']}")
        if artifact.get("content_preview"):
            lines.append(f"preview: {artifact['content_preview']}")
        if artifact.get("error"):
            lines.append(f"error: {artifact['error']}")
        if isinstance(current_page, dict) and current_page.get("screenshot_path"):
            lines.append(f"screenshot_path: {current_page['screenshot_path']}")
        if artifact.get("ocr_text_preview"):
            lines.append(f"ocr_text_preview: {artifact['ocr_text_preview']}")
        if artifact.get("full_content_path"):
            lines.append(f"full_content_path: {artifact['full_content_path']}")
        if artifact.get("current_page"):
            lines.append(f"current_page: {artifact['current_page']}")
        if artifact.get("previous_page"):
            lines.append(f"previous_page: {artifact['previous_page']}")
        if artifact.get("metadata"):
            lines.append(f"metadata: {artifact['metadata']}")
        return "\n".join(lines)

    def _build_preview(self, content: str, *, limit: int = 500) -> str:
        if len(content) <= limit:
            return content
        return content[:limit] + "..."
