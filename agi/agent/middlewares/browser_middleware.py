import asyncio
import base64
import logging
import random
import time
from collections.abc import Awaitable, Callable
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
from langchain_core.messages import ToolMessage
from langchain_core.messages.content import create_image_block
from langchain_core.tools import BaseTool, StructuredTool
from langgraph.types import Command
from typing_extensions import TypedDict

from agi.deepagents.middleware._utils import append_to_system_message
from agi.web.browser_backend import PageInfo, StatefulBrowserBackend

logger = logging.getLogger(__name__)

BROWSER_SYSTEM_PROMPT = """## Browser Tools

You have access to a stateful browser session.

- Always navigate before interacting with a new website.
- Prefer `browser_find` before `browser_click` or `browser_fill` when selectors are uncertain.
- Use `browser_extract` as the primary content-reading tool; it prioritizes full-page screenshot OCR before falling back to DOM content.
- Screenshots are primarily used as OCR input, and secondarily for visual debugging or layout verification.
- Large HTML responses may be truncated and optionally evicted to disk.
"""

BROWSER_NAVIGATE_TOOL_DESCRIPTION = """
Navigates the browser to a specific URL.

Assume this tool can access most public websites. If the user provides a URL, assume it is valid unless known otherwise.
This tool maintains the current browser session state (cookies, local storage, history).

Usage:
- Stateful navigation updates the current page context.
- Automatically waits for `domcontentloaded` and then for the page to stabilize.
- Returns a summary of the page (title, URL, preview text) instead of dumping full HTML.
- To inspect the full HTML/text, call `browser_extract` after navigation.
"""

BROWSER_CLICK_TOOL_DESCRIPTION = """
Clicks an element on the current page using a CSS selector.

Usage:
- Call `browser_find` first if the selector is uncertain.
- The tool waits for the page to stabilize after the click, including possible navigation.
- Returns an error if the element is missing, hidden, or not interactable.
"""

BROWSER_FILL_TOOL_DESCRIPTION = """
Fills a text input field on the current page with the provided text.

Usage:
- The selector should target an `<input>`, `<textarea>`, or editable field.
- Existing content is cleared before the new text is entered.
- Returns the updated page context after the field is filled.
"""

BROWSER_EXTRACT_TOOL_DESCRIPTION = """
Extracts page content from the current page, prioritizing OCR.

Usage:
- The tool first captures a full-page screenshot and uses OCR to read page content.
- DOM text and HTML are treated as fallback/reference data when OCR is unavailable or incomplete.
- If the HTML content exceeds the limit, only previews are returned.
- When an eviction handler is configured, large HTML is written to disk and the file path is returned.
"""

BROWSER_SCREENSHOT_TOOL_DESCRIPTION = """
Captures a screenshot of the current browser page.

Usage:
- The screenshot is primarily intended to feed OCR-based page extraction.
- It can also be used for visual debugging and page verification.
- Returns a multimodal image content block the model can inspect.
- The response also includes metadata such as the current URL and saved file path.
"""

BROWSER_FIND_TOOL_DESCRIPTION = """
Finds elements on the current page matching a CSS selector.

Usage:
- Returns text and attributes for up to the first 10 matches.
- Useful for selector discovery before clicking or filling fields.
"""


class BrowserState(AgentState):
    """State for browser middleware."""

    browser_last_result: NotRequired[dict[str, Any]]


class BrowserToolArtifact(TypedDict):
    """Structured browser tool payload returned alongside text results."""

    status: str
    url: str
    title: str | None
    metadata: dict[str, Any]
    content_preview: str
    screenshot_path: NotRequired[str | None]
    full_content_path: NotRequired[str]
    is_truncated: NotRequired[bool]
    text_preview: NotRequired[str]
    html_preview: NotRequired[str]
    history_length: NotRequired[int]
    error: NotRequired[str]
    ocr_text_preview: NotRequired[str]


class BrowserMiddleware(AgentMiddleware):
    """Stateful browser middleware with filesystem-style tool wrappers."""

    def __init__(
        self,
        backend: StatefulBrowserBackend,
        ocr_engine: Any | None = None,
        max_retries: int = 3,
        enable_ocr_fallback: bool = True,
        content_token_limit: int = 15_000,
        eviction_handler: Callable[[str], str] | None = None,
        system_prompt: str | None = None,
    ):
        super().__init__()
        self.backend = backend
        self.ocr = ocr_engine
        self.max_retries = max_retries
        self.enable_ocr = enable_ocr_fallback
        self.content_limit = content_token_limit
        self.eviction_handler = eviction_handler
        self._custom_system_prompt = system_prompt
        self._last_result: PageInfo | None = None
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

    def wrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], ModelResponse[ResponseT]],
    ) -> ModelResponse[ResponseT]:
        system_prompt = self._custom_system_prompt or BROWSER_SYSTEM_PROMPT
        if system_prompt:
            request = request.override(system_message=append_to_system_message(request.system_message, system_prompt))
        return handler(request)

    async def awrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], Awaitable[ModelResponse[ResponseT]]],
    ) -> ModelResponse[ResponseT]:
        system_prompt = self._custom_system_prompt or BROWSER_SYSTEM_PROMPT
        if system_prompt:
            request = request.override(system_message=append_to_system_message(request.system_message, system_prompt))
        return await handler(request)

    def _create_navigate_tool(self) -> BaseTool:
        async def async_navigate(
            url: Annotated[str, "URL to open in the current browser session."],
            runtime: ToolRuntime[None, BrowserState],
        ) -> Command:
            result = await self._execute_with_retry("navigate", url=url)
            return self._command_for_result("browser_navigate", runtime.tool_call_id, self._format_page_result(result))

        return StructuredTool.from_function(
            name="browser_navigate",
            description=BROWSER_NAVIGATE_TOOL_DESCRIPTION,
            coroutine=async_navigate,
        )

    def _create_click_tool(self) -> BaseTool:
        async def async_click(
            selector: Annotated[str, "CSS selector to click on the current page."],
            runtime: ToolRuntime[None, BrowserState],
        ) -> Command:
            result = await self._execute_with_retry("click", selector=selector)
            return self._command_for_result("browser_click", runtime.tool_call_id, self._format_page_result(result))

        return StructuredTool.from_function(
            name="browser_click",
            description=BROWSER_CLICK_TOOL_DESCRIPTION,
            coroutine=async_click,
        )

    def _create_fill_tool(self) -> BaseTool:
        async def async_fill(
            selector: Annotated[str, "CSS selector for the field to fill."],
            text: Annotated[str, "Text to enter into the selected field."],
            runtime: ToolRuntime[None, BrowserState],
        ) -> Command:
            result = await self._execute_with_retry("fill", selector=selector, text=text)
            return self._command_for_result("browser_fill", runtime.tool_call_id, self._format_page_result(result))

        return StructuredTool.from_function(
            name="browser_fill",
            description=BROWSER_FILL_TOOL_DESCRIPTION,
            coroutine=async_fill,
        )

    def _create_extract_tool(self) -> BaseTool:
        async def async_extract(runtime: ToolRuntime[None, BrowserState]) -> Command:
            artifact = await self._tool_extract()
            return self._command_for_result("browser_extract", runtime.tool_call_id, artifact)

        return StructuredTool.from_function(
            name="browser_extract",
            description=BROWSER_EXTRACT_TOOL_DESCRIPTION,
            coroutine=async_extract,
        )

    def _create_screenshot_tool(self) -> BaseTool:
        async def async_screenshot(runtime: ToolRuntime[None, BrowserState]) -> ToolMessage | Command:
            return await self._tool_screenshot(runtime.tool_call_id)

        return StructuredTool.from_function(
            name="browser_screenshot",
            description=BROWSER_SCREENSHOT_TOOL_DESCRIPTION,
            coroutine=async_screenshot,
        )

    def _create_find_tool(self) -> BaseTool:
        async def async_find(
            selector: Annotated[str, "CSS selector to query on the current page."],
            runtime: ToolRuntime[None, BrowserState],
        ) -> Command:
            artifact = await self._tool_find(selector)
            return self._command_for_result("browser_find", runtime.tool_call_id, artifact)

        return StructuredTool.from_function(
            name="browser_find",
            description=BROWSER_FIND_TOOL_DESCRIPTION,
            coroutine=async_find,
        )

    async def _tool_extract(self) -> BrowserToolArtifact:
        """Extract page content from the last successfully loaded page, prioritizing OCR."""
        if self._last_result is None:
            return self._error_artifact("No page loaded. Please navigate first.")

        html = self._last_result.html or ""
        text = self._last_result.text or ""
        ocr_text, screenshot_path = await self._extract_content_with_ocr()
        if not ocr_text and not html and not text:
            return self._error_artifact(
                "Page content is empty and OCR extraction was unavailable.",
                url=self._last_result.url,
                metadata=self._last_result.metadata,
                screenshot_path=screenshot_path,
            )

        primary_content = ocr_text or text or html
        artifact: BrowserToolArtifact = {
            "status": "success",
            "url": self._last_result.url,
            "title": self._last_result.title,
            "metadata": {
                **dict(self._last_result.metadata),
                "ocr_priority": True,
                "ocr_applied": bool(ocr_text),
            },
            "content_preview": self._build_preview(primary_content),
            "history_length": len(self.backend.get_history()),
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

        return artifact

    async def _tool_find(self, selector: str) -> BrowserToolArtifact:
        """Find candidate elements on the current page."""
        matches = await self.backend.find_elements(selector)
        metadata = {
            "selector": selector,
            "count": len(matches),
            "matches": [{"text": match.text, "attrs": match.attributes} for match in matches[:10]],
        }
        return {
            "status": "success",
            "url": self._last_result.url if self._last_result else "",
            "title": self._last_result.title if self._last_result else None,
            "metadata": metadata,
            "content_preview": f"Found {len(matches)} element(s) for selector: {selector}",
            "history_length": len(self.backend.get_history()),
        }

    async def _extract_content_with_ocr(self) -> tuple[str, str | None]:
        """Capture a full-page screenshot and use OCR as the primary extraction path."""
        if not self.enable_ocr or self.ocr is None:
            return "", self._last_result.screenshot_path if self._last_result else None

        screenshot = await self.backend.read_screenshot_bytes(full_page=True)
        if screenshot is None:
            return "", self._last_result.screenshot_path if self._last_result else None

        screenshot_path, image_bytes = screenshot
        try:
            ocr_text = await self.ocr.parse(image_bytes)
        except Exception:
            logger.exception("OCR extraction failed for %s", self._last_result.url if self._last_result else "current page")
            return "", screenshot_path

        normalized_text = str(ocr_text).strip()
        if self._last_result is not None and normalized_text:
            self._last_result.text = normalized_text
            self._last_result.screenshot_path = screenshot_path
            self._last_result.metadata = {
                **self._last_result.metadata,
                "ocr_applied": True,
                "ocr_text_length": len(normalized_text),
                "ocr_screenshot_path": screenshot_path,
            }
        return normalized_text, screenshot_path

    async def _tool_screenshot(self, tool_call_id: str) -> ToolMessage | Command:
        """Capture a screenshot and return a multimodal tool response."""
        screenshot = await self.backend.read_screenshot_bytes(full_page=True)
        if screenshot is None:
            artifact = self._error_artifact("Failed to take screenshot")
            return self._command_for_result("browser_screenshot", tool_call_id, artifact)

        screenshot_path, image_bytes = screenshot
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
        current_url = self._last_result.url if self._last_result else ""
        text = f"Screenshot captured for {current_url}" if current_url else "Screenshot captured"
        artifact: BrowserToolArtifact = {
            "status": "success",
            "url": current_url,
            "title": self._last_result.title if self._last_result else None,
            "metadata": {"screenshot_path": screenshot_path},
            "content_preview": text,
            "screenshot_path": screenshot_path,
            "history_length": len(self.backend.get_history()),
        }
        return Command(
            update={
                "browser_last_result": artifact,
                "messages": [
                    ToolMessage(
                        content=text,
                        content_blocks=[create_image_block(base64=image_b64, mime_type="image/png")],
                        name="browser_screenshot",
                        tool_call_id=tool_call_id,
                        additional_kwargs={"artifact": artifact},
                    )
                ],
            }
        )

    async def _execute_with_retry(self, action: str, **kwargs: Any) -> PageInfo:
        """Execute a browser action with retries and optional OCR fallback."""
        last_error: Exception | None = None

        for attempt in range(self.max_retries):
            try:
                if attempt > 0:
                    delay = random.uniform(1.0, 3.0)
                    logger.info("Retrying browser action '%s' after %.2fs", action, delay)
                    await asyncio.sleep(delay)

                started_at = time.perf_counter()
                result = await self._dispatch_action(action, **kwargs)
                logger.info("Browser action '%s' completed in %.2fs", action, time.perf_counter() - started_at)

                if result.metadata.get("error"):
                    msg = str(result.metadata["error"])
                    raise RuntimeError(msg)

                await self._maybe_apply_ocr(result)
                self._last_result = result
                return result
            except Exception as exc:
                last_error = exc
                logger.warning("Browser action '%s' attempt %s/%s failed: %s", action, attempt + 1, self.max_retries, exc)
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2**attempt)

        error_url = kwargs.get("url") or (self._last_result.url if self._last_result else "unknown")
        return PageInfo(
            url=error_url,
            title=None,
            html=None,
            text=None,
            screenshot_path=None,
            metadata={"error": f"Failed after {self.max_retries} retries: {last_error}"},
        )

    async def _dispatch_action(self, action: str, **kwargs: Any) -> PageInfo:
        if action == "navigate":
            return await self.backend.navigate(kwargs["url"])
        if action == "click":
            return await self.backend.click(kwargs["selector"])
        if action == "fill":
            return await self.backend.fill(kwargs["selector"], kwargs["text"])
        msg = f"Unknown action: {action}"
        raise ValueError(msg)

    async def _maybe_apply_ocr(self, result: PageInfo) -> None:
        if not self.enable_ocr or self.ocr is None:
            return
        if result.html and len(result.html) >= 100:
            return

        screenshot = await self.backend.read_screenshot_bytes(full_page=True)
        if screenshot is None:
            return

        screenshot_path, image_bytes = screenshot
        logger.warning("Applying OCR fallback for %s", result.url)
        ocr_text = await self.ocr.parse(image_bytes)
        result.text = str(ocr_text)
        result.screenshot_path = screenshot_path
        result.metadata = {
            **result.metadata,
            "ocr_applied": True,
            "ocr_text_length": len(result.text),
        }

    def _format_page_result(self, result: PageInfo) -> BrowserToolArtifact:
        if result.metadata.get("error"):
            return self._error_artifact(
                str(result.metadata["error"]),
                url=result.url,
                title=result.title,
                metadata=result.metadata,
                screenshot_path=result.screenshot_path,
            )

        preview_source = result.text or result.html or ""
        artifact: BrowserToolArtifact = {
            "status": "success",
            "url": result.url,
            "title": result.title,
            "metadata": dict(result.metadata),
            "content_preview": self._build_preview(preview_source),
            "history_length": len(self.backend.get_history()),
        }
        if result.screenshot_path:
            artifact["screenshot_path"] = result.screenshot_path
        return artifact

    def _error_artifact(
        self,
        error: str,
        *,
        url: str = "",
        title: str | None = None,
        metadata: dict[str, Any] | None = None,
        screenshot_path: str | None = None,
    ) -> BrowserToolArtifact:
        artifact: BrowserToolArtifact = {
            "status": "error",
            "url": url,
            "title": title,
            "metadata": dict(metadata or {}),
            "content_preview": error,
            "error": error,
            "history_length": len(self.backend.get_history()),
        }
        if screenshot_path:
            artifact["screenshot_path"] = screenshot_path
        return artifact

    def _command_for_result(self, tool_name: str, tool_call_id: str, artifact: BrowserToolArtifact) -> Command:
        text = self._artifact_to_text(artifact)
        return Command(
            update={
                "browser_last_result": artifact,
                "messages": [
                    ToolMessage(
                        content=text,
                        name=tool_name,
                        tool_call_id=tool_call_id,
                        additional_kwargs={"artifact": artifact},
                    )
                ],
            }
        )

    def _artifact_to_text(self, artifact: BrowserToolArtifact) -> str:
        lines = [f"status: {artifact['status']}"]
        if artifact.get("url"):
            lines.append(f"url: {artifact['url']}")
        if artifact.get("title"):
            lines.append(f"title: {artifact['title']}")
        if artifact.get("content_preview"):
            lines.append(f"preview: {artifact['content_preview']}")
        if artifact.get("error"):
            lines.append(f"error: {artifact['error']}")
        if artifact.get("screenshot_path"):
            lines.append(f"screenshot_path: {artifact['screenshot_path']}")
        if artifact.get("ocr_text_preview"):
            lines.append(f"ocr_text_preview: {artifact['ocr_text_preview']}")
        if artifact.get("full_content_path"):
            lines.append(f"full_content_path: {artifact['full_content_path']}")
        if artifact.get("metadata"):
            lines.append(f"metadata: {artifact['metadata']}")
        return "\n".join(lines)

    def _build_preview(self, content: str, *, limit: int = 500) -> str:
        if len(content) <= limit:
            return content
        return content[:limit] + "..."
