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
from langchain_core.messages import ToolMessage
from langchain_core.messages.content import create_image_block
from langchain_core.tools import BaseTool, StructuredTool
from langgraph.types import Command
from typing_extensions import NotRequired, TypedDict

from agi.config import BROWSER_STORAGE_PATH
from agi.deepagents.middleware._utils import append_to_system_message
from agi.web.browser_backend import BrowserBackendPool, PageInfo, UserBrowserSession

MAX_PROMPT_EVENTS = 8

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
    browser_session_state: NotRequired[dict[str, Any]]



class BrowserSessionState(TypedDict):
    """Serializable browser/session state exposed back to the agent."""

    user_id: str
    storage_dir: str
    browser: dict[str, Any]
    context: dict[str, Any]
    page: dict[str, Any]
    history_length: int
    recent_events: list[dict[str, Any]]
    last_event: NotRequired[dict[str, Any] | None]
    event_version: int


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
        storage_dir: str = BROWSER_STORAGE_PATH,
        ocr_engine: Any | None = None,
        max_retries: int = 3,
        enable_ocr_fallback: bool = True,
        content_token_limit: int = 15_000,
        eviction_handler: Callable[[str], str] | None = None,
        system_prompt: str | None = None,
        idle_timeout_seconds: float = 60.0,
    ):
        super().__init__()
        self._session_pool = BrowserBackendPool(
            storage_dir=storage_dir,
            idle_timeout_seconds=idle_timeout_seconds,
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

    def wrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], ModelResponse[ResponseT]],
    ) -> ModelResponse[ResponseT]:
        system_prompt = self._build_model_system_prompt(request)
        if system_prompt:
            request = request.override(system_message=append_to_system_message(request.system_message, system_prompt))
        return handler(request)

    async def awrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], Awaitable[ModelResponse[ResponseT]]],
    ) -> ModelResponse[ResponseT]:
        system_prompt = self._build_model_system_prompt(request)
        if system_prompt:
            request = request.override(system_message=append_to_system_message(request.system_message, system_prompt))
        return await handler(request)

    def _create_navigate_tool(self) -> BaseTool:
        async def async_navigate(
            url: Annotated[str, "URL to open in the current browser session."],
            runtime: ToolRuntime[None, BrowserState],
        ) -> Command:
            result = await self._execute_with_retry(runtime, "navigate", url=url)
            return self._command_for_result(
                "browser_navigate",
                runtime.tool_call_id,
                self._format_page_result(result),
                session_state=self._session_state_from_result(result),
            )

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
            result = await self._execute_with_retry(runtime, "click", selector=selector)
            return self._command_for_result(
                "browser_click",
                runtime.tool_call_id,
                self._format_page_result(result),
                session_state=self._session_state_from_result(result),
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
            runtime: ToolRuntime[None, BrowserState],
        ) -> Command:
            result = await self._execute_with_retry(runtime, "fill", selector=selector, text=text)
            return self._command_for_result(
                "browser_fill",
                runtime.tool_call_id,
                self._format_page_result(result),
                session_state=self._session_state_from_result(result),
            )

        return StructuredTool.from_function(
            name="browser_fill",
            description=BROWSER_FILL_TOOL_DESCRIPTION,
            coroutine=async_fill,
        )

    def _create_extract_tool(self) -> BaseTool:
        async def async_extract(runtime: ToolRuntime[None, BrowserState]) -> Command:
            artifact = await self._tool_extract(runtime)
            return self._command_for_result(
                "browser_extract",
                runtime.tool_call_id,
                artifact,
                session_state=self._session_state_from_artifact(artifact),
            )

        return StructuredTool.from_function(
            name="browser_extract",
            description=BROWSER_EXTRACT_TOOL_DESCRIPTION,
            coroutine=async_extract,
        )

    def _create_screenshot_tool(self) -> BaseTool:
        async def async_screenshot(runtime: ToolRuntime[None, BrowserState]) -> ToolMessage | Command:
            return await self._tool_screenshot(runtime, runtime.tool_call_id)

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
            artifact = await self._tool_find(runtime, selector)
            return self._command_for_result(
                "browser_find",
                runtime.tool_call_id,
                artifact,
                session_state=self._session_state_from_artifact(artifact),
            )

        return StructuredTool.from_function(
            name="browser_find",
            description=BROWSER_FIND_TOOL_DESCRIPTION,
            coroutine=async_find,
        )

    async def _tool_extract(self, runtime: ToolRuntime[None, BrowserState]) -> BrowserToolArtifact:
        """Extract page content from the last successfully loaded page, prioritizing OCR."""
        async with self._session_for_user(runtime) as session:
            if session.last_result is None:
                return self._artifact_with_state(
                    self._error_artifact("No page loaded. Please navigate first."),
                    session,
                )

            html = session.last_result.html or ""
            text = session.last_result.text or ""
            ocr_text, screenshot_path = await self._extract_content_with_ocr(session)
            if not ocr_text and not html and not text:
                return self._artifact_with_state(
                    self._error_artifact(
                        "Page content is empty and OCR extraction was unavailable.",
                        url=session.last_result.url,
                        metadata=session.last_result.metadata,
                        screenshot_path=screenshot_path,
                    ),
                    session,
                )

            primary_content = ocr_text or text or html
            artifact: BrowserToolArtifact = {
                "status": "success",
                "url": session.last_result.url,
                "title": session.last_result.title,
                "metadata": {
                    **dict(session.last_result.metadata),
                    "ocr_priority": True,
                    "ocr_applied": bool(ocr_text),
                },
                "content_preview": self._build_preview(primary_content),
                "history_length": len(session.backend.get_history()),
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

            return self._artifact_with_state(artifact, session)

    async def _tool_find(self, runtime: ToolRuntime[None, BrowserState], selector: str) -> BrowserToolArtifact:
        """Find candidate elements on the current page."""
        async with self._session_for_user(runtime) as session:
            matches = await session.backend.find_elements(selector)
            metadata = {
                "selector": selector,
                "count": len(matches),
                "matches": [{"text": match.text, "attrs": match.attributes} for match in matches[:10]],
            }
            return self._artifact_with_state(
                {
                    "status": "success",
                    "url": session.last_result.url if session.last_result else "",
                    "title": session.last_result.title if session.last_result else None,
                    "metadata": metadata,
                    "content_preview": f"Found {len(matches)} element(s) for selector: {selector}",
                    "history_length": len(session.backend.get_history()),
                },
                session,
            )

    async def _extract_content_with_ocr(self, session: UserBrowserSession) -> tuple[str, str | None]:
        """Capture a full-page screenshot and use OCR as the primary extraction path."""
        if not self.enable_ocr or self.ocr is None:
            return "", session.last_result.screenshot_path if session.last_result else None

        screenshot = await session.backend.read_screenshot_bytes(full_page=True)
        if screenshot is None:
            return "", session.last_result.screenshot_path if session.last_result else None

        screenshot_path, image_bytes = screenshot
        try:
            ocr_text = await self.ocr.parse(image_bytes)
        except Exception:
            logger.exception("OCR extraction failed for %s", session.last_result.url if session.last_result else "current page")
            return "", screenshot_path

        normalized_text = str(ocr_text).strip()
        if session.last_result is not None and normalized_text:
            session.last_result.text = normalized_text
            session.last_result.screenshot_path = screenshot_path
            session.last_result.metadata = {
                **session.last_result.metadata,
                "ocr_applied": True,
                "ocr_text_length": len(normalized_text),
                "ocr_screenshot_path": screenshot_path,
            }
        return normalized_text, screenshot_path

    async def _tool_screenshot(self, runtime: ToolRuntime[None, BrowserState], tool_call_id: str) -> ToolMessage | Command:
        """Capture a screenshot and return a multimodal tool response."""
        async with self._session_for_user(runtime) as session:
            screenshot = await session.backend.read_screenshot_bytes(full_page=True)
            if screenshot is None:
                artifact = self._artifact_with_state(self._error_artifact("Failed to take screenshot"), session)
                return self._command_for_result(
                    "browser_screenshot",
                    tool_call_id,
                    artifact,
                    session_state=self._session_state_from_artifact(artifact),
                )

            screenshot_path, image_bytes = screenshot
            image_b64 = base64.b64encode(image_bytes).decode("utf-8")
            current_url = session.last_result.url if session.last_result else ""
            text = f"Screenshot captured for {current_url}" if current_url else "Screenshot captured"
            artifact: BrowserToolArtifact = {
                "status": "success",
                "url": current_url,
                "title": session.last_result.title if session.last_result else None,
                "metadata": {"screenshot_path": screenshot_path},
                "content_preview": text,
                "screenshot_path": screenshot_path,
                "history_length": len(session.backend.get_history()),
            }
            artifact = self._artifact_with_state(artifact, session)
            session_state = self._session_state_from_artifact(artifact)
            return Command(
                update={
                    "browser_last_result": artifact,
                    "browser_session_state": session_state,
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

    async def _execute_with_retry(self, runtime: ToolRuntime[None, BrowserState], action: str, **kwargs: Any) -> PageInfo:
        """Execute a browser action with retries and optional OCR fallback."""
        last_error: Exception | None = None
        user_id = self._resolve_user_id(runtime)

        async with self._session_pool.session(user_id) as session:
            for attempt in range(self.max_retries):
                try:
                    if attempt > 0:
                        delay = random.uniform(1.0, 3.0)
                        logger.info("Retrying browser action '%s' for user_id=%s after %.2fs", action, user_id, delay)
                        await asyncio.sleep(delay)

                    started_at = time.perf_counter()
                    result = await self._dispatch_action(session, action, **kwargs)
                    logger.info(
                        "Browser action '%s' for user_id=%s completed in %.2fs",
                        action,
                        user_id,
                        time.perf_counter() - started_at,
                    )

                    if result.metadata.get("error"):
                        msg = str(result.metadata["error"])
                        raise RuntimeError(msg)

                    await self._maybe_apply_ocr(session, result)
                    session.last_result = result
                    result.metadata = {
                        **result.metadata,
                        "browser_session_state": self._build_session_state(session),
                    }
                    return result
                except Exception as exc:
                    last_error = exc
                    logger.warning(
                        "Browser action '%s' for user_id=%s attempt %s/%s failed: %s",
                        action,
                        user_id,
                        attempt + 1,
                        self.max_retries,
                        exc,
                    )
                    if attempt < self.max_retries - 1:
                        await asyncio.sleep(2**attempt)

            error_url = kwargs.get("url") or (session.last_result.url if session.last_result else "unknown")
            error_result = PageInfo(
                url=error_url,
                title=None,
                html=None,
                text=None,
                screenshot_path=None,
                metadata={"error": f"Failed after {self.max_retries} retries: {last_error}"},
            )
            error_result.metadata["browser_session_state"] = self._build_session_state(session)
            return error_result

    async def _dispatch_action(self, session: UserBrowserSession, action: str, **kwargs: Any) -> PageInfo:
        if action == "navigate":
            return await session.backend.navigate(kwargs["url"])
        if action == "click":
            return await session.backend.click(kwargs["selector"])
        if action == "fill":
            return await session.backend.fill(kwargs["selector"], kwargs["text"])
        msg = f"Unknown action: {action}"
        raise ValueError(msg)

    async def _maybe_apply_ocr(self, session: UserBrowserSession, result: PageInfo) -> None:
        if not self.enable_ocr or self.ocr is None:
            return
        if result.html and len(result.html) >= 100:
            return

        screenshot = await session.backend.read_screenshot_bytes(full_page=True)
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
            "history_length": int(result.metadata.get("history_length", 0)),
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
            "history_length": 0,
        }
        if screenshot_path:
            artifact["screenshot_path"] = screenshot_path
        return artifact

    def _session_for_user(self, runtime: ToolRuntime[None, BrowserState] | None = None):
        user_id = self._resolve_user_id(runtime)
        return self._session_pool.session(user_id)

    def _resolve_user_id(self, runtime: ToolRuntime[None, BrowserState] | None = None) -> str:
        if runtime is not None:
            context = getattr(runtime, "context", None)
            if getattr(context, "user_id", None):
                return str(context.user_id)
            config = getattr(runtime, "config", {}) or {}
            configurable = config.get("configurable", {})
            if configurable.get("user_id"):
                return str(configurable["user_id"])
        return "default"

    def _build_model_system_prompt(self, request: ModelRequest[ContextT]) -> str:
        system_prompt = self._custom_system_prompt or BROWSER_SYSTEM_PROMPT
        session_state = self._resolve_session_state_for_request(request)
        if not session_state:
            return system_prompt
        return f"{system_prompt}\n\n{self._format_browser_state_for_prompt(session_state)}"

    def _resolve_session_state_for_request(self, request: ModelRequest[ContextT]) -> BrowserSessionState | None:
        state = getattr(request, "state", None) or {}
        if isinstance(state, dict):
            session_state = state.get("browser_session_state")
            if isinstance(session_state, dict):
                user_id = session_state.get("user_id")
                live_state = self._get_live_session_state(str(user_id)) if user_id else None
                return live_state or session_state

        user_id = self._resolve_user_id_from_request(request)
        if not user_id:
            return None
        return self._get_live_session_state(user_id)

    def _resolve_user_id_from_request(self, request: ModelRequest[ContextT]) -> str | None:
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

    def _get_live_session_state(self, user_id: str) -> BrowserSessionState | None:
        session = self._session_pool.get_existing_session(user_id)
        if session is None:
            return None
        return self._build_session_state(session)

    def _format_browser_state_for_prompt(self, session_state: BrowserSessionState) -> str:
        browser = session_state.get("browser", {})
        context = session_state.get("context", {})
        page = session_state.get("page", {})
        recent_events = session_state.get("recent_events", [])[-MAX_PROMPT_EVENTS:]
        recent_lines = []
        for event in recent_events:
            metadata = event.get("metadata", {}) if isinstance(event.get("metadata"), dict) else {}
            target = metadata.get("target") if isinstance(metadata.get("target"), dict) else None
            target_desc = ""
            if target:
                tag = target.get("tag") or "element"
                target_text = target.get("text") or target.get("value") or ""
                target_desc = f" target={tag}:{target_text[:40]}" if target_text else f" target={tag}"
            recent_lines.append(
                f"- #{event.get('seq')} {event.get('type')} url={event.get('url') or metadata.get('url')}{target_desc}"
            )

        tabs = context.get("pages", [])
        tab_lines = []
        for tab in tabs[:5]:
            active_mark = "*" if tab.get("is_active") else "-"
            tab_lines.append(
                f"{active_mark} {tab.get('page_id')} url={tab.get('url')} load={tab.get('load_state')} closed={tab.get('is_closed')}"
            )

        return "\n".join(
            [
                "## Current Browser Session State",
                f"user_id: {session_state.get('user_id')}",
                f"browser_open: {browser.get('is_open')} | browser_closed: {browser.get('is_closed')} | storage_dir: {session_state.get('storage_dir')}",
                f"active_page_url: {page.get('url') or page.get('observed_url')} | load_state: {page.get('load_state')} | title: {page.get('title') or page.get('observed_title')}",
                f"active_page_last_interaction: {page.get('last_interaction')}",
                f"active_page_last_user_event: {page.get('last_user_event')}",
                f"tab_count: {context.get('page_count')} | event_version: {session_state.get('event_version')} | history_length: {session_state.get('history_length')}",
                "tabs:",
                *(tab_lines or ["- none"]),
                "recent_events:",
                *(recent_lines or ["- none"]),
                "Use this live browser state to decide whether to navigate, wait, inspect, click, fill, or recover from a closed browser/page.",
            ]
        )

    def _build_session_state(self, session: UserBrowserSession) -> BrowserSessionState:
        return session.backend.get_state_snapshot(user_id=session.user_id, last_result=session.last_result)

    def _artifact_with_state(self, artifact: BrowserToolArtifact, session: UserBrowserSession) -> BrowserToolArtifact:
        artifact["metadata"] = {
            **dict(artifact.get("metadata", {})),
            "browser_session_state": self._build_session_state(session),
        }
        return artifact

    def _session_state_from_result(self, result: PageInfo) -> BrowserSessionState | None:
        state = result.metadata.get("browser_session_state")
        return state if isinstance(state, dict) else None

    def _session_state_from_artifact(self, artifact: BrowserToolArtifact) -> BrowserSessionState | None:
        metadata = artifact.get("metadata", {})
        state = metadata.get("browser_session_state") if isinstance(metadata, dict) else None
        return state if isinstance(state, dict) else None

    def _command_for_result(
        self,
        tool_name: str,
        tool_call_id: str,
        artifact: BrowserToolArtifact,
        session_state: BrowserSessionState | None = None,
    ) -> Command:
        text = self._artifact_to_text(artifact)
        update: dict[str, Any] = {
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
        if session_state is not None:
            update["browser_session_state"] = session_state
        return Command(update=update)

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
