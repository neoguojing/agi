# pdf_middleware.py
# ruff: noqa: E501

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Annotated, Any, Literal , cast
from typing_extensions import TypedDict,NotRequired

from langchain.agents.middleware.types import AgentMiddleware, AgentState, ContextT, ResponseT, ModelRequest, ModelResponse
from langchain.tools import ToolRuntime
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.tools import StructuredTool, BaseTool
from langchain_core.messages import ToolMessage
from langgraph.types import Command

from deepagents.backends import StateBackend
from deepagents.backends.protocol import BackendProtocol
from deepagents.backends.utils import validate_path
from deepagents.middleware._utils import append_to_system_message

from agi.agent.prompt import get_middleware_prompt


# =========================
# State 定义
# =========================

class PDFPageData(TypedDict):
    """Data structure for storing PDF page contents with metadata."""

    page: int
    """Page number (0-indexed)."""

    content: str | None
    """Extracted text content from the page."""

    image_path: str | None
    """Path to rendered image, if available."""

    processed: bool
    """Whether the page has been processed and summarized."""

    summary: NotRequired[str]
    """Generated summary for the page."""


def _pdf_page_reducer(
    left: dict[str, PDFPageData] | None,
    right: dict[str, PDFPageData | None],
) -> dict[str, PDFPageData]:
    """Merge PDF page updates with support for deletions.

    This reducer enables page deletion by treating `None` values in the right
    dictionary as deletion markers. It's designed to work with LangGraph's
    state management where annotated reducers control how state updates merge.

    Args:
        left: Existing pages dictionary. May be `None` during initialization.
        right: New pages dictionary to merge. Pages with `None` values are
            treated as deletion markers and removed from the result.

    Returns:
        Merged dictionary where right overwrites left for matching keys,
        and `None` values in right trigger deletions.

    Example:
        ```python
        existing = {"file.pdf#page_0": PDFPageData(...), "file.pdf#page_1": PDFPageData(...)}
        updates = {"file.pdf#page_1": None, "file.pdf#page_2": PDFPageData(...)}
        result = _pdf_page_reducer(existing, updates)
        # Result: {"file.pdf#page_0": PDFPageData(...), "file.pdf#page_2": PDFPageData(...)}
        ```
    """
    if left is None:
        return {k: v for k, v in right.items() if v is not None}

    result = {**left}
    for k, v in right.items():
        if v is None:
            result.pop(k, None)
        else:
            result[k] = v
    return result


class PDFState(AgentState):
    """State for the PDF middleware."""

    pdf_pages: Annotated[NotRequired[dict[str, PDFPageData]], _pdf_page_reducer]
    """PDF pages in the workspace."""


# =========================
# Middleware
# =========================

class PDFMiddleware(AgentMiddleware[PDFState, ContextT, ResponseT]):
    """Middleware for providing PDF processing tools to an agent.

    This middleware adds PDF processing tools to the agent:
    - `parse_pdf`: Parse PDF(s) to extract page information
    - `read_pdf_page`: Read PDF page text content
    - `render_pdf_page`: Render PDF page to image
    - `prepare_pdf_page`: Prepare page for LLM processing
    - `set_page_summary`: Save summary for processed page
    - `export_pdf`: Export summaries to text file

    Args:
        backend: Backend for file storage.

            If not provided, defaults to `StateBackend` (ephemeral storage in agent state).

    Example:
        ```python
        from agi.agent.middlewares.pdf_middleware import PDFMiddleware

        agent = create_agent(middleware=[PDFMiddleware()])
        ```
    """

    state_schema = PDFState

    def __init__(self, *, backend: BackendProtocol | None = None) -> None:
        """Initialize the PDF middleware.

        Args:
            backend: Backend for file storage, or a factory callable.

                Defaults to StateBackend if not provided.

        Raises:
            ValueError: If `backend` is not a valid BackendProtocol or factory.
        """
        self.backend = backend if backend is not None else (StateBackend)

        self.tools = [
            self._create_parse_pdf_tool(),
            self._create_read_pdf_page_tool(),
            self._create_render_pdf_page_tool(),
            self._create_prepare_pdf_page_tool(),
            self._create_set_page_summary_tool(),
            self._create_export_pdf_tool(),
        ]

    def _get_backend(self, runtime) -> BackendProtocol:
        """Get the resolved backend instance from backend or factory.

        Args:
            runtime: The tool runtime context.

        Returns:
            Resolved backend instance.
        """
        if callable(self.backend):
            return self.backend(runtime)
        return self.backend

    # =========================
    # 1️⃣ parse_pdf
    # =========================

     # =========================
    # helper
    # =========================

    def _cmd(
        self,
        runtime: ToolRuntime,
        msg: str,
        update: dict[str, Any],
    ) -> Command:
        """Create a Command with update messages.

        Args:
            runtime: The tool runtime context.
            msg: The message content.
            update: The state update dictionary.

        Returns:
            A Command object with the update.
        """
        return Command(
            update={
                **update,
                "messages": [
                    ToolMessage(
                        content=msg,
                        tool_call_id=runtime.tool_call_id,
                    )
                ],
            }
        )

    # =========================
    # parse_pdf
    # =========================

    def _create_parse_pdf_tool(self) -> BaseTool:
        """Create the parse_pdf tool.

        Parses PDF(s) to extract page information.

        Returns:
            A structured tool for parsing PDFs.
        """

        async def async_parse(
            path: Annotated[str, "Absolute path to the PDF file(s). Must be absolute, not relative."],
            runtime: ToolRuntime[None, PDFState],
        ) -> Command | str:
            """Parse PDF file(s).

            Args:
                path: Path to the PDF file(s) to parse.
                runtime: The tool runtime context.

            Returns:
                A Command with the parsing result.
            """
            backend = self._get_backend(runtime)
            validated = validate_path(path)

            if validated.endswith(".pdf"):
                files = [validated]
            else:
                infos = await backend.als_info(validated)
                files = [i["path"] for i in infos if i["path"].endswith(".pdf")]

            import pdfplumber

            updates: dict[str, PDFPageData] = {}
            total = 0

            for f in files:
                with pdfplumber.open(f) as pdf:
                    for i, _ in enumerate(pdf.pages):
                        key = f"{f}#page_{i}"
                        updates[key] = {
                            "page": i,
                            "content": None,
                            "image_path": None,
                            "processed": False,
                        }
                        total += 1

            return self._cmd(runtime, f"Parsed {len(files)} PDFs, {total} pages", {"pdf_pages": updates})

        def sync_parse(
            path: Annotated[str, "Absolute path to the PDF file(s). Must be absolute, not relative."],
            runtime: ToolRuntime[None, PDFState],
        ) -> Command | str:
            """Synchronous wrapper for parse_pdf tool."""
            return asyncio.run(async_parse(path, runtime))

        return StructuredTool.from_function(
            name="parse_pdf",
            description="Parse PDF(s)",
            func=sync_parse,
            coroutine=async_parse,
        )

    # =========================
    # read_pdf_page
    # =========================

    def _create_read_pdf_page_tool(self) -> BaseTool:
        """Create the read_pdf_page tool.

        Reads PDF page text content.

        Returns:
            A structured tool for reading PDF pages.
        """

        async def async_read(
            file_path: Annotated[str, "Absolute path to the PDF file. Must be absolute, not relative."],
            page: Annotated[int, "Page number to read (0-indexed)."],
            runtime: ToolRuntime[None, PDFState],
        ) -> Command | str:
            """Read PDF page text content.

            Args:
                file_path: Path to the PDF file.
                page: Page number to read (0-indexed).
                runtime: The tool runtime context.

            Returns:
                A Command with the reading result.
            """
            import pdfplumber

            with pdfplumber.open(file_path) as pdf:
                text = pdf.pages[page].extract_text() or ""

            key = f"{file_path}#page_{page}"

            return self._cmd(
                runtime,
                f"Loaded page {page} (chars={len(text)})",
                {
                    "pdf_pages": {
                        key: {
                            "page": page,
                            "content": text,
                            "image_path": None,
                            "processed": False,
                        }
                    }
                },
            )

        def sync_read(
            file_path: Annotated[str, "Absolute path to the PDF file. Must be absolute, not relative."],
            page: Annotated[int, "Page number to read (0-indexed)."],
            runtime: ToolRuntime[None, PDFState],
        ) -> Command | str:
            """Synchronous wrapper for read_pdf_page tool."""
            return asyncio.run(async_read(file_path, page, runtime))

        return StructuredTool.from_function(
            name="read_pdf_page",
            description="Read PDF page text",
            func=sync_read,
            coroutine=async_read,
        )

    # =========================
    # render_pdf_page
    # =========================

    def _create_render_pdf_page_tool(self) -> BaseTool:
        """Create the render_pdf_page tool.

        Renders PDF page to image.

        Returns:
            A structured tool for rendering PDF pages.
        """

        async def async_render(
            file_path: Annotated[str, "Absolute path to the PDF file. Must be absolute, not relative."],
            page: Annotated[int, "Page number to render (0-indexed)."],
            runtime: ToolRuntime[None, PDFState],
        ) -> Command | str:
            """Render PDF page to image.

            Args:
                file_path: Path to the PDF file.
                page: Page number to render (0-indexed).
                runtime: The tool runtime context.

            Returns:
                A Command with the rendering result.
            """
            import fitz

            doc = fitz.open(file_path)
            pix = doc[page].get_pixmap()

            out_dir = "/tmp/pdf_images"
            Path(out_dir).mkdir(exist_ok=True)

            path = f"{out_dir}/{Path(file_path).name}_{page}.png"
            pix.save(path)

            key = f"{file_path}#page_{page}"

            return self._cmd(
                runtime,
                f"Rendered page {page} -> {path}",
                {
                    "pdf_pages": {
                        key: {
                            "page": page,
                            "content": None,
                            "image_path": path,
                            "processed": False,
                        }
                    }
                },
            )

        def sync_render(
            file_path: Annotated[str, "Absolute path to the PDF file. Must be absolute, not relative."],
            page: Annotated[int, "Page number to render (0-indexed)."],
            runtime: ToolRuntime[None, PDFState],
        ) -> Command | str:
            """Synchronous wrapper for render_pdf_page tool."""
            return asyncio.run(async_render(file_path, page, runtime))

        return StructuredTool.from_function(
            name="render_pdf_page",
            description="Render page to image",
            func=sync_render,
            coroutine=async_render,
        )

    # =========================
    # prepare_pdf_page
    # =========================

    def _create_prepare_pdf_page_tool(self) -> BaseTool:
        """Create the prepare_pdf_page tool.

        Prepares a PDF page for LLM processing.

        Returns:
            A structured tool for preparing PDF pages.
        """

        async def async_prepare(
            key: Annotated[str, "Key identifying the page (file_path#page_N)."],
            runtime: ToolRuntime[None, PDFState],
        ) -> ToolMessage | str:
            """Prepare page for LLM processing.

            Args:
                key: Key identifying the page (file_path#page_N).
                runtime: The tool runtime context.

            Returns:
                A ToolMessage with the preparation result.
            """
            page = runtime.state.get("pdf_pages", {}).get(key)

            if not page:
                return "Error: page not found"

            if page["content"]:
                text = page["content"][:8000]
                return f"Summarize:\n\n{text}"

            if page["image_path"]:
                return f"Analyze image: {page['image_path']}"

            return "Error: no data"

        def sync_prepare(
            key: Annotated[str, "Key identifying the page (file_path#page_N)."],
            runtime: ToolRuntime[None, PDFState],
        ) -> ToolMessage | str:
            """Synchronous wrapper for prepare_pdf_page tool."""
            return asyncio.run(async_prepare(key, runtime))

        return StructuredTool.from_function(
            name="prepare_pdf_page",
            description="Prepare page for LLM",
            func=sync_prepare,
            coroutine=async_prepare,
        )

    # =========================
    # set_page_summary
    # =========================

    def _create_set_page_summary_tool(self) -> BaseTool:
        """Create the set_page_summary tool.

        Saves a summary for a processed page.

        Returns:
            A structured tool for setting page summaries.
        """

        async def async_set(
            key: Annotated[str, "Key identifying the page (file_path#page_N)."],
            summary: Annotated[str, "The summary to save for the page."],
            runtime: ToolRuntime[None, PDFState],
        ) -> Command | str:
            """Save summary for a page.

            Args:
                key: Key identifying the page (file_path#page_N).
                summary: The summary to save.
                runtime: The tool runtime context.

            Returns:
                A Command with the save result.
            """
            return self._cmd(
                runtime,
                f"Saved summary for {key}",
                {
                    "pdf_pages": {
                        key: {
                            "summary": summary,
                            "processed": True,
                        }
                    }
                },
            )

        def sync_set(
            key: Annotated[str, "Key identifying the page (file_path#page_N)."],
            summary: Annotated[str, "The summary to save for the page."],
            runtime: ToolRuntime[None, PDFState],
        ) -> Command | str:
            """Synchronous wrapper for set_page_summary tool."""
            return asyncio.run(async_set(key, summary, runtime))

        return StructuredTool.from_function(
            name="set_page_summary",
            description="Save summary",
            func=sync_set,
            coroutine=async_set,
        )

    # =========================
    # export
    # =========================

    def _create_export_pdf_tool(self) -> BaseTool:
        """Create the export_pdf tool.

        Exports processed page summaries to a text file.

        Returns:
            A structured tool for exporting PDF summaries.
        """

        async def async_export(
            output_path: Annotated[str, "Absolute path where the exported text file should be written. Must be absolute, not relative."],
            runtime: ToolRuntime[None, PDFState],
        ) -> Command | str:
            """Export processed summaries to a text file.

            Args:
                output_path: Path where the exported text file should be written.
                runtime: The tool runtime context.

            Returns:
                A Command with the export result.
            """
            pages = runtime.state.get("pdf_pages", {})

            lines: list[str] = []
            for k in sorted(pages.keys()):
                p = pages[k]
                if "summary" in p:
                    lines.append(f"# Page {p['page']}\n{p['summary']}\n")

            text = "\n".join(lines)

            backend = self._get_backend(runtime)
            res = await backend.awrite(output_path, text)

            if res.error:
                return res.error

            return self._cmd(runtime, f"Exported to {output_path}", {})

        def sync_export(
            output_path: Annotated[str, "Absolute path where the exported text file should be written. Must be absolute, not relative."],
            runtime: ToolRuntime[None, PDFState],
        ) -> Command | str:
            """Synchronous wrapper for export_pdf tool."""
            return asyncio.run(async_export(output_path, runtime))

        return StructuredTool.from_function(
            name="export_pdf",
            description="Export summaries",
            func=sync_export,
            coroutine=async_export,
        )


    def wrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], ModelResponse[ResponseT]],
    ) -> ModelResponse[ResponseT]:
        """Inject PDF system prompt and ensure tool visibility.

        Args:
            request: The model request being processed.
            handler: The handler function to call with the modified request.

        Returns:
            The model response from the handler.
        """
        # 是否存在 PDF tools
        pdf_tool_names = {
            "parse_pdf",
            "read_pdf_page",
            "render_pdf_page",
            "prepare_pdf_page",
            "set_page_summary",
            "export_pdf",
        }

        has_pdf_tools = any(
            (tool.name if hasattr(tool, "name") else tool.get("name")) in pdf_tool_names
            for tool in request.tools
        )

        # 构造 system prompt
        if has_pdf_tools:
            system_prompt = get_middleware_prompt("pdf")
        else:
            system_prompt = ""

        # 注入 system message（复用你已有工具）
        if system_prompt:
            new_system_message = append_to_system_message(
                request.system_message,
                system_prompt,
            )
            request = request.override(system_message=new_system_message)

        return handler(request)

    async def awrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], Awaitable[ModelResponse[ResponseT]]],
    ) -> ModelResponse[ResponseT]:
        """(async) Inject PDF system prompt and ensure tool visibility.

        Args:
            request: The model request being processed.
            handler: The handler function to call with the modified request.

        Returns:
            The model response from the handler.
        """
        pdf_tool_names = {
            "parse_pdf",
            "read_pdf_page",
            "render_pdf_page",
            "prepare_pdf_page",
            "set_page_summary",
            "export_pdf",
        }

        has_pdf_tools = any(
            (tool.name if hasattr(tool, "name") else tool.get("name")) in pdf_tool_names
            for tool in request.tools
        )

        if has_pdf_tools:
            system_prompt = get_middleware_prompt("pdf")
        else:
            system_prompt = ""

        if system_prompt:
            new_system_message = append_to_system_message(
                request.system_message,
                system_prompt,
            )
            request = request.override(system_message=new_system_message)

        return await handler(request)