# pdf_middleware.py
# ruff: noqa: E501

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Annotated, Any
from typing_extensions import TypedDict, NotRequired

from langchain.agents.middleware.types import AgentMiddleware, AgentState, ContextT, ResponseT, ModelRequest, ModelResponse
from langchain.tools import ToolRuntime
from langchain_core.tools import StructuredTool, BaseTool
from langchain_core.messages import ToolMessage
from langgraph.types import Command

from deepagents.backends import StateBackend
from deepagents.backends.protocol import BackendProtocol
from deepagents.backends.utils import validate_path
from deepagents.middleware._utils import append_to_system_message

from agi.agent.prompt import get_middleware_prompt


PDF_TEXT_MIN_CHARS = 40
PDF_DEFAULT_CHUNK_CHARS = 6000
PDF_MAX_CHUNK_CHARS = 12000
PDF_EXTRACT_DIR = Path("/tmp/pdf_extracts")
PDF_IMAGE_DIR = Path("/tmp/pdf_images")
PDF_SUMMARY_DIR = Path("/tmp/pdf_summaries")


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

    text_path: NotRequired[str | None]
    """Path to extracted text persisted outside of agent state."""

    text_chars: NotRequired[int]
    """Number of extracted text characters available for chunking."""

    extraction_method: NotRequired[str]
    """Extraction method selected for the page: text, image, or failed."""

    processed: bool
    """Whether the page has been processed and summarized."""

    summary: NotRequired[str]
    """Generated summary for the page."""

    summary_file: NotRequired[str]
    """Path to an exported page summary."""

    export_path: NotRequired[str]
    """Path to an exported combined summary."""

    error: NotRequired[str]
    """Extraction or processing error details."""


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
    - `read_pdf_page`: Extract page text to a temp file, falling back to an image when text is unusable
    - `render_pdf_page`: Render PDF page to image
    - `prepare_pdf_page`: Prepare a bounded text chunk or image reference for LLM processing
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

    def _get_backend(self, runtime: ToolRuntime) -> BackendProtocol:
        """Get the resolved backend instance from backend or factory.

        Args:
            runtime: The tool runtime context.

        Returns:
            Resolved backend instance.
        """
        if callable(self.backend):
            return self.backend(runtime)
        return self.backend

    def _page_key(self, file_path: str, page: int) -> str:
        """Build the canonical state key for a PDF page."""
        return f"{file_path}#page_{page}"

    def _extract_text_path(self, file_path: str, page: int) -> Path:
        """Return the temp path used to persist extracted text outside LLM state."""
        safe_name = f"{Path(file_path).name}_page_{page}.txt"
        return PDF_EXTRACT_DIR / safe_name

    def _render_page_to_image(self, file_path: str, page: int) -> str:
        """Render a PDF page to an image and return the image path."""
        import fitz

        with fitz.open(file_path) as doc:
            if page >= len(doc):
                raise ValueError(f"Page {page} does not exist (total pages: {len(doc)})")
            pix = doc[page].get_pixmap()

        PDF_IMAGE_DIR.mkdir(exist_ok=True)
        image_path = PDF_IMAGE_DIR / f"{Path(file_path).name}_{page}.png"
        pix.save(str(image_path))
        return str(image_path)

    def _extract_page_to_state(self, file_path: str, page: int) -> PDFPageData:
        """Extract usable page text to disk; fallback to rendered image when text is missing."""
        import pdfplumber

        with pdfplumber.open(file_path) as pdf:
            if page >= len(pdf.pages):
                raise ValueError(f"Page {page} does not exist in {file_path} (total pages: {len(pdf.pages)})")
            text = pdf.pages[page].extract_text() or ""

        normalized = text.strip()
        if len(normalized) >= PDF_TEXT_MIN_CHARS:
            PDF_EXTRACT_DIR.mkdir(exist_ok=True)
            text_path = self._extract_text_path(file_path, page)
            text_path.write_text(normalized, encoding="utf-8")
            return {
                "page": page,
                "content": None,
                "image_path": None,
                "text_path": str(text_path),
                "text_chars": len(normalized),
                "extraction_method": "text",
                "processed": False,
            }

        image_path = self._render_page_to_image(file_path, page)
        return {
            "page": page,
            "content": None,
            "image_path": image_path,
            "text_path": None,
            "text_chars": len(normalized),
            "extraction_method": "image",
            "processed": False,
        }

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
            pdf_errors: list[str] = []

            for f in files:
                try:
                    with pdfplumber.open(f) as pdf:
                        for i, _ in enumerate(pdf.pages):
                            key = self._page_key(f, i)
                            updates[key] = {
                                "page": i,
                                "content": None,
                                "image_path": None,
                                "text_path": None,
                                "text_chars": 0,
                                "extraction_method": "pending",
                                "processed": False,
                            }
                            total += 1
                except Exception as e:
                    key = self._page_key(f, 0)
                    updates[key] = {
                        "page": 0,
                        "content": None,
                        "image_path": None,
                        "text_path": None,
                        "text_chars": 0,
                        "extraction_method": "failed",
                        "processed": False,
                        "error": f"Failed to open PDF: {e}",
                    }
                    total += 1
                    pdf_errors.append(f)

            if total == 0:
                return f"Error: No PDF files found at path: {validated}"

            if pdf_errors:
                return self._cmd(
                    runtime,
                    f"Parsed {len(files)} PDFs, {total} pages (some had errors: {', '.join(pdf_errors)})",
                    {"pdf_pages": updates},
                )

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

        Extracts PDF page text to a temp file instead of storing the entire page
        in agent state. If text extraction is unusable, renders the page to an
        image so a vision-capable LLM can extract the page content later.

        Returns:
            A structured tool for extracting PDF page content.
        """

        async def async_read(
            file_path: Annotated[str, "Absolute path to the PDF file. Must be absolute, not relative."],
            page: Annotated[int, "Page number to read (0-indexed)."],
            runtime: ToolRuntime[None, PDFState],
        ) -> Command | str:
            """Extract text or prepare an image fallback for one PDF page."""
            import pdfplumber

            try:
                page_data = self._extract_page_to_state(file_path, page)
            except FileNotFoundError:
                return f"Error: File not found: {file_path}"
            except pdfplumber.pdf.PdfReadError as e:
                return f"Error: Invalid PDF file: {file_path} - {e}"
            except Exception as e:
                return f"Error: Failed to extract page {page}: {e}"

            key = self._page_key(file_path, page)
            if page_data.get("extraction_method") == "text":
                msg = (
                    f"Extracted page {page} text to {page_data['text_path']} "
                    f"(chars={page_data.get('text_chars', 0)}). Use prepare_pdf_page with chunk_index to summarize bounded chunks."
                )
            else:
                msg = (
                    f"Text was unavailable or too short on page {page}; rendered image fallback -> "
                    f"{page_data['image_path']}. Use prepare_pdf_page for vision extraction/summarization."
                )

            return self._cmd(runtime, msg, {"pdf_pages": {key: page_data}})

        def sync_read(
            file_path: Annotated[str, "Absolute path to the PDF file. Must be absolute, not relative."],
            page: Annotated[int, "Page number to read (0-indexed)."],
            runtime: ToolRuntime[None, PDFState],
        ) -> Command | str:
            """Synchronous wrapper for read_pdf_page tool."""
            return asyncio.run(async_read(file_path, page, runtime))

        return StructuredTool.from_function(
            name="read_pdf_page",
            description="Extract one PDF page to text chunks, or render an image fallback when text is unavailable",
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
            """Render PDF page to image."""
            try:
                image_path = self._render_page_to_image(file_path, page)
            except FileNotFoundError:
                return f"Error: File not found: {file_path}"
            except Exception as e:
                return f"Error: Failed to render page {page}: {e}"

            key = self._page_key(file_path, page)

            return self._cmd(
                runtime,
                f"Rendered page {page} -> {image_path}",
                {
                    "pdf_pages": {
                        key: {
                            "page": page,
                            "content": None,
                            "image_path": image_path,
                            "text_path": None,
                            "text_chars": 0,
                            "extraction_method": "image",
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

        Prepares only one bounded text chunk, or one image reference, for LLM
        processing. This is the context guardrail: never pass a whole PDF or a
        whole long page into the model.

        Returns:
            A structured tool for preparing bounded PDF content.
        """

        async def async_prepare(
            key: Annotated[str, "Key identifying the page (file_path#page_N)."],
            runtime: ToolRuntime[None, PDFState],
            chunk_index: Annotated[int, "0-based chunk index for text pages."] = 0,
            max_chars: Annotated[int, "Maximum text characters to return for this chunk."] = PDF_DEFAULT_CHUNK_CHARS,
        ) -> str:
            """Prepare a bounded text chunk or image fallback for LLM processing."""
            page = runtime.state.get("pdf_pages", {}).get(key)

            if not page:
                return "Error: page not found"

            max_chars = max(1, min(max_chars, PDF_MAX_CHUNK_CHARS))
            chunk_index = max(0, chunk_index)

            text_path = page.get("text_path")
            if text_path:
                try:
                    text = Path(text_path).read_text(encoding="utf-8")
                except Exception as e:
                    return f"Error: failed to read extracted text for {key}: {e}"

                start = chunk_index * max_chars
                end = start + max_chars
                if start >= len(text):
                    total_chunks = max(1, (len(text) + max_chars - 1) // max_chars)
                    return f"Error: chunk_index {chunk_index} out of range for {key}; total_chunks={total_chunks}"

                chunk = text[start:end]
                total_chunks = max(1, (len(text) + max_chars - 1) // max_chars)
                return (
                    f"Summarize PDF page chunk. Do not request the whole PDF.\n"
                    f"Page key: {key}\n"
                    f"Chunk: {chunk_index + 1}/{total_chunks}\n"
                    f"Character range: {start}-{min(end, len(text))} of {len(text)}\n\n"
                    f"{chunk}"
                )

            legacy_content = page.get("content")
            if legacy_content:
                start = chunk_index * max_chars
                end = start + max_chars
                return legacy_content[start:end]

            image_path = page.get("image_path")
            if image_path:
                return (
                    "Text extraction was unavailable for this PDF page. "
                    "Use the page image to extract the visible content and produce a concise structured summary.\n"
                    f"Page key: {key}\n"
                    f"Image path: {image_path}"
                )

            return "Error: no extracted text or image fallback. Call read_pdf_page first."

        def sync_prepare(
            key: Annotated[str, "Key identifying the page (file_path#page_N)."],
            runtime: ToolRuntime[None, PDFState],
            chunk_index: Annotated[int, "0-based chunk index for text pages."] = 0,
            max_chars: Annotated[int, "Maximum text characters to return for this chunk."] = PDF_DEFAULT_CHUNK_CHARS,
        ) -> str:
            """Synchronous wrapper for prepare_pdf_page tool."""
            return asyncio.run(async_prepare(key, runtime, chunk_index, max_chars))

        return StructuredTool.from_function(
            name="prepare_pdf_page",
            description="Prepare one bounded text chunk or one image fallback for LLM summarization",
            func=sync_prepare,
            coroutine=async_prepare,
        )

    # =========================
    # set_page_summary
    # =========================

    def _create_set_page_summary_tool(self) -> BaseTool:
        """Create the set_page_summary tool.

        Saves the LLM-generated final summary for one processed page. Chunk
        summaries should be merged by the LLM before this tool is called, so
        export_pdf writes summaries rather than raw extracted PDF content.

        Returns:
            A structured tool for setting page summaries.
        """

        async def async_set(
            key: Annotated[str, "Key identifying the page (file_path#page_N)."],
            summary: Annotated[str, "The LLM-generated final summary to save for the page."],
            runtime: ToolRuntime[None, PDFState],
        ) -> Command | str:
            """Save an LLM-generated summary for a page and persist it to disk."""
            pages = runtime.state.get("pdf_pages", {})
            key = key or ""
            page = pages.get(key)

            if not page:
                return f"Error: Page not found: {key}"

            file_path = key.split("#page_")[0]
            PDF_SUMMARY_DIR.mkdir(exist_ok=True)
            summary_file = PDF_SUMMARY_DIR / f"{Path(file_path).name}_page_{page['page']}.txt"
            text_content = f"# Page {page['page']}\n{summary}\n"

            try:
                backend = self._get_backend(runtime)
                res = await backend.awrite(str(summary_file), text_content)
                if res.error:
                    return f"Error: Failed to write summary to {summary_file}: {res.error}"
            except Exception as e:
                return f"Error: {e}"

            updated_page: PDFPageData = {
                **page,
                "summary": summary,
                "summary_file": str(summary_file),
                "processed": True,
            }

            return self._cmd(
                runtime,
                f"Saved LLM summary for {key} to {summary_file}",
                {"pdf_pages": {key: updated_page}},
            )

        def sync_set(
            key: Annotated[str, "Key identifying the page (file_path#page_N)."],
            summary: Annotated[str, "The LLM-generated final summary to save for the page."],
            runtime: ToolRuntime[None, PDFState],
        ) -> Command | str:
            """Synchronous wrapper for set_page_summary tool."""
            return asyncio.run(async_set(key, summary, runtime))

        return StructuredTool.from_function(
            name="set_page_summary",
            description="Save an LLM-generated page summary; never pass raw extracted PDF text as the summary",
            func=sync_set,
            coroutine=async_set,
        )

    # =========================
    # export
    # =========================

    def _create_export_pdf_tool(self) -> BaseTool:
        """Create the export_pdf tool.

        Exports all processed PDF page summaries to a single text file. Raw
        extracted page text is intentionally excluded from this output.

        Returns:
            A structured tool for exporting PDF summaries.
        """

        async def async_export(
            output_path: Annotated[str, "Absolute path where the exported text file should be written. Must be absolute, not relative."],
            runtime: ToolRuntime[None, PDFState],
        ) -> Command | str:
            """Export all processed LLM summaries to a text file."""
            pages = runtime.state.get("pdf_pages", {})

            page_summaries: list[tuple[str, PDFPageData]] = [
                (key, page)
                for key, page in pages.items()
                if page.get("processed") and page.get("summary") and "#page_" in key
            ]
            page_summaries.sort(key=lambda item: (item[0].split("#page_")[0], item[1].get("page", 0)))

            if not page_summaries:
                return "Warning: No processed LLM summaries found. Summarize pages with prepare_pdf_page, then call set_page_summary before export_pdf."

            lines: list[str] = []
            current_file: str | None = None
            for key, page in page_summaries:
                file_path = key.split("#page_")[0]
                if file_path != current_file:
                    current_file = file_path
                    lines.append(f"# PDF Summary: {Path(file_path).name}\n")
                lines.append(f"## Page {page['page']}\n{page['summary']}\n")

            text = "\n".join(lines).strip() + "\n"

            backend = self._get_backend(runtime)
            res = await backend.awrite(output_path, text)

            if res.error:
                return res.error

            output_path_key = f"{Path(output_path).name}#export"
            export_record: PDFPageData = {
                "page": -1,
                "content": None,
                "image_path": None,
                "text_path": None,
                "text_chars": len(text),
                "extraction_method": "summary_export",
                "summary": text,
                "processed": True,
                "export_path": str(output_path),
            }

            return self._cmd(
                runtime,
                f"Exported {len(page_summaries)} LLM summaries to {output_path}",
                {"pdf_pages": {output_path_key: export_record}},
            )

        def sync_export(
            output_path: Annotated[str, "Absolute path where the exported text file should be written. Must be absolute, not relative."],
            runtime: ToolRuntime[None, PDFState],
        ) -> Command | str:
            """Synchronous wrapper for export_pdf tool."""
            return asyncio.run(async_export(output_path, runtime))

        return StructuredTool.from_function(
            name="export_pdf",
            description="Export processed LLM summaries to a file; raw extracted PDF content is never exported",
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