# pdf_middleware.py

from __future__ import annotations

import os
import asyncio
from pathlib import Path
from typing import Annotated, NotRequired, TypedDict, Any

from langchain.agents.middleware.types import AgentMiddleware, AgentState
from langchain.tools import ToolRuntime
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.tools import StructuredTool, BaseTool
from langchain_core.messages import ToolMessage
from langgraph.types import Command

from deepagents.backends.protocol import BackendProtocol
from deepagents.backends import StateBackend
from deepagents.backends.utils import validate_path

from agi.agent.prompt import get_middleware_prompt


# =========================
# State 定义
# =========================

class PDFPageData(TypedDict):
    page: int
    content: str | None
    image_path: str | None
    processed: bool
    summary: NotRequired[str]


def _pdf_page_reducer(
    left: dict[str, PDFPageData] | None,
    right: dict[str, PDFPageData | None],
) -> dict[str, PDFPageData]:
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
    pdf_pages: Annotated[NotRequired[dict[str, PDFPageData]], _pdf_page_reducer]


# =========================
# Middleware
# =========================

class PDFMiddleware(AgentMiddleware[PDFState, Any, Any]):

    state_schema = PDFState

    def __init__(self, *, backend: BackendProtocol | None = None):
        self.backend = backend if backend is not None else StateBackend

        self.tools = [
            self._create_parse_pdf_tool(),
            self._create_read_pdf_page_tool(),
            self._create_render_pdf_page_tool(),
            self._create_prepare_pdf_page_tool(),
            self._create_set_page_summary_tool(),
            self._create_export_pdf_tool(),
        ]

    def _get_backend(self, runtime: ToolRuntime) -> BackendProtocol:
        if callable(self.backend):
            return self.backend(runtime)
        return self.backend

    # =========================
    # 1️⃣ parse_pdf
    # =========================

     # =========================
    # helper
    # =========================

    def _cmd(self, runtime, msg, update):
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

        async def async_parse(path: str, runtime: ToolRuntime):

            backend = self._get_backend(runtime)
            validated = validate_path(path)

            if validated.endswith(".pdf"):
                files = [validated]
            else:
                infos = await backend.als_info(validated)
                files = [i["path"] for i in infos if i["path"].endswith(".pdf")]

            import pdfplumber

            updates = {}
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

        def sync_parse(path: str, runtime: ToolRuntime):
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

    def _create_read_pdf_page_tool(self):

        async def async_read(file_path: str, page: int, runtime: ToolRuntime):
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

        def sync_read(file_path, page, runtime):
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

    def _create_render_pdf_page_tool(self):

        async def async_render(file_path: str, page: int, runtime):
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
                f"Rendered page {page} → {path}",
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

        def sync_render(file_path, page, runtime):
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

    def _create_prepare_pdf_page_tool(self):

        async def async_prepare(key: str, runtime):
            page = runtime.state.get("pdf_pages", {}).get(key)

            if not page:
                return ToolMessage(content="Error: page not found", tool_call_id=runtime.tool_call_id)

            if page["content"]:
                text = page["content"][:8000]
                return ToolMessage(
                    content=f"Summarize:\n\n{text}",
                    tool_call_id=runtime.tool_call_id,
                )

            if page["image_path"]:
                return ToolMessage(
                    content=f"Analyze image: {page['image_path']}",
                    tool_call_id=runtime.tool_call_id,
                )

            return ToolMessage(content="Error: no data", tool_call_id=runtime.tool_call_id)

        def sync_prepare(key, runtime):
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

    def _create_set_page_summary_tool(self):

        async def async_set(key: str, summary: str, runtime):
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

        def sync_set(key, summary, runtime):
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

    def _create_export_pdf_tool(self):

        async def async_export(output_path: str, runtime):
            pages = runtime.state.get("pdf_pages", {})

            lines = []
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

        def sync_export(output_path, runtime):
            return asyncio.run(async_export(output_path, runtime))

        return StructuredTool.from_function(
            name="export_pdf",
            description="Export summaries",
            func=sync_export,
            coroutine=async_export,
        )


    def wrap_model_call(
        self,
        request,
        handler,
    ):
        """
        Inject PDF system prompt and ensure tool visibility
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
            from deepagents.middleware._utils import append_to_system_message

            new_system_message = append_to_system_message(
                request.system_message,
                system_prompt,
            )
            request = request.override(system_message=new_system_message)

        return handler(request)

    async def awrap_model_call(
        self,
        request,
        handler,
    ):
        """
        Async version of wrap_model_call
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
            from deepagents.middleware._utils import append_to_system_message

            new_system_message = append_to_system_message(
                request.system_message,
                system_prompt,
            )
            request = request.override(system_message=new_system_message)

        return await handler(request)