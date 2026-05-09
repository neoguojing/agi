"""Middleware for providing stock and finance MCP tools to an agent.

Exposes all MCP tools directly and proactively gathers resource context.
Large content is written to backend files to avoid context bloat.
Tracks task progress so the LLM can maintain awareness across turns.
"""

import logging
from collections.abc import Awaitable, Callable
from typing import Any, NotRequired

from deepagents.backends.protocol import BackendProtocol
from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ContextT,
    ModelRequest,
    ModelResponse,
    ResponseT,
)
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.types import Command

from agi.utils.common import append_to_system_message

logger = logging.getLogger(__name__)

MANAGEMENT_TOOL_NAMES = {
    "available_categories",
    "available_tools",
    "activate_tools",
    "deactivate_tools",
    "activate_category",
}

AUTO_INVOKED_TOOL_NAMES = {"available_categories"}

INLINE_THRESHOLD = 500

STOCK_SYSTEM_PROMPT = """## Stock & Finance Tools

You have access to stock and finance tools powered by an MCP server.
All tools are available directly — no activation needed."""


class StockMiddlewareState(AgentState):
    active_tools: NotRequired[list[str]]
    stock_task_progress: NotRequired[dict[str, Any]]


class StockMiddleware(AgentMiddleware):
    state_schema = StockMiddlewareState

    def __init__(
        self,
        *,
        server_config: dict[str, Any] | None = None,
        system_prompt: str | None = None,
        backend: BackendProtocol | Callable | None = None,
    ) -> None:
        self._server_config = server_config or {
            "stock": {
                "transport": "http",
                "url": "http://localhost:8001/mcp",
            }
        }
        self._custom_system_prompt = system_prompt
        self._client: MultiServerMCPClient | None = None
        self._all_tools: list[BaseTool] = []
        self._tool_map: dict[str, BaseTool] = {}
        self._backend = backend
        self._context_file_path: str = ""
        self._progress_file_path: str = ""

    def _resolve_backend(self, runtime: Any) -> BackendProtocol | None:
        if callable(self._backend):
            return self._backend(runtime)
        if self._backend is not None:
            return self._backend  # type: ignore[return-value]
        return None

    def _has_backend(self) -> bool:
        return self._backend is not None

    async def _write_to_backend(
        self, runtime: Any, file_path: str, content: str
    ) -> bool:
        backend = self._resolve_backend(runtime)
        if backend is None:
            return False
        try:
            await backend.awrite(file_path, content)
            return True
        except Exception as e:
            logger.debug("Failed to write %s: %s", file_path, e)
            return False

    async def _ensure_initialized(self) -> None:
        if self._client is None:
            self._client = MultiServerMCPClient(self._server_config)
            self._all_tools = await self._client.get_tools()
            self._tool_map = {tool.name: tool for tool in self._all_tools}

    async def _gather_context(self) -> str:
        """Proactively invoke discovery tools to gather available tool info."""
        parts: list[str] = []
        for tool_name in AUTO_INVOKED_TOOL_NAMES:
            tool = self._tool_map.get(tool_name)
            if tool:
                try:
                    result = await tool.ainvoke({})
                    summary = str(result) if result else ""
                    if summary:
                        parts.append(f"--- {tool_name} ---\n{summary}")
                except Exception as e:
                    logger.debug("Failed to auto-invoke %s: %s", tool_name, e)
        return "\n\n".join(parts)

    def _get_progress_from_state(self, request: ModelRequest[ContextT]) -> dict[str, Any]:
        state = getattr(request, "state", None)
        if isinstance(state, dict):
            progress = state.get("stock_task_progress", {})
            if isinstance(progress, dict):
                return progress
        return {}

    def _format_progress_inline(self, progress: dict[str, Any]) -> str:
        """Short progress summary that fits in the prompt."""
        if not progress:
            return ""
        done = progress.get("completed_count", 0)
        total = progress.get("total_steps", 0)
        phase = progress.get("current_phase", "")
        last_tool = progress.get("last_tool", "")

        parts = [f"Task progress: {done}/{total} steps completed"]
        if phase:
            parts.append(f"Current phase: {phase}")
        if last_tool:
            parts.append(f"Last tool: {last_tool}")
        return " | ".join(parts)

    def _format_state_prompt(
        self,
        request: ModelRequest[ContextT],
        context_file_path: str,
        progress_file_path: str,
    ) -> str:
        parts = []
        progress = self._get_progress_from_state(request)
        inline_progress = self._format_progress_inline(progress)
        if inline_progress:
            parts.append(inline_progress)

        file_refs: list[str] = []
        if context_file_path:
            file_refs.append(f"  - Available tools info: `{context_file_path}`")
        if progress_file_path:
            file_refs.append(f"  - Detailed progress: `{progress_file_path}`")
        if file_refs:
            parts.append("Refer to these files for details:\n" + "\n".join(file_refs))

        return "\n\n".join(parts)

    def wrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], ModelResponse[ResponseT]],
    ) -> ModelResponse[ResponseT]:
        if self._all_tools:
            request = request.override(tools=self._all_tools)
        return handler(request)

    async def awrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], Awaitable[ModelResponse[ResponseT]]],
    ) -> ModelResponse[ResponseT]:
        await self._ensure_initialized()

        request = request.override(tools=self._all_tools)

        context_data = await self._gather_context()
        runtime = getattr(request, "runtime", None)

        context_file_path = self._context_file_path
        progress_file_path = self._progress_file_path

        if context_data:
            if self._has_backend() and len(context_data) > INLINE_THRESHOLD:
                if self._resolve_backend(runtime):
                    await self._write_to_backend(
                        runtime, "/stock_context.txt", context_data
                    )
                    context_file_path = "/stock_context.txt"
            # Small content kept inline — appended below

        system_prompt = self._custom_system_prompt or STOCK_SYSTEM_PROMPT
        state_prompt = self._format_state_prompt(
            request, context_file_path, progress_file_path,
        )

        # Append small context inline when it fits
        if context_data and len(context_data) <= INLINE_THRESHOLD:
            state_prompt += "\n\n### Available Tools\n" + context_data

        combined = "\n\n".join(p for p in [system_prompt, state_prompt] if p)
        if combined:
            request = request.override(
                system_message=append_to_system_message(request.system_message, combined)
            )

        return await handler(request)

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        return handler(request)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        result = await handler(request)
        tool_name = request.tool_call.get("name", "")

        if tool_name in MANAGEMENT_TOOL_NAMES:
            return result

        tool_args = request.tool_call.get("args", {})
        progress = self._build_progress_update(tool_name, tool_args, result)
        return self._wrap_result_with_progress(result, progress)

    def _build_progress_update(
        self, tool_name: str, tool_args: dict[str, Any], result: ToolMessage | Command
    ) -> dict[str, Any]:
        existing = self._extract_progress_from_result(result)
        completed = existing.get("completed_steps", [])
        remaining = existing.get("remaining_steps", [])
        current_phase = existing.get("current_phase", "")
        total = existing.get("total_steps", 0)

        completed.append(f"{tool_name} called")
        return {
            "completed_steps": completed,
            "completed_count": len(completed),
            "remaining_steps": remaining,
            "total_steps": max(total, len(completed)),
            "current_phase": current_phase or tool_name,
            "last_tool": tool_name,
            "last_result_summary": self._summary_result(result),
        }

    def _extract_progress_from_result(
        self, result: ToolMessage | Command
    ) -> dict[str, Any]:
        if isinstance(result, Command) and result.update:
            progress = result.update.get("stock_task_progress", {})
            if isinstance(progress, dict):
                return progress
        return {}

    @staticmethod
    def _summary_result(result: ToolMessage | Command) -> str:
        if isinstance(result, ToolMessage):
            return (result.content)[:200] if result.content else ""
        if isinstance(result, Command) and result.update:
            msgs = result.update.get("messages", [])
            if msgs:
                msg = msgs[0]
                if isinstance(msg, ToolMessage):
                    return (msg.content)[:200] if msg.content else ""
        return ""

    def _wrap_result_with_progress(
        self, result: ToolMessage | Command, *, stock_task_progress: dict[str, Any]
    ) -> Command:
        if isinstance(result, Command) and result.update:
            return Command(update={**result.update, "stock_task_progress": stock_task_progress})
        if isinstance(result, ToolMessage):
            return Command(update={
                "stock_task_progress": stock_task_progress,
                "messages": [result],
            })
        return Command(update={"stock_task_progress": stock_task_progress})
