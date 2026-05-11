"""Middleware for providing stock and finance MCP tools to an agent.

Dynamically refreshes the tool set from the MCP session before each model call.
Large content is written to backend files to avoid context bloat.
Tracks task progress so the LLM can maintain awareness across turns.
"""

import logging
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
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
from agi.agent.prompt import get_middleware_prompt

logger = logging.getLogger(__name__)

MANAGEMENT_TOOL_NAMES = {
    "available_categories",
    "available_tools",
    "activate_tools",
    "deactivate_tools",
    "activate_category",
}

INLINE_THRESHOLD = 500


class StockMiddlewareState(AgentState):
    stock_task_progress: NotRequired[dict[str, Any]]


class StockMiddleware(AgentMiddleware):
    """Injects stock/finance MCP tools dynamically per model call.

    Keeps a persistent MCP session so that after the LLM activates tools,
    the next model call picks them up via ``session.list_tools()``.
    """
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
        self._backend = backend
        # Populated lazily on first model call
        self._session: Any = None
        self._current_tools: list[BaseTool] = []

    # -- lifecycle -------------------------------------------------------

    async def _ensure_session(self) -> None:
        """Open the MCP client + persistent session on first use."""
        if self._session is not None:
            return
        self._client = MultiServerMCPClient(self._server_config)
        # `session()` is an async context manager; we manually enter it
        # to keep it alive for the middleware's lifetime.
        server_name = next(iter(self._server_config))
        session_cm = self._client.session(server_name)
        self._session = await session_cm.__aenter__()
        await self._refresh_tools()

    async def _refresh_tools(self) -> None:
        """Pull the current tool set from the MCP session."""
        result = await self._session.list_tools()
        self._current_tools = result.tools

    async def _close_session(self) -> None:
        if self._session is not None:
            try:
                await self._session.__aexit__(None, None, None)
            except Exception:
                pass
            self._session = None

    # -- backend helpers -------------------------------------------------

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

    # -- prompt building -------------------------------------------------

    @staticmethod
    def _format_tools_list(tools: list[BaseTool]) -> str:
        if not tools:
            return ""
        lines = []
        for tool in tools:
            desc = tool.description or ""
            if desc:
                lines.append(f"  - **{tool.name}**: {desc}")
            else:
                lines.append(f"  - **{tool.name}**")
        return "\n".join(lines)

    @staticmethod
    def _is_management(name: str) -> bool:
        return name in MANAGEMENT_TOOL_NAMES

    def _get_progress_from_state(
        self, request: ModelRequest[ContextT]
    ) -> dict[str, Any]:
        state = getattr(request, "state", None)
        if isinstance(state, dict):
            progress = state.get("stock_task_progress", {})
            if isinstance(progress, dict):
                return progress
        return {}

    def _format_progress_inline(self, progress: dict[str, Any]) -> str:
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

    def _build_state_prompt(self, request: ModelRequest[ContextT]) -> str:
        progress = self._get_progress_from_state(request)
        return self._format_progress_inline(progress)

    # -- model call hooks ------------------------------------------------

    def wrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], ModelResponse[ResponseT]],
    ) -> ModelResponse[ResponseT]:
        if self._current_tools:
            request = request.override(tools=self._current_tools)
        return handler(request)

    async def awrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], Awaitable[ModelResponse[ResponseT]]],
    ) -> ModelResponse[ResponseT]:
        # 1. Open session + refresh tool set (picks up activations)
        await self._ensure_session()
        await self._refresh_tools()

        current = self._current_tools
        if not current:
            return await handler(request)

        request = request.override(tools=current)

        # 2. Build prompt with activated tool catalog
        system_prompt = self._custom_system_prompt or get_middleware_prompt("stock")
        state_prompt = self._build_state_prompt(request)

        activated = [t for t in current if not self._is_management(t.name)]
        tool_list = self._format_tools_list(activated)
        if tool_list:
            state_prompt += "\n\n### Activated Tools\n" + tool_list

        combined = "\n\n".join(p for p in [system_prompt, state_prompt] if p)
        if combined:
            request = request.override(
                system_message=append_to_system_message(
                    request.system_message, combined
                )
            )

        return await handler(request)

    # -- tool call hooks -------------------------------------------------

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

    # -- progress tracking -----------------------------------------------

    def _build_progress_update(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        result: ToolMessage | Command,
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
        self,
        result: ToolMessage | Command,
        stock_task_progress: dict[str, Any],
    ) -> Command:
        if isinstance(result, Command):
            update = dict(result.update) if result.update else {}
            update["stock_task_progress"] = stock_task_progress
            return Command(update=update)
        if isinstance(result, ToolMessage):
            return Command(update={
                "stock_task_progress": stock_task_progress,
                "messages": [result],
            })
        return Command(update={"stock_task_progress": stock_task_progress})
