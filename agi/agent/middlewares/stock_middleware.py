"""Middleware for providing stock and finance MCP tools to an agent.

Each conversation gets an independent MCP session to avoid cross-session
data leakage.

Features:
- Fully async-safe
- Thread-safe / multi-concurrency safe
- Persistent MCP session per conversation
- Proper MCP session lifecycle management
- Progress tracking
- Dynamic tool injection
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import Awaitable, Callable
from typing import Any,Annotated

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from typing_extensions import NotRequired

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
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_mcp_adapters.client import MultiServerMCPClient

from langgraph.types import Command

from agi.agent.prompt import get_middleware_prompt
from agi.utils.common import append_to_system_message

logger = logging.getLogger(__name__)

MANAGEMENT_TOOL_NAMES = {
    "available_categories",
    "available_tools",
    "activate_tools",
    "deactivate_tools",
    "activate_category",
}

def merge_progress(
    left: dict[str, Any] | None,
    right: dict[str, Any] | None,
) -> dict[str, Any]:

    left = left or {}
    right = right or {}

    completed_left = list(
        left.get("completed_steps", [])
    )

    completed_right = list(
        right.get("completed_steps", [])
    )

    merged_completed = completed_left.copy()

    for item in completed_right:
        if item not in merged_completed:
            merged_completed.append(item)

    return {
        **left,
        **right,
        "completed_steps": merged_completed,
        "completed_count": len(merged_completed),
    }


class StockMiddlewareState(AgentState):
    """State schema for stock middleware."""

    stock_task_progress:  Annotated[
        dict[str, Any],
        merge_progress,
    ]


class StockMiddleware(AgentMiddleware):
    """Injects stock/finance MCP tools dynamically.

    Design:
    - One persistent MCP session per conversation
    - Session survives multiple LLM/tool rounds
    - Safe for async concurrent workloads
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
        self._backend = backend

        # async-safe lock
        self._lock = asyncio.Lock()

        # session storage
        #
        # {
        #   session_key: {
        #       "ctx": async_context_manager,
        #       "session": actual_session,
        #   }
        # }
        self._sessions: dict[str, dict[str, Any]] = {}

        self.client = MultiServerMCPClient(
            {
                "stock": {
                    "url": "http://localhost:8001/mcp",
                    "transport": "http",
                }
            }
        )

        self.tools = asyncio.run(
            self.client.get_tools(server_name="stock")
        )
    

    # ------------------------------------------------------------------
    # session management
    # ------------------------------------------------------------------

    def _get_session_key(self, runtime: Any) -> str:
        """Get stable session key.

        Priority:
        conversation_id > thread_id > session_id > user_id
        """

        if hasattr(runtime, "context"):
            ctx = runtime.context

            for field in [
                "conversation_id",
                "thread_id",
                "session_id",
                "user_id",
            ]:
                value = getattr(ctx, field, None)
                if value:
                    return str(value)

        return str(uuid.uuid4())

    async def _ensure_session(self, session_key: str) -> None:
        """Ensure MCP session exists."""
        async with self._lock:
            if session_key in self._sessions:
                return

            server_config = self._server_config.get("stock", {})
            url = server_config.get("url", "http://localhost:8001/mcp")

            logger.info(
                "Creating MCP session: session_key=%s url=%s",
                session_key,
                url,
            )

            # async context manager
            ctx = streamable_http_client(url)

            # actual MCP session
            read_stream, write_stream, _ = await ctx.__aenter__()
            session = ClientSession(read_stream, write_stream)
            await session.initialize()

            self._sessions[session_key] = {
                "ctx": ctx,
                "session": session,
            }

    def _get_session(self, session_key: str) -> Any | None:
        data = self._sessions.get(session_key)
        if not data:
            return None
        return data["session"]

    async def _close_session(self, session_key: str) -> None:
        """Close MCP session."""

        async with self._lock:
            data = self._sessions.pop(session_key, None)

        if not data:
            return

        logger.info("Closing MCP session: %s", session_key)

        ctx = data["ctx"]

        try:
            await ctx.__aexit__(None, None, None)
        except Exception:
            logger.exception(
                "Failed to close MCP session: %s",
                session_key,
            )

    async def cleanup(self) -> None:
        """Cleanup all sessions."""

        async with self._lock:
            session_keys = list(self._sessions.keys())

        for session_key in session_keys:
            await self._close_session(session_key)

    # ------------------------------------------------------------------
    # tool discovery
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # prompt helpers
    # ------------------------------------------------------------------

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
        self,
        request: ModelRequest[ContextT],
    ) -> dict[str, Any]:
        state = getattr(request, "state", None)

        if isinstance(state, dict):
            progress = state.get("stock_task_progress", {})

            if isinstance(progress, dict):
                return progress

        return {}

    def _format_progress_inline(
        self,
        progress: dict[str, Any],
    ) -> str:
        if not progress:
            return ""

        done = progress.get("completed_count", 0)
        total = progress.get("total_steps", 0)
        phase = progress.get("current_phase", "")
        last_tool = progress.get("last_tool", "")

        parts = [f"Task progress: {done}/{total}"]

        if phase:
            parts.append(f"Current phase: {phase}")

        if last_tool:
            parts.append(f"Last tool: {last_tool}")

        return " | ".join(parts)

    def _build_state_prompt(
        self,
        request: ModelRequest[ContextT],
    ) -> str:
        progress = self._get_progress_from_state(request)
        return self._format_progress_inline(progress)

    # ------------------------------------------------------------------
    # model hooks
    # ------------------------------------------------------------------

    def wrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[
            [ModelRequest[ContextT]],
            Awaitable[ModelResponse[ResponseT]],
        ],
    ) -> ModelResponse[ResponseT]:
        """Sync execution is not supported."""

        raise RuntimeError(
            "StockMiddleware only supports async execution. "
            "Use awrap_model_call()."
        )

    async def awrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[
            [ModelRequest[ContextT]],
            Awaitable[ModelResponse[ResponseT]],
        ],
    ) -> ModelResponse[ResponseT]:
        """Inject MCP tools before model call."""
        session_key = self._get_session_key(request.runtime)

        await self._ensure_session(session_key)

        session = self._get_session(session_key)

        if session is None:
            return await handler(request)

        try:
            current_tools = await load_mcp_tools(session)
            print(f"***********************{current_tools}")
            if current_tools:
                request = request.override(tools=current_tools)
                system_prompt = (
                    self._custom_system_prompt
                    or get_middleware_prompt("stock")
                )

                state_prompt = self._build_state_prompt(
                    request,
                )

                activated_tools = [
                    t
                    for t in current_tools
                    if not self._is_management(t.name)
                ]

                tool_list = self._format_tools_list(
                    activated_tools,
                )

                if tool_list:
                    state_prompt += (
                        "\n\n### Activated Tools\n"
                        + tool_list
                    )

                combined_prompt = "\n\n".join(
                    p
                    for p in [
                        system_prompt,
                        state_prompt,
                    ]
                    if p
                )

                if combined_prompt:
                    request = request.override(
                        system_message=append_to_system_message(
                            request.system_message,
                            combined_prompt,
                        )
                    )

            return await handler(request)

        except Exception:
            logger.exception(
                "awrap_model_call failed: session=%s",
                session_key,
            )
            raise

    # ------------------------------------------------------------------
    # tool hooks
    # ------------------------------------------------------------------

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[
            [ToolCallRequest],
            ToolMessage | Command,
        ],
    ) -> ToolMessage | Command:
        """Sync execution is not supported."""

        raise RuntimeError(
            "StockMiddleware only supports async execution. "
            "Use awrap_tool_call()."
        )

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[
            [ToolCallRequest],
            Awaitable[ToolMessage | Command],
        ],
    ) -> ToolMessage | Command:
        """Handle tool call."""

        session_key = self._get_session_key(request.runtime)

        await self._ensure_session(session_key)

        try:
            result = await handler(request)

            tool_name = request.tool_call["name"]
            tool_args = request.tool_call.get("args", {})

            progress = self._build_progress_update(
                tool_name=tool_name,
                tool_args=tool_args,
                result=result,
            )

            return self._wrap_result_with_progress(
                result=result,
                stock_task_progress=progress,
            )

        except Exception:
            logger.exception(
                "awrap_tool_call failed: session=%s",
                session_key,
            )
            raise

    # ------------------------------------------------------------------
    # progress tracking
    # ------------------------------------------------------------------

    def _build_progress_update(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        result: ToolMessage | Command,
    ) -> dict[str, Any]:
        existing = self._extract_progress_from_result(
            result,
        )

        completed = existing.get(
            "completed_steps",
            [],
        )

        remaining = existing.get(
            "remaining_steps",
            [],
        )

        current_phase = existing.get(
            "current_phase",
            "",
        )

        total = existing.get(
            "total_steps",
            0,
        )

        completed.append(f"{tool_name} called")

        return {
            "completed_steps": completed,
            "completed_count": len(completed),
            "remaining_steps": remaining,
            "total_steps": max(
                total,
                len(completed),
            ),
            "current_phase": (
                current_phase or tool_name
            ),
            "last_tool": tool_name,
            "last_tool_args": tool_args,
            "last_result_summary": self._summary_result(
                result
            ),
        }

    def _extract_progress_from_result(
        self,
        result: ToolMessage | Command,
    ) -> dict[str, Any]:
        if isinstance(result, Command):
            if result.update:
                progress = result.update.get(
                    "stock_task_progress",
                    {},
                )

                if isinstance(progress, dict):
                    return progress

        return {}

    @staticmethod
    def _summary_result(
        result: ToolMessage | Command,
    ) -> str:
        if isinstance(result, ToolMessage):
            content = result.content

            if isinstance(content, str):
                return content[:200]

            return str(content)[:200]

        if isinstance(result, Command):
            if result.update:
                messages = result.update.get(
                    "messages",
                    [],
                )

                if messages:
                    msg = messages[0]

                    if isinstance(msg, ToolMessage):
                        content = msg.content

                        if isinstance(content, str):
                            return content[:200]

                        return str(content)[:200]

        return ""

    def _wrap_result_with_progress(
        self,
        result: ToolMessage | Command,
        stock_task_progress: dict[str, Any],
    ) -> Command:
        """Merge progress into graph state."""

        if isinstance(result, Command):
            update = (
                dict(result.update)
                if result.update
                else {}
            )

            update["stock_task_progress"] = (
                stock_task_progress
            )

            return Command(update=update)

        if isinstance(result, ToolMessage):
            return Command(
                update={
                    "stock_task_progress": (
                        stock_task_progress
                    ),
                    "messages": [result],
                }
            )

        return Command(
            update={
                "stock_task_progress": (
                    stock_task_progress
                )
            }
        )