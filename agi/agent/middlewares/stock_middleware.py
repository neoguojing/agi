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
from contextlib import suppress

from anyio import (
    BrokenResourceError,
    ClosedResourceError,
    EndOfStream,
)


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
from langchain_core.tools import BaseTool, StructuredTool
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

        self.client = MultiServerMCPClient(
            {
                "stock": {
                    "url": "http://localhost:8001/mcp",
                    "transport": "http",
                }
            }
        )

        self._session = None
        self._session_ctx = None

        self.tools = [
            self._create_available_categories_tool(),
            self._create_available_tools_tool(),
            self._create_activate_tools_tool(),
            self._create_deactivate_tools_tool(),
            self._create_activate_category_tool(),
        ]

        # asyncio.run(self.get_session())



    # ------------------------------------------------------------------
    # management tools
    # ------------------------------------------------------------------

    def _create_available_categories_tool(self) -> BaseTool:
        """Create the available_categories tool."""
        description = "List available tool categories and subcategories with tool counts."

        async def async_available_categories(
            runtime: Any,
        ) -> str:
            session = await self.get_session()
            result = await session.call_tool("available_categories", {})
            return str(result.content)

        def sync_available_categories(runtime: Any) -> str:
            raise RuntimeError("StockMiddleware only supports async execution.")

        return StructuredTool.from_function(
            name="available_categories",
            description=description,
            func=sync_available_categories,
            coroutine=async_available_categories,
        )

    def _create_available_tools_tool(self) -> BaseTool:
        """Create the available_tools tool."""
        description = "List tools in a specific category and subcategory."

        async def async_available_tools(
            runtime: Any,
            category: Annotated[str, "The category of tools to list"],
            subcategory: Annotated[str | None, "Optional subcategory to filter by. Use 'general' for tools directly under the category."] = None,
        ) -> str:
            session = await self.get_session()
            result = await session.call_tool("available_tools", {"category": category, "subcategory": subcategory})
            return str(result.content)

        def sync_available_tools(
            runtime: Any,
            category: str,
            subcategory: str | None = None,
        ) -> str:
            raise RuntimeError("StockMiddleware only supports async execution.")

        return StructuredTool.from_function(
            name="available_tools",
            description=description,
            func=sync_available_tools,
            coroutine=async_available_tools,
        )

    def _create_activate_tools_tool(self) -> BaseTool:
        """Create the activate_tools tool."""
        description = "Activate one or more tools for this session."

        async def async_activate_tools(
            runtime: Any,
            tool_names: Annotated[list[str], "Names of tools to activate"],
        ) -> str:
            session = await self.get_session()
            result = await session.call_tool("activate_tools", {"tool_names": tool_names})
            return str(result.content)

        def sync_activate_tools(runtime: Any, tool_names: list[str]) -> str:
            raise RuntimeError("StockMiddleware only supports async execution.")

        return StructuredTool.from_function(
            name="activate_tools",
            description=description,
            func=sync_activate_tools,
            coroutine=async_activate_tools,
        )

    def _create_deactivate_tools_tool(self) -> BaseTool:
        """Create the deactivate_tools tool."""
        description = "Deactivate one or more tools for this session."

        async def async_deactivate_tools(
            runtime: Any,
            tool_names: Annotated[list[str], "Names of tools to deactivate"],
        ) -> str:
            session = await self.get_session()
            result = await session.call_tool("deactivate_tools", {"tool_names": tool_names})
            return str(result.content)

        def sync_deactivate_tools(runtime: Any, tool_names: list[str]) -> str:
            raise RuntimeError("StockMiddleware only supports async execution.")

        return StructuredTool.from_function(
            name="deactivate_tools",
            description=description,
            func=sync_deactivate_tools,
            coroutine=async_deactivate_tools,
        )

    def _create_activate_category_tool(self) -> BaseTool:
        """Create the activate_category tool."""
        description = "Activate all tools in a category (or subcategory) for this session."

        async def async_activate_category(
            runtime: Any,
            category: Annotated[str, "Category name to activate all tools for"],
            subcategory: Annotated[str | None, "Optional subcategory to narrow activation"] = None,
        ) -> str:
            session = await self.get_session()
            result = await session.call_tool("activate_category", {"category": category, "subcategory": subcategory})
            return str(result.content)

        def sync_activate_category(runtime: Any, category: str, subcategory: str | None = None) -> str:
            raise RuntimeError("StockMiddleware only supports async execution.")

        return StructuredTool.from_function(
            name="activate_category",
            description=description,
            func=sync_activate_category,
            coroutine=async_activate_category,
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
    
    async def get_session(self):

        async with self._lock:

            # session 已存在 -> 先检查是否还活着
            if self._session is not None:

                try:
                    await self._session.send_ping()

                    self.tools = await load_mcp_tools(
                        self._session
                    )

                    return self._session

                except (
                    BrokenPipeError,
                    ConnectionResetError,
                    BrokenResourceError,
                    ClosedResourceError,
                    EndOfStream,
                    TimeoutError,
                    EOFError,
                ):

                    logger.warning(
                        "MCP session disconnected, recreating..."
                    )

                    # 清理旧 session
                    if self._session_ctx is not None:

                        with suppress(Exception):
                            await self._session_ctx.__aexit__(
                                None,
                                None,
                                None,
                            )

                    self._session = None
                    self._session_ctx = None

            # 创建新 session
            self._session_ctx = self.client.session(
                server_name="stock"
            )

            self._session = await self._session_ctx.__aenter__()

            self.tools = await load_mcp_tools(
                self._session
            )

            return self._session
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

        try:
            system_prompt = (
                self._custom_system_prompt
                or get_middleware_prompt("stock")
            )

            state_prompt = self._build_state_prompt(
                request,
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

            await self.get_session()
            request = request.override(tools=self.tools)
        except Exception as e:
            logger.exception(
                "StockMiddleware.awrap_model_call setup failed: %s", e
            )

        return await handler(request)
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

        logger.info(
            "*******************\n" +
            "\n".join(
                tool.name for tool in self.tools
            )
        )
        tool_name = request.tool_call["name"]
        tool_args = request.tool_call["args"]
        logger.info(f"tool call :{tool_name},{tool_args}")

        try:
            if request.tool is None:
                target_tool = next(
                    (
                        tool
                        for tool in self.tools
                        if tool.name == tool_name
                    ),
                    None
                )
                request = request.override(tool=target_tool)

            
            result = await handler(request)
            

            logger.info(f"tool result :{result}")

            tool_args_current = request.tool_call.get("args", {})

            progress = self._build_progress_update(
                tool_name=tool_name,
                tool_args=tool_args_current,
                result=result,
            )

            return self._wrap_result_with_progress(
                result=result,
                stock_task_progress=progress,
            )

        except Exception as e:
            logger.exception("StockMiddleware.awrap_tool_call wrapping failed: %s", e)
            tool_call_id = request.tool_call.get("id", "unknown")
            return ToolMessage(
                content=f"Internal middleware error during tool call {tool_name}: {str(e)}",
                tool_call_id=tool_call_id,
            )

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