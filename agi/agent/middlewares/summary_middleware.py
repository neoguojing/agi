from __future__ import annotations

import uuid
from typing import Any, Awaitable, Callable

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool

# 假设这些类型来自你的系统
# from langgraph.types import ModelRequest, ModelResponse, ExtendedModelResponse, Command
# from langgraph.errors import ContextOverflowError
# from langgraph.runtime import Runtime
# from langgraph.store import BackendProtocol


# =========================================================
# Context-only Middleware (REFACTORED)
# =========================================================

class _DeepAgentsContextMiddleware:
    """
    Lightweight middleware.

    Responsibility:
    - reconstruct conversation context
    - optional tool argument truncation
    - NO summarization
    - NO persistence
    """

    def __init__(
        self,
        *,
        truncate_args_settings: dict[str, Any] | None = None,
    ):
        # ---- tool arg truncation config ----
        if truncate_args_settings is None:
            self._truncate_args_trigger = None
            self._truncate_args_keep = ("messages", 20)
            self._max_arg_length = 2000
            self._truncation_text = "...(argument truncated)"
        else:
            self._truncate_args_trigger = truncate_args_settings.get("trigger")
            self._truncate_args_keep = truncate_args_settings.get("keep", ("messages", 20))
            self._max_arg_length = truncate_args_settings.get("max_length", 2000)
            self._truncation_text = truncate_args_settings.get("truncation_text", "...(argument truncated)")

    # =========================================================
    # 1. Context reconstruction
    # =========================================================

    def _get_effective_messages(self, request) -> list[Any]:
        """
        Rebuild message list from state event if exists.
        (Only supports summarization-event style compatibility)
        """
        event = request.state.get("_summarization_event")
        return self._apply_event_to_messages(request.messages, event)

    @staticmethod
    def _apply_event_to_messages(messages: list[Any], event: dict | None) -> list[Any]:
        """
        If summary exists:
            [summary_message] + messages[from cutoff:]
        """
        if event is None:
            return list(messages)

        try:
            summary_msg = event["summary_message"]
            cutoff_idx = event["cutoff_index"]
        except Exception:
            return list(messages)

        # defensive
        if cutoff_idx < 0 or cutoff_idx > len(messages):
            return list(messages)

        return [summary_msg] + messages[cutoff_idx:]

    # =========================================================
    # 2. Optional tool argument truncation
    # =========================================================

    def _truncate_tool_call(self, tool_call: dict[str, Any]) -> dict[str, Any]:
        args = tool_call.get("args", {})

        new_args = {}
        modified = False

        for k, v in args.items():
            if isinstance(v, str) and len(v) > self._max_arg_length:
                new_args[k] = v[:20] + self._truncation_text
                modified = True
            else:
                new_args[k] = v

        return {**tool_call, "args": new_args} if modified else tool_call

    def _truncate_args(
        self,
        messages: list[Any],
        system_message: SystemMessage | None = None,
        tools: list[BaseTool | dict[str, Any]] | None = None,
    ) -> tuple[list[Any], bool]:
        """
        Lightweight optional preprocessing.
        (kept for compatibility, but does minimal work)
        """
        # 当前版本：不做复杂 truncation
        return messages, False

    # =========================================================
    # 3. Core entry point (SYNC)
    # =========================================================

    def wrap_model_call(
        self,
        request,
        handler: Callable,
    ):
        """
        Context builder only.

        No summarization.
        No persistence.
        No backend logic.
        """

        # Step 1: restore context view
        effective_messages = self._get_effective_messages(request)

        # Step 2: optional tool arg truncation (noop here)
        final_messages, _ = self._truncate_args(
            effective_messages,
            request.system_message,
            request.tools,
        )

        # Step 3: call LLM handler
        return handler(
            request.override(messages=final_messages)
        )

    # =========================================================
    # 4. Core entry point (ASYNC)
    # =========================================================

    async def awrap_model_call(
        self,
        request,
        handler: Callable[..., Awaitable],
    ):
        """
        Async version of context-only middleware.
        """

        effective_messages = self._get_effective_messages(request)

        final_messages, _ = self._truncate_args(
            effective_messages,
            request.system_message,
            request.tools,
        )

        return await handler(
            request.override(messages=final_messages)
        )