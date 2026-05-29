from __future__ import annotations

import logging
from typing import Any

from langchain.agents.middleware.types import AgentMiddleware
from langgraph.runtime import Runtime
from langgraph_sdk import get_client

logger = logging.getLogger(__name__)


def _get_parent_ids(config: dict[str, Any]) -> tuple[str | None, str | None]:
    """Extract parent thread and assistant IDs from the runnable config."""
    configurable = config.get("configurable", {})
    return (
        configurable.get("parent_thread_id"),
        configurable.get("parent_assistant_id"),
    )


async def _notify_parent(
    notification: str,
    subagent_name: str,
    url: str | None = None,
    parent_thread_id: str | None = None,
    parent_assistant_id: str | None = None,
) -> None:
    """Send a notification run to the parent's thread."""
    if not parent_thread_id or not parent_assistant_id:
        logger.warning("Missing parent_thread_id or parent_assistant_id, cannot notify.")
        return

    try:
        # 正确传递 URL
        client = get_client(url=url)
        await client.runs.create(
            thread_id=parent_thread_id,
            assistant_id=parent_assistant_id,
            input={
                "messages": [{"role": "user", "content": notification}],
            },
        )
        logger.info(
            "Notified parent thread %s that subagent '%s' finished",
            parent_thread_id,
            subagent_name,
        )
    except Exception:
        logger.warning(
            "Failed to notify parent thread %s",
            parent_thread_id,
            exc_info=True,
        )


class CompletionNotifierMiddleware(AgentMiddleware):
    """Notifies the supervisor's thread when this subagent completes or errors."""

    def __init__(
        self,
        parent_thread_id: str | None,
        parent_assistant_id: str | None,
        subagent_name: str | None = None,
        url: str | None = None,
    ):
        self.parent_thread_id = parent_thread_id
        self.parent_assistant_id = parent_assistant_id
        self.subagent_name = subagent_name or "subagent"
        self.url = url
        self._notified = False

    def _should_notify(self) -> bool:
        # 修复优先级漏洞：确保未通知过，且（有URL 或 有完整的父级ID）
        if self._notified:
            return False
        
        has_parent_ids = bool(self.parent_thread_id) and bool(self.parent_assistant_id)
        return bool(self.url) or has_parent_ids

    async def _send_notification(self, message: str) -> None:
        if not self._should_notify():
            return
        self._notified = True
        
        # 修复核心：全部改用关键字参数传递，移除 type: ignore 隐患
        await _notify_parent(
            notification=message,
            subagent_name=self.subagent_name,
            url=self.url,
            parent_thread_id=self.parent_thread_id,
            parent_assistant_id=self.parent_assistant_id,
        )

    def _extract_last_message(self, state: dict[str, Any]) -> str:
        """Extract a summary from the subagent's final message."""
        messages = state.get("messages", [])
        if not messages:
            return "(no output)"
        last = messages[-1]
        if hasattr(last, "content"):
            content = last.content
            return content[:500] if isinstance(content, str) else str(content)[:500]
        if isinstance(last, dict):
            return str(last.get("content", ""))[:500]
        return str(last)[:500]

    async def aafter_agent(
        self, state: dict[str, Any], runtime: Runtime
    ) -> dict[str, Any] | None:
        """After-agent hook: fires when the subagent run completes successfully."""
        summary = self._extract_last_message(state)
        await self._send_notification(
            f"[Async subagent '{self.subagent_name}' has completed] Result: {summary}"
        )
        return None

    async def awrap_model_call(self, request, handler):
        """Wrap-model-call hook: catches errors and notifies the supervisor."""
        try:
            return await handler(request)
        except Exception as e:
            await self._send_notification(
                f"[Async subagent '{self.subagent_name}' encountered an error] "
                f"Error: {e!s}"
            )
            raise


def build_completion_notifier(
    parent_thread_id: str | None,
    parent_assistant_id: str | None,
    subagent_name: str | None = None,
    url: str | None = None,
) -> CompletionNotifierMiddleware:
    """Build a completion notifier middleware."""
    return CompletionNotifierMiddleware(
        parent_thread_id=parent_thread_id,
        parent_assistant_id=parent_assistant_id,
        subagent_name=subagent_name,
        url=url,
    )