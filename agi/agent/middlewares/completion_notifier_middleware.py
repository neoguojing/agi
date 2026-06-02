from __future__ import annotations

import logging
from typing import Any

from langchain.agents.middleware.types import AgentMiddleware
from langgraph.runtime import Runtime
from langgraph_sdk import get_client

from agi.config import LANGGRAPH_MAIN_URL

logger = logging.getLogger(__name__)


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
        self.url = url or LANGGRAPH_MAIN_URL

        self._client = get_client(url=self.url)
        self._notified = False

    async def _ensure_parent_ids(self) -> None:
        """
        如果未指定 parent_thread_id，则自动查找主线程。
        """
        if self.parent_thread_id:
            return

        threads = await self._client.threads.search(
            metadata={"graph_id": "main"},
            limit=1,
        )

        if not threads:
            raise RuntimeError("No main thread found")

        self.parent_thread_id = threads[0]["thread_id"]

    async def _notify_parent(self, notification: str) -> None:
        """
        Send a notification run to the parent's thread.
        """
        try:
            await self._ensure_parent_ids()

            await self._client.runs.create(
                thread_id=self.parent_thread_id,
                assistant_id=self.parent_assistant_id or "main",
                input={
                    "messages": [
                        {
                            "role": "user",
                            "content": notification,
                        }
                    ]
                },
            )

            logger.info(
                "Notified parent thread %s that subagent '%s' finished",
                self.parent_thread_id,
                self.subagent_name,
            )

        except Exception as e:
            logger.error(
                "Failed to notify parent thread %s: %s",
                self.parent_thread_id,
                e,
                exc_info=True,
            )

    def _should_notify(self) -> bool:
        if self._notified:
            return False

        return bool(self.url) or (
            bool(self.parent_thread_id)
            and bool(self.parent_assistant_id)
        )

    async def _send_notification(self, message: str) -> None:
        if not self._should_notify():
            return

        self._notified = True
        await self._notify_parent(message)

    def _extract_last_message(self, state: dict[str, Any]) -> str:
        messages = state.get("messages", [])

        if not messages:
            return "(no output)"

        last = messages[-1]

        if hasattr(last, "content"):
            content = last.content
            return (
                content[:500]
                if isinstance(content, str)
                else str(content)[:500]
            )

        if isinstance(last, dict):
            return str(last.get("content", ""))[:500]

        return str(last)[:500]

    async def aafter_agent(
        self,
        state: dict[str, Any],
        runtime: Runtime,
    ) -> dict[str, Any] | None:
        summary = self._extract_last_message(state)

        await self._send_notification(
            f"[Async subagent '{self.subagent_name}' has completed] "
            f"Result: {summary}"
        )

        return None

    async def awrap_model_call(self, request, handler):
        try:
            return await handler(request)

        except Exception as e:
            await self._send_notification(
                f"[Async subagent '{self.subagent_name}' encountered an error] "
                f"Error: {e}"
            )
            raise


def build_completion_notifier(
    parent_thread_id: str | None,
    parent_assistant_id: str | None,
    subagent_name: str | None = None,
    url: str | None = None,
) -> CompletionNotifierMiddleware:
    return CompletionNotifierMiddleware(
        parent_thread_id=parent_thread_id,
        parent_assistant_id=parent_assistant_id,
        subagent_name=subagent_name,
        url=url or LANGGRAPH_MAIN_URL,
    )