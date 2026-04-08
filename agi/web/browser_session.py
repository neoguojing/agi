import time
import asyncio
from copy import deepcopy
from typing import Any, Awaitable, Callable
from .browser_protocal import AbstractBrowserBackend
from .browser_types import PageInfo


class BrowserSession:
    """
    单用户会话（强隔离）
    """

    def __init__(self, user_id: str, backend: AbstractBrowserBackend):
        self.user_id = user_id
        self.backend = backend

        self.created_at = time.time()
        self.last_active_at = time.time()
        self.last_result: PageInfo | None = None
        self.previous_result: PageInfo | None = None

        # 防止同一用户并发操作冲突
        self._lock = asyncio.Lock()

    async def run(
        self,
        op: Callable[..., Awaitable[Any]],
        *args,
        **kwargs
    ) -> Any:
        """
        串行执行操作（核心入口）
        """
        async with self._lock:
            self.last_active_at = time.time()
            result = await op(*args, **kwargs)
            if isinstance(result, PageInfo):
                self.previous_result = self.last_result
                self.last_result = result
            return result

    async def close(self):
        """
        关闭session
        """
        await self.backend.close()

    async def get_runtime_context(self, *, include_environment: bool = False) -> dict[str, Any]:
        """线程安全读取会话核心上下文，减少重复 get_state/get_last_result 调用。"""
        async with self._lock:
            self.last_active_at = time.time()
            state_snapshot = self.backend.get_state_snapshot(
                user_id=self.user_id,
                last_result=self.last_result,
                previous_result=self.previous_result,
            )
            context: dict[str, Any] = {
                "state": state_snapshot,
                "last_result": deepcopy(self.last_result) if self.last_result is not None else None,
            }
            if include_environment:
                context["environment"] = await self.backend.get_environment_status()
            return context

    async def apply_ocr_result(
        self,
        *,
        text: str,
        screenshot_path: str,
        metadata_update: dict[str, Any],
    ) -> None:
        """把 OCR 结果回写到 last_result，供后续 extract/snapshot 复用。"""
        async with self._lock:
            self.last_active_at = time.time()
            if self.last_result is None:
                return
            self.last_result.text = text
            self.last_result.screenshot_path = screenshot_path
            self.last_result.metadata = {
                **self.last_result.metadata,
                **metadata_update,
            }
