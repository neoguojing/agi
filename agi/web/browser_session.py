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

    async def get_last_result(self) -> PageInfo | None:
        """线程安全读取最近一次页面结果（返回副本，避免外部误改）。"""
        async with self._lock:
            self.last_active_at = time.time()
            return deepcopy(self.last_result) if self.last_result is not None else None

    async def get_history(self) -> list[dict[str, Any]]:
        """线程安全读取 backend 历史记录。"""
        async with self._lock:
            self.last_active_at = time.time()
            return self.backend.get_history()

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
