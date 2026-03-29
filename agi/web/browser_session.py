import time
import asyncio
from typing import Any, Awaitable, Callable
from .browser_protocal import AbstractBrowserBackend


class BrowserSession:
    """
    单用户会话（强隔离）
    """

    def __init__(self, user_id: str, backend: AbstractBrowserBackend):
        self.user_id = user_id
        self.backend = backend

        self.created_at = time.time()
        self.last_active_at = time.time()

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
            return await op(*args, **kwargs)

    async def close(self):
        """
        关闭session
        """
        await self.backend.close()