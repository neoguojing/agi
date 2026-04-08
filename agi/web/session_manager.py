import asyncio
import time
from typing import Dict, Callable, Optional, Any

from .browser_protocal import AbstractBrowserBackend
from .browser_session import BrowserSession


class BrowserSessionManager:
    """
    user_id -> BrowserSession 管理器
    """

    def __init__(
        self,
        backend_factory: Callable[[], AbstractBrowserBackend],
        session_ttl: int = 1800,  # 30分钟
        cleanup_interval: int = 60
    ):
        self._sessions: Dict[str, BrowserSession] = {}
        self._backend_factory = backend_factory

        self._ttl = session_ttl
        self._cleanup_interval = cleanup_interval

        self._lock = asyncio.Lock()
        self._cleanup_task: Optional[asyncio.Task] = None

    # ========================
    # 生命周期
    # ========================

    async def start(self):
        if self._cleanup_task is None:
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())

    async def shutdown(self):
        if self._cleanup_task:
            self._cleanup_task.cancel()

        async with self._lock:
            sessions = list(self._sessions.values())
            self._sessions.clear()

        for session in sessions:
            await session.close()

    # ========================
    # Session获取
    # ========================

    async def get_session(self, user_id: str) -> BrowserSession:
        async with self._lock:
            session = self._sessions.get(user_id)
            if session:
                return session

            # 创建新backend
            backend = self._backend_factory()
            await backend.initialize()

            session = BrowserSession(user_id, backend)
            self._sessions[user_id] = session
            return session

    async def close_session(self, user_id: str):
        async with self._lock:
            session = self._sessions.pop(user_id, None)

        if session:
            await session.close()

    # ========================
    # 对外统一API（推荐用这个）
    # ========================

    async def navigate(self, user_id: str, url: str, **kwargs):
        session = await self.get_session(user_id)
        return await session.run(session.backend.navigate, url, **kwargs)

    async def click(self, user_id: str, selector: str):
        session = await self.get_session(user_id)
        return await session.run(session.backend.click, selector)

    async def fill(self, user_id: str, selector: str, value: str):
        session = await self.get_session(user_id)
        return await session.run(session.backend.fill, selector, value)

    async def scroll(self, user_id: str, direction: str = "down", distance: int = 800):
        session = await self.get_session(user_id)
        return await session.run(session.backend.scroll, direction, distance)

    async def scroll(self, user_id: str, direction: str = "down", distance: int = 800):
        session = await self.get_session(user_id)
        return await session.run(session.backend.scroll, direction, distance)

    async def find_elements(self, user_id: str, selector: str):
        session = await self.get_session(user_id)
        return await session.run(session.backend.find_elements, selector)

    async def extract_ui(self, user_id: str, limit: int = 12):
        session = await self.get_session(user_id)
        return await session.run(session.backend.extract_ui, limit)

    async def inspect_element_property(self, user_id: str, selector: str, property_name: str):
        session = await self.get_session(user_id)
        return await session.run(session.backend.inspect_element_property, selector, property_name)

    async def screenshot(self, user_id: str):
        session = await self.get_session(user_id)
        return await session.run(session.backend.get_screenshot)

    async def get_runtime_context(self, user_id: str) -> dict[str, Any]:
        """统一读取状态与最近结果，避免中间件多次重复调用。"""
        session = await self.get_session(user_id)
        return await session.get_runtime_context()

    async def apply_ocr_result(
        self,
        user_id: str,
        *,
        text: str,
        screenshot_path: str,
        metadata_update: dict[str, Any],
    ) -> None:
        """回写 OCR 结果到当前会话的 last_result。"""
        session = await self.get_session(user_id)
        await session.apply_ocr_result(
            text=text,
            screenshot_path=screenshot_path,
            metadata_update=metadata_update,
        )

    # ========================
    # 自动清理
    # ========================

    async def _cleanup_loop(self):
        while True:
            await asyncio.sleep(self._cleanup_interval)

            now = time.time()
            to_delete = []

            async with self._lock:
                for user_id, session in self._sessions.items():
                    if now - session.last_active_at > self._ttl:
                        to_delete.append(user_id)

                for user_id in to_delete:
                    session = self._sessions.pop(user_id)

                    # 异步关闭（不要卡住锁）
                    asyncio.create_task(session.close())
