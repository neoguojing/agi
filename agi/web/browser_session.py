from asyncio import Lock
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from collections.abc import AsyncIterator
from .browser_backend import StatefulBrowserBackend
from .browser_types import UserBrowserSession
import logging

logger = logging.getLogger(__name__)


class BrowserBackendPool:
    """Manage one browser backend per user with idle-time eviction."""

    # middleware 只需要关心“给我某个 user_id 的会话”；具体创建/复用/空闲回收由这里统一管理。

    def __init__(
        self,
        storage_dir: str,
        *,
        idle_timeout_seconds: float = 60.0*30.0,
        headless: bool = False,
        timeout: int = 30_000,
        max_content_length: int = 2_000_000,
        max_retry: int = 2,
    ) -> None:
        self.storage_dir = Path(storage_dir).resolve()
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        # 每个用户都有独立目录，用于保存截图、状态快照、storage_state。
        self.idle_timeout_seconds = idle_timeout_seconds
        self.headless = headless
        self.timeout = timeout
        self.max_content_length = max_content_length
        self.max_retry = max_retry
        self._lock = Lock()
        self._sessions: dict[str, UserBrowserSession] = {}

    @asynccontextmanager
    async def session(self, user_id: str) -> AsyncIterator[UserBrowserSession]:
        """Yield the active session for a user and refresh its idle timer."""
        session = await self._acquire_session(user_id)
        async with session.operation_lock:
            try:
                yield session
            finally:
                await self._release_session(user_id)

    def get_existing_session(self, user_id: str) -> UserBrowserSession | None:
        """Return the current session for a user without creating a new browser."""
        return self._sessions.get(user_id)

    async def close_all(self) -> None:
        """Close every managed browser session."""
        async with self._lock:
            sessions = list(self._sessions.items())
            self._sessions.clear()

        for _user_id, session in sessions:
            if session.idle_task is not None:
                session.idle_task.cancel()
                with suppress(BaseException):
                    await session.idle_task
            await session.backend.close()

    async def _acquire_session(self, user_id: str) -> UserBrowserSession:
        async with self._lock:
            session = self._sessions.get(user_id)
            if session is None:
                session = UserBrowserSession(
                    user_id=user_id,
                    backend=StatefulBrowserBackend(
                        storage_dir=str(self.storage_dir / self._sanitize_user_id(user_id)),
                        headless=self.headless,
                        timeout=self.timeout,
                        max_content_length=self.max_content_length,
                        max_retry=self.max_retry,
                    ),
                )
                self._sessions[user_id] = session
                logger.info("Created browser session for user_id=%s", user_id)

            if session.idle_task is not None:
                session.idle_task.cancel()
                session.idle_task = None

            loop = session.backend._get_running_loop()
            session.active_operations += 1
            session.last_used_at = loop.time()
            return session

    async def _release_session(self, user_id: str) -> None:
        async with self._lock:
            session = self._sessions.get(user_id)
            if session is None:
                return

            session.active_operations = max(0, session.active_operations - 1)
            loop = session.backend._get_running_loop()
            session.last_used_at = loop.time()
            if session.active_operations == 0:
                session.idle_task = loop.create_task(self._close_when_idle(user_id))

    async def _close_when_idle(self, user_id: str) -> None:
        try:
            await self._sleep(self.idle_timeout_seconds)
            async with self._lock:
                session = self._sessions.get(user_id)
                if session is None or session.active_operations > 0:
                    return

                loop = session.backend._get_running_loop()
                if loop.time() - session.last_used_at < self.idle_timeout_seconds:
                    session.idle_task = loop.create_task(self._close_when_idle(user_id))
                    return

                self._sessions.pop(user_id, None)
            logger.info("Closing idle browser session for user_id=%s", user_id)
            await session.backend.close()
        except Exception:
            logger.exception("Failed to close idle browser session for user_id=%s", user_id)

    async def _sleep(self, seconds: float) -> None:
        import asyncio

        await asyncio.sleep(seconds)

    def _sanitize_user_id(self, user_id: str) -> str:
        return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in user_id) or "default"
