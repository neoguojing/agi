from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

pytest.importorskip("playwright.async_api")

from agi.web.browser_backend import BrowserBackendPool, PageInfo


@pytest.mark.asyncio
async def test_browser_backend_pool_reuses_sessions_per_user(tmp_path: Path) -> None:
    pool = BrowserBackendPool(storage_dir=str(tmp_path), idle_timeout_seconds=60)

    async with pool.session("alice") as alice_session_1:
        pass

    async with pool.session("bob") as bob_session:
        pass

    async with pool.session("alice") as alice_session_2:
        assert alice_session_1.backend is alice_session_2.backend
        assert alice_session_1.user_id == "alice"
        assert bob_session.user_id == "bob"
        assert alice_session_1.backend is not bob_session.backend

    await pool.close_all()


@pytest.mark.asyncio
async def test_browser_backend_pool_closes_idle_sessions_and_recreates(tmp_path: Path) -> None:
    pool = BrowserBackendPool(storage_dir=str(tmp_path), idle_timeout_seconds=0.001)

    async with pool.session("alice") as session:
        first_backend = session.backend
        first_backend.close = AsyncMock()

    assert session.idle_task is not None
    await session.idle_task
    first_backend.close.assert_awaited_once()

    async with pool.session("alice") as recreated_session:
        assert recreated_session.backend is not first_backend

    await pool.close_all()


class FakePool:
    def __init__(self) -> None:
        self.sessions: dict[str, SimpleNamespace] = {}

    @asynccontextmanager
    async def session(self, user_id: str):
        if user_id not in self.sessions:
            self.sessions[user_id] = SimpleNamespace(
                user_id=user_id,
                backend=SimpleNamespace(get_history=lambda: []),
                last_result=None,
            )
        yield self.sessions[user_id]


@pytest.mark.asyncio
async def test_browser_middleware_tracks_last_result_per_user() -> None:
    pytest.importorskip("langchain")

    from agi.agent.middlewares.browser_middleware import BrowserMiddleware

    middleware = BrowserMiddleware(enable_ocr_fallback=False, max_retries=1)
    middleware._session_pool = FakePool()

    async def fake_dispatch(session, action: str, **kwargs):
        return PageInfo(
            url=f"https://example.com/{session.user_id}/{action}",
            title=session.user_id,
            html="<html></html>",
            text=session.user_id,
            screenshot_path=None,
            metadata={"action": action, **kwargs},
        )

    middleware._dispatch_action = fake_dispatch
    middleware._maybe_apply_ocr = AsyncMock()

    alice_runtime = SimpleNamespace(context=SimpleNamespace(user_id="alice"), config={}, tool_call_id="tool-1")
    bob_runtime = SimpleNamespace(context=SimpleNamespace(user_id="bob"), config={}, tool_call_id="tool-2")

    alice_result = await middleware._execute_with_retry(alice_runtime, "navigate", url="https://example.com/alice")
    bob_result = await middleware._execute_with_retry(bob_runtime, "navigate", url="https://example.com/bob")

    assert alice_result.url == "https://example.com/alice/navigate"
    assert bob_result.url == "https://example.com/bob/navigate"
    assert middleware._session_pool.sessions["alice"].last_result == alice_result
    assert middleware._session_pool.sessions["bob"].last_result == bob_result
