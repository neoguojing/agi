from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

pytest.importorskip("playwright.async_api")

from .browser_backend import MAX_FIND_RESULTS, PageInfo, PlaywrightTimeoutError, StatefulBrowserBackend


class FakeResponse:
    def __init__(self, status: int = 200):
        self.status = status


class FakeElement:
    def __init__(self, text: str = "", attributes: dict | None = None):
        self._text = text
        self._attributes = attributes or {}
        self.scroll_into_view_if_needed = AsyncMock()
        self.click = AsyncMock()
        self.fill = AsyncMock()

    async def inner_text(self) -> str:
        return self._text

    async def evaluate(self, _script: str) -> dict:
        return self._attributes


class FakePage:
    def __init__(self) -> None:
        self.url = "https://example.com/current"
        self.keyboard = SimpleNamespace(type=AsyncMock())
        self.goto = AsyncMock(return_value=FakeResponse(200))
        self.click = AsyncMock()
        self.fill = AsyncMock()
        self.focus = AsyncMock()
        self.wait_for_timeout = AsyncMock()
        self.wait_for_load_state = AsyncMock()
        self.content = AsyncMock(return_value="<html><body>Hello world</body></html>")
        self.inner_text = AsyncMock(return_value="Hello world")
        self.title = AsyncMock(return_value="Example")
        self.query_selector = AsyncMock(return_value=FakeElement())
        self.query_selector_all = AsyncMock(return_value=[])
        self.screenshot = AsyncMock()
        self.add_init_script = AsyncMock()
        self._listeners = {}

    def on(self, event: str, callback):
        self._listeners[event] = callback

    def is_closed(self) -> bool:
        return False


@pytest.fixture
def backend(tmp_path: Path) -> StatefulBrowserBackend:
    instance = StatefulBrowserBackend(
        storage_dir=str(tmp_path),
        headless=True,
        max_content_length=20,
        max_retry=1,
    )
    instance._browser = object()
    instance._page = FakePage()
    instance._context = SimpleNamespace(pages=[instance._page])
    return instance


@pytest.mark.asyncio
async def test_read_screenshot_bytes_reads_saved_file(backend: StatefulBrowserBackend, tmp_path: Path) -> None:
    image_path = tmp_path / "capture.png"
    image_path.write_bytes(b"png-bytes")
    backend.get_screenshot = AsyncMock(return_value=str(image_path))

    screenshot = await backend.read_screenshot_bytes()

    assert screenshot == (str(image_path), b"png-bytes")
    backend.get_screenshot.assert_awaited_once_with(full_page=True)


@pytest.mark.asyncio
async def test_capture_page_info_truncates_html_and_marks_ocr_ready(backend: StatefulBrowserBackend, tmp_path: Path) -> None:
    page = backend._page
    assert page is not None
    page.content.return_value = "x" * 50
    page.inner_text.return_value = "short"
    page.title.return_value = "Short page"

    screenshot_path = tmp_path / "page-shot.png"
    backend._take_screenshot = AsyncMock(return_value=screenshot_path)

    result = await backend._capture_page_info(page, "https://example.com", FakeResponse(status=403))

    assert result.url == page.url
    assert result.title == "Short page"
    assert result.html.endswith("[TRUNCATED]")
    assert result.screenshot_path == str(screenshot_path)
    assert result.metadata["status"] == 403
    assert result.metadata["ocr_ready"] is True
    backend._take_screenshot.assert_awaited_once_with(page, prefix="page", full_page=True)


@pytest.mark.asyncio
async def test_run_page_action_records_history_and_uses_capture(backend: StatefulBrowserBackend) -> None:
    page = backend._page
    assert page is not None
    captured = PageInfo(
        url=page.url,
        title="Captured",
        html="<html></html>",
        text="captured",
        screenshot_path=None,
        metadata={"status": 200},
    )
    backend._smart_wait = AsyncMock()
    backend._capture_page_info = AsyncMock(return_value=captured)

    async def operation(current_page):
        assert current_page is page
        return FakeResponse(status=201)

    history_entry = {"action": "navigate", "url": "https://example.com"}
    result = await backend._run_page_action(
        action="navigate",
        operation=operation,
        capture_url="https://example.com",
        history_entry=history_entry,
    )

    assert result == captured
    backend._smart_wait.assert_awaited_once_with(page)
    backend._capture_page_info.assert_awaited_once()


@pytest.mark.asyncio
async def test_run_page_action_returns_error_after_timeouts(backend: StatefulBrowserBackend) -> None:
    async def operation(_page):
        raise PlaywrightTimeoutError("timed out")

    result = await backend._run_page_action(
        action="navigate",
        operation=operation,
        capture_url="https://example.com",
        history_entry=None,
    )

    assert result.metadata["error"].startswith("Max retries exceeded")
    assert result.metadata["action"] == "navigate"


@pytest.mark.asyncio
async def test_click_by_text_returns_error_when_match_missing(backend: StatefulBrowserBackend) -> None:
    page = backend._page
    assert page is not None
    page.query_selector_all.return_value = []

    result = await backend.click_by_text("Missing")

    assert result.metadata["error"] == "No element with text 'Missing'"


@pytest.mark.asyncio
async def test_click_by_text_clicks_first_match_and_updates_history(backend: StatefulBrowserBackend) -> None:
    page = backend._page
    assert page is not None
    element = FakeElement(text="Open")
    page.query_selector_all.return_value = [element]
    backend._human_delay = AsyncMock()
    backend._smart_wait = AsyncMock()
    expected = PageInfo(
        url=page.url,
        title="After click",
        html="<html></html>",
        text="done",
        screenshot_path=None,
        metadata={},
    )
    backend._capture_page_info = AsyncMock(return_value=expected)

    result = await backend.click_by_text("Open")

    assert result == expected
    element.scroll_into_view_if_needed.assert_awaited_once()
    element.click.assert_awaited_once()


@pytest.mark.asyncio
async def test_find_elements_limits_results_and_returns_attributes(backend: StatefulBrowserBackend) -> None:
    page = backend._page
    assert page is not None
    elements = [
        FakeElement(text=f"item-{idx}", attributes={"data-id": str(idx)})
        for idx in range(MAX_FIND_RESULTS + 5)
    ]
    page.query_selector_all.return_value = elements

    results = await backend.find_elements(".item")

    assert len(results) == MAX_FIND_RESULTS
    assert results[0].selector == ".item"
    assert results[0].text == "item-0"
    assert results[0].attributes == {"data-id": "0"}


@pytest.mark.asyncio
async def test_fill_human_like_types_each_character(backend: StatefulBrowserBackend) -> None:
    page = backend._page
    assert page is not None
    backend._scroll_into_view = AsyncMock()
    backend._human_delay = AsyncMock()
    backend._smart_wait = AsyncMock()
    backend._capture_page_info = AsyncMock(
        return_value=PageInfo(
            url=page.url,
            title="Filled",
            html="<html></html>",
            text="typed",
            screenshot_path=None,
            metadata={},
        )
    )

    result = await backend.fill_human_like("#query", "abc")

    page.focus.assert_awaited_once_with("#query")
    page.fill.assert_awaited_once_with("#query", "")
    assert page.keyboard.type.await_count == 3
    assert result.title == "Filled"


@pytest.mark.asyncio
async def test_get_state_snapshot_exposes_browser_context_and_page_events(backend: StatefulBrowserBackend) -> None:
    page = backend._page
    assert page is not None

    await backend._instrument_page(page, source="test")
    backend._record_event("page_navigated", page=page, metadata={"url": page.url})
    backend._record_event("page_load", page=page)

    result = PageInfo(
        url=page.url,
        title="Example",
        html="<html></html>",
        text="hello",
        screenshot_path=None,
        metadata={},
    )
    snapshot = backend.get_state_snapshot(user_id="alice", last_result=result)

    assert snapshot["user_id"] == "alice"
    assert snapshot["browser"]["is_closed"] is False
    assert snapshot["context"]["page_count"] == 1
    assert snapshot["page"]["url"] == page.url
    assert snapshot["page"]["load_state"] == "loaded"
    assert snapshot["recent_events"][-1]["type"] == "page_load"


@pytest.mark.asyncio
async def test_record_page_dom_event_updates_runtime_state(backend: StatefulBrowserBackend) -> None:
    page = backend._page
    assert page is not None

    await backend._record_page_dom_event(
        None,
        {
            "type": "dom_input",
            "url": page.url,
            "title": "Interactive Page",
            "timestamp": "2026-03-23T00:00:00Z",
            "target": {"tag": "input", "text": "", "value": "hello"},
        },
    )

    snapshot = backend.get_state_snapshot(user_id="alice")

    assert snapshot["page"]["last_user_event"]["type"] == "dom_input"
    assert snapshot["page"]["last_interaction"]["target"]["value"] == "hello"
    assert snapshot["page"]["user_interaction_count"] == 1
    assert snapshot["context"]["recent_user_events"][-1]["type"] == "dom_input"


@pytest.mark.asyncio
async def test_state_snapshot_is_persisted_and_restored(tmp_path: Path) -> None:
    backend = StatefulBrowserBackend(storage_dir=str(tmp_path), headless=True, max_retry=1)
    backend._browser = object()
    backend._page = FakePage()
    backend._context = SimpleNamespace(pages=[backend._page])

    await backend._record_page_dom_event(
        None,
        {
            "type": "dom_click",
            "url": backend._page.url,
            "title": "Persisted Page",
            "timestamp": "2026-03-23T00:00:01Z",
            "target": {"tag": "button", "text": "Continue"},
        },
    )

    restored = StatefulBrowserBackend(storage_dir=str(tmp_path), headless=True, max_retry=1)
    snapshot = restored.get_state_snapshot(user_id="alice")

    assert snapshot["browser"]["was_restored"] is True
    assert snapshot["page"]["last_user_event"]["type"] == "dom_click"
    assert snapshot["state_messages"][-1]["event"]["type"] == "dom_click"

