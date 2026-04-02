from pathlib import Path

import pytest

pytest.importorskip("playwright.async_api")

from .browser_backend import PageInfo, StatefulBrowserBackend


def test_normalize_actionable_elements_accepts_legacy_keys(tmp_path: Path) -> None:
    backend = StatefulBrowserBackend(storage_dir=str(tmp_path), headless=True, max_retry=1)

    normalized = backend._normalize_actionable_elements(
        [
            {"t": "button", "c": "Search", "sel": "#search", "action": "click"},
            {"type": "input", "text": "", "selector": "#q", "placeholder": "query"},
            {"foo": "bar"},
        ]
    )

    assert normalized == [
        {
            "type": "button",
            "text": "Search",
            "placeholder": "",
            "selector": "#search",
            "action": "click",
        },
        {
            "type": "input",
            "text": "",
            "placeholder": "query",
            "selector": "#q",
            "action": "",
        },
    ]


def test_page_summary_keeps_actionable_elements(tmp_path: Path) -> None:
    backend = StatefulBrowserBackend(storage_dir=str(tmp_path), headless=True, max_retry=1)
    info = PageInfo(
        url="https://example.com",
        title="Example",
        html="<html></html>",
        text="example",
        screenshot_path=None,
        metadata={
            "actionable_count": 1,
            "actionable_elements": [{"type": "button", "selector": "#go", "text": "Go", "placeholder": ""}],
        },
    )

    summary = backend._page_summary(info)

    assert summary["metadata"]["actionable_count"] == 1
    assert summary["metadata"]["actionable_elements"][0]["selector"] == "#go"
