"""Shared browser domain types.

This module intentionally keeps all browser-facing state shapes in one place so
middleware/backend/session layers read and write the same schema.
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Literal
from typing_extensions import NotRequired, TypedDict


# --- 常量 ---
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)
DEFAULT_VIEWPORT = {"width": 1280, "height": 720}
DEFAULT_WAIT_UNTIL = "domcontentloaded"
DEFAULT_SMART_WAIT_TIMEOUT_MS = 5_000
DEFAULT_CLICK_TIMEOUT_MS = 5_000
DEFAULT_SCROLL_TIMEOUT_MS = 2_000
DEFAULT_CAPTURE_DELAY_MS = 300
DEFAULT_NETWORK_IDLE_TIMEOUT_MS = 5_000
MAX_FIND_RESULTS = 5
STATE_SNAPSHOT_FILENAME = "browser_session_state.json"
PLAYWRIGHT_STORAGE_STATE_FILENAME = "playwright_storage_state.json"


WaitUntilState = Literal["commit", "domcontentloaded", "load", "networkidle"]


def build_browser_runtime_key(user_id: str, conversation_id: str | None = None) -> str:
    """Build a stable runtime key for browser session routing."""
    # session_id = (conversation_id or "default").strip() or "default"
    return user_id


# --- Action result payloads ---
@dataclass(slots=True)
class PageInfo:
    """Canonical browser page state for agent planning."""

    url: str
    title: str | None
    dom_snapshot: str | None
    page_text: str | None
    screenshot_path: str | None
    response_status: int | None = None
    last_action: str | None = None
    actionable_elements: list[dict[str, Any]] = field(default_factory=list)
    network_idle: bool | None = None
    url_changed: bool | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass(slots=True)
class BrowserToolResult:
    """Generic result for browser tool operations that don't necessarily result in a new page state."""
    status: str
    content: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


@dataclass(slots=True)
class QueryMatch:
    selector: str
    text: str
    attributes: dict[str, Any]

# --- Unified session snapshot exposed to upper layers ---
class BrowserHistoryEntry(TypedDict):
    action: str
    timestamp: str
    params: dict[str, Any]


class BrowserRuntimeState(TypedDict):
    is_open: bool
    is_closed: bool


class BrowserSessionSnapshot(TypedDict):
    browser: BrowserRuntimeState
    current_page: dict[str, Any]
    previous_page: NotRequired[dict[str, Any] | None]



def _normalize_page_snapshot(page: Any) -> dict[str, Any]:
    if isinstance(page, PageInfo):
        return {
            "url": page.url,
            "title": page.title,
            "dom_snapshot": page.dom_snapshot,
            "page_text": page.page_text,
            "screenshot_path": page.screenshot_path,
            "response_status": page.response_status,
            "last_action": page.last_action,
            "actionable_elements": list(page.actionable_elements),
            "network_idle": page.network_idle,
            "url_changed": page.url_changed,
            "diagnostics": dict(page.diagnostics),
            "metadata": dict(page.metadata),
        }
    if not isinstance(page, dict):
        return {
            "url": "",
            "title": None,
            "dom_snapshot": None,
            "page_text": None,
            "screenshot_path": None,
            "response_status": None,
            "last_action": None,
            "actionable_elements": [],
            "network_idle": None,
            "url_changed": None,
            "diagnostics": {},
            "metadata": {},
        }
    dom_snapshot = page.get("dom_snapshot", page.get("html"))
    page_text = page.get("page_text", page.get("text"))
    response_status = page.get("response_status", page.get("status"))
    last_action = page.get("last_action", page.get("action"))
    return {
        "url": str(page.get("url", "")),
        "title": page.get("title"),
        "dom_snapshot": dom_snapshot,
        "page_text": page_text,
        "screenshot_path": page.get("screenshot_path"),
        "response_status": response_status,
        "last_action": last_action,
        "actionable_elements": list(page.get("actionable_elements", [])) if isinstance(page.get("actionable_elements"), list) else [],
        "network_idle": page.get("network_idle"),
        "url_changed": page.get("url_changed"),
        "diagnostics": dict(page.get("diagnostics", {})) if isinstance(page.get("diagnostics"), dict) else {},
        "metadata": dict(page.get("metadata", {})) if isinstance(page.get("metadata"), dict) else {},
    }


def normalize_browser_session_snapshot(state: dict[str, Any] | None) -> BrowserSessionSnapshot:
    """Single normalization entrypoint for middleware/backend session schema."""
    source = state or {}
    browser = source.get("browser", {})
    browser_state: BrowserRuntimeState = {
        "is_open": bool(browser.get("is_open", False)) if isinstance(browser, dict) else False,
        "is_closed": bool(browser.get("is_closed", True)) if isinstance(browser, dict) else True,
    }
    previous = source.get("previous_page")
    return {
        "browser": browser_state,
        "current_page": _normalize_page_snapshot(source.get("current_page")),
        "previous_page": _normalize_page_snapshot(previous) if previous is not None else None,
    }
