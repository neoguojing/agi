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
MAX_FIND_RESULTS = 5
STATE_SNAPSHOT_FILENAME = "browser_session_state.json"
PLAYWRIGHT_STORAGE_STATE_FILENAME = "playwright_storage_state.json"

USER_EVENT_TYPES = {
    "dom_click", "dom_input", "dom_change", "dom_submit",
    "page_hashchange", "page_popstate", "page_focusin"
}

BROWSER_OBSERVER_SCRIPT = """(() => {
    if (window.__agiBrowserObserverInstalled) return;
    window.__agiBrowserObserverInstalled = true;

    const buildTarget = (target) => {
        if (!(target instanceof Element)) {
            return { tag: null, id: null, classes: [], text: "" };
        }
        return {
            tag: target.tagName ? target.tagName.toLowerCase() : null,
            id: target.id || null,
            name: target.getAttribute?.("name") || null,
            type: target.getAttribute?.("type") || null,
            classes: Array.from(target.classList || []),
            text: (target.innerText || target.textContent || "").trim().slice(0, 120),
            value: ["INPUT", "TEXTAREA", "SELECT"].includes(target.tagName) ? String(target.value || "").slice(0, 120) : null,
        };
    };

    const emit = (type, extra = {}) => {
        const payload = {
            type,
            url: window.location.href,
            title: document.title,
            timestamp: new Date().toISOString(),
            ...extra,
        };
        if (window.__agiRecordBrowserEvent) {
            window.__agiRecordBrowserEvent(payload).catch(() => undefined);
        }
    };

    document.addEventListener("click", (event) => emit("dom_click", { target: buildTarget(event.target) }), true);
    document.addEventListener("input", (event) => emit("dom_input", { target: buildTarget(event.target) }), true);
    document.addEventListener("change", (event) => emit("dom_change", { target: buildTarget(event.target) }), true);
    document.addEventListener("submit", (event) => emit("dom_submit", { target: buildTarget(event.target) }), true);
    document.addEventListener("focusin", (event) => emit("page_focusin", { target: buildTarget(event.target) }), true);
    window.addEventListener("hashchange", () => emit("page_hashchange"), true);
    window.addEventListener("popstate", () => emit("page_popstate"), true);
})();"""

WaitUntilState = Literal["commit", "domcontentloaded", "load", "networkidle"]


# --- Action result payloads ---
@dataclass(slots=True)
class PageInfo:
    url: str
    title: str | None
    html: str | None
    text: str | None
    screenshot_path: str | None
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass(slots=True)
class QueryMatch:
    selector: str
    text: str
    attributes: dict[str, Any]


# --- Unified session snapshot exposed to upper layers ---
class BrowserPageState(TypedDict):
    url: str
    title: str | None
    load_state: str


class BrowserRuntimeState(TypedDict):
    is_open: bool
    is_closed: bool


class BrowserSessionSnapshot(TypedDict):
    browser: BrowserRuntimeState
    current_page: BrowserPageState
    previous_page: NotRequired[BrowserPageState | None]


class BrowserEventType(str, Enum):
    NAVIGATE = "navigate"
    CLICK = "click"
    CLICK_INTERCEPTED = "click_intercepted"
    FILL = "fill"
    DOM_MUTATION = "dom_mutation"
    PAGE_READY = "page_ready"
    PAGE_OPENED = "page_opened"
    PAGE_CLOSED = "browser_closed"
    TITLE_CHANGED = "title_changed"
    INSTRUMENTED = "page_instrumented"
    NAVIGATION_START = "navigation_start"

@dataclass
class PageRuntimeState:
    """页面的实时运行状态"""
    page_id: str
    url: str = ""
    title: str = ""
    load_state: str = "unknown"
    user_interaction_count: int = 0
    last_update: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class BrowserEvent:
    """标准化的事件模型"""
    type: BrowserEventType
    timestamp: str
    page_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type.value,
            "timestamp": self.timestamp,
            "page_id": self.page_id,
            "metadata": self.metadata
        }
