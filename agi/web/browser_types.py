from typing import Dict, List, Any, Optional, Literal

# --- 常量 ---
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)
DEFAULT_VIEWPORT = {"width": 1280, "height": 720}
DEFAULT_WAIT_UNTIL = "networkidle"
MAX_FIND_RESULTS = 5
STATE_SNAPSHOT_FILENAME = "browser_session_state.json"
PLAYWRIGHT_STORAGE_STATE_FILENAME = "playwright_storage_state.json"

# --- 超时常量 (ms) ---
DEFAULT_CLICK_TIMEOUT_MS = 5_000
DEFAULT_SCROLL_TIMEOUT_MS = 2_000
DEFAULT_SMART_WAIT_TIMEOUT_MS = 5_000
DEFAULT_CAPTURE_DELAY_MS = 300
DEFAULT_NETWORK_IDLE_TIMEOUT_MS = 5_000

WaitUntilState = Literal["load", "domcontentloaded", "networkidle", "commit"]


class Rect:
    """Element bounding box coordinates."""
    
    def __init__(self, x: float = 0.0, y: float = 0.0, width: float = 0.0, height: float = 0.0):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
    
    def __repr__(self) -> str:
        return f"Rect(x={self.x}, y={self.y}, width={self.width}, height={self.height})"


class PageInfo:
    """Page state snapshot for browser automation."""
    
    def __init__(
        self,
        url: str,
        title: Optional[str] = None,
        viewport: Dict[str, int] = DEFAULT_VIEWPORT,
        is_loading: bool = False,
        last_action_status: Literal["success", "fail", "timeout"] = "unknown",
        error_message: Optional[str] = None,
    ):
        self.url = url
        self.title = title
        self.viewport = viewport
        self.is_loading = is_loading
        self.last_action_status = last_action_status
        self.error_message = error_message
    
    def __repr__(self) -> str:
        return f"PageInfo(url={self.url!r}, title={self.title!r}, status={self.last_action_status})"


class QueryMatch:
    """A matched UI element with its properties."""
    
    def __init__(
        self,
        id: str,
        selector: str,
        tag_name: str,
        text: str,
        attributes: Dict[str, str],
        rect: Rect,
        is_visible: bool = True,
        is_enabled: bool = True,
    ):
        self.id = id
        self.selector = selector
        self.tag_name = tag_name
        self.text = text
        self.attributes = attributes
        self.rect = rect
        self.is_visible = is_visible
        self.is_enabled = is_enabled
    
    def __repr__(self) -> str:
        return f"QueryMatch(id={self.id}, selector={self.selector!r}, tag={self.tag_name})"


class BrowserSessionSnapshot:
    """Canonical browser session state for middleware."""
    
    def __init__(
        self,
        browser: Dict[str, Any],
        current_page: Optional[Dict[str, Any]],
        previous_page: Optional[Dict[str, Any]],
    ):
        self.browser = browser
        self.current_page = current_page
        self.previous_page = previous_page
