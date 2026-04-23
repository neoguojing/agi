from typing import TypedDict, List, Dict, Any, Optional, Literal

# --- 常量 ---
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)
DEFAULT_VIEWPORT = {"width": 1280, "height": 720}
DEFAULT_WAIT_UNTIL = "domcontentloaded"
MAX_FIND_RESULTS = 5
STATE_SNAPSHOT_FILENAME = "browser_session_state.json"
PLAYWRIGHT_STORAGE_STATE_FILENAME = "playwright_storage_state.json"

WaitUntilState = Literal["load", "domcontentloaded", "networkidle", "commit"]

class Rect(TypedDict):
    x: float
    y: float
    width: float
    height: float

class PageInfo(TypedDict):
    url: str
    title: str
    viewport: Dict[str, int]  # {'width': int, 'height': int}
    is_loading: bool
    # 记录最后一次操作的结果，便于 AI 诊断
    last_action_status: Literal["success", "fail", "timeout"]
    error_message: Optional[str]

class QueryMatch(TypedDict):
    id: str                   # 内部唯一 ID (backend_node_id)
    selector: str             # 匹配到的选择器
    tag_name: str
    text: str
    attributes: Dict[str, str]
    rect: Rect                # 元素在视口中的位置
    is_visible: bool          # 关键：是否在 Viewport 内可见
    is_enabled: bool
