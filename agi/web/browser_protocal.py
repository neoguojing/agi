from abc import ABC, abstractmethod
from typing import TypedDict, List, Dict, Any, Optional, Literal
from .browser_types import PageInfo, QueryMatch, WaitUntilState


class AbstractBrowserBackend(ABC):
    """
    浏览器后端的抽象接口。
    设计原则：
    1. 所有的交互动作 (Actions) 必须返回执行后的 PageInfo。
    2. 视觉探测 (Extraction) 必须包含 Viewport 相关的空间信息。
    3. 异常处理：非致命错误应封装在 PageInfo 中，而非直接抛出。
    """

    # --- 生命周期管理 ---
    @abstractmethod
    async def initialize(self, viewport_size: Dict[str, int] = {"width": 1280, "height": 720}) -> None:
        """初始化浏览器。建议在此处设定初始 Viewport 大小以保证响应式布局一致."""
        pass

    @abstractmethod
    async def close(self) -> None:
        """释放所有资源（浏览器、上下文、页面）。"""
        pass

    @property
    @abstractmethod
    def is_closed(self) -> bool:
        """返回当前实例是否已失效。"""
        pass

    # --- 页面导航与快照 ---
    @abstractmethod
    async def navigate(self, url: str, wait_until: WaitUntilState = "networkidle") -> PageInfo:
        """跳转 URL 并等待状态。返回最新的页面快照。"""
        pass

    @abstractmethod
    async def get_state_snapshot(self) -> PageInfo:
        """实时获取当前页面的核心状态信息。"""
        pass

    @abstractmethod
    async def get_screenshot(self, *, full_page: bool = False) -> str:
        """
        获取 Base64 编码的 PNG 截图。
        full_page=False 时仅截取 Viewport 区域。
        """
        pass

    # --- 元素交互动作 ---
    # 每一个动作都代表一次状态改变，因此返回 PageInfo
    @abstractmethod
    async def click(self, selector: str) -> PageInfo:
        """点击指定的 CSS 选择器对应的元素。"""
        pass

    @abstractmethod
    async def fill(self, selector: str, value: str) -> PageInfo:
        """清空输入框并填充指定文本。"""
        pass

    @abstractmethod
    async def scroll(self, direction: Literal["up", "down"] = "down", distance: int = 800) -> PageInfo:
        """
        移动 Viewport 视口。
        distance: 滚动的像素值。
        """
        pass

    # --- 查询与 UI 提取 ---
    @abstractmethod
    async def find_elements(self, selector: str) -> List[QueryMatch]:
        """
        局部查询。用于获取一组相似元素的详情（如搜索结果列表）。
        必须包含每个元素的物理坐标 (Rect) 和可见性。
        """
        pass

    @abstractmethod
    async def extract_ui(self, limit: int = 50) -> List[QueryMatch]:
        """
        全局提取。将当前 Viewport 内的可交互元素转换为 AOM (辅助功能模型) 风格。
        这是 AI 决策的主要输入源。
        """
        pass

    @abstractmethod
    async def inspect_element_property(self, selector: str, property_name: str) -> Any:
        """
        深度探测某个元素的运行时属性（如 get_attribute, computed_style）。
        """
        pass

    # --- 等待机制 (生产环境必备) ---
    @abstractmethod
    async def wait_for_selector(self, selector: str, timeout: int = 5000) -> Optional[QueryMatch]:
        """等待元素出现在 DOM 中且变为可见。"""
        pass
