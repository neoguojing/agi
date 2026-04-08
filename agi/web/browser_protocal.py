from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from .browser_types import PageInfo, QueryMatch, WaitUntilState

class AbstractBrowserBackend(ABC):
    """
    浏览器后端的抽象接口。
    所有具体的浏览器实现（如 Playwright, Selenium）都必须继承此类并实现其方法。
    """

    # --- 生命周期管理 ---
    @abstractmethod
    async def initialize(self) -> None:
        """初始化浏览器实例。"""
        pass

    @abstractmethod
    async def close(self) -> None:
        """关闭浏览器实例并释放资源。"""
        pass

    @property
    @abstractmethod
    def is_closed(self) -> bool:
        """检查浏览器是否已关闭。"""
        pass

    # --- 页面操作 ---
    @abstractmethod
    async def navigate(self, url: str, wait_until: WaitUntilState = "networkidle") -> PageInfo:
        """导航到指定 URL。"""
        pass

    @abstractmethod
    async def click(self, selector: str) -> PageInfo:
        """点击匹配给定 CSS 选择器的元素。"""
        pass

    @abstractmethod
    async def fill(self, selector: str, value: str) -> PageInfo:
        """向匹配给定 CSS 选择器的输入框填充文本。"""
        pass

    @abstractmethod
    async def scroll(self, direction: str = "down", distance: int = 800) -> PageInfo:
        """滚动视口，暴露视口外内容并触发懒加载。"""
        pass

    @abstractmethod
    async def inspect_element_property(self, selector: str, property_name: str) -> Dict[str, Any]:
        """探测元素实时交互属性（如 disabled / aria-busy）。"""
        pass

    # --- 查询与获取 ---
    @abstractmethod
    async def extract_ui(self, limit: int = 12) -> Dict[str, Any]:
        """抽取当前页面精简可交互结构（AOM风格）。"""
        pass

    @abstractmethod
    async def find_elements(self, selector: str) -> List[QueryMatch]:
        """查找匹配给定 CSS 选择器的元素列表。"""
        pass

    @abstractmethod
    async def get_screenshot(self, *, full_page: bool = True) -> str:
        """获取当前页面的截图。"""
        pass

    @abstractmethod
    def get_state_snapshot(
        self,
        *,
        user_id: Optional[str] = None,
        last_result: Optional[PageInfo] = None,
        previous_result: Optional[PageInfo] = None,
    ) -> Dict[str, Any]:
        """获取当前浏览器状态的快照。"""
        pass
