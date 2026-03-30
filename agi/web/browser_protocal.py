# browser_interface.py
from abc import ABC, abstractmethod
from typing import Any, List, Dict, Optional
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
    async def click_by_text(self, text: str) -> PageInfo:
        """点击包含指定文本的元素。"""
        pass

    @abstractmethod
    async def fill(self, selector: str, value: str) -> PageInfo:
        """向匹配给定 CSS 选择器的输入框填充文本。"""
        pass

    @abstractmethod
    async def fill_by_label(self, label_text: str, value: str) -> PageInfo:
        """根据标签文本找到对应的输入框并填充。"""
        pass

    @abstractmethod
    async def fill_human_like(self, selector: str, value: str) -> PageInfo:
        """模拟人类打字速度向输入框填充文本。"""
        pass

    # --- 查询与获取 ---
    @abstractmethod
    async def find_elements(self, selector: str) -> List[QueryMatch]:
        """查找匹配给定 CSS 选择器的元素列表。"""
        pass

    @abstractmethod
    async def get_screenshot(self, *, full_page: bool = True) -> str:
        """获取当前页面的截图。"""
        pass

    # --- 历史与状态 ---
    @abstractmethod
    def get_history(self) -> List[Dict[str, Any]]:
        """获取操作历史记录。"""
        pass

    @abstractmethod
    def get_recent_events(self, limit: int = 5) -> List[Dict[str, Any]]:
        """获取最近的事件列表。"""
        pass

    @abstractmethod
    def peek_state_messages(self, limit: int = 1) -> List[Dict[str, Any]]:
        """预览而不消耗最新的状态消息。"""
        pass

    @abstractmethod
    def drain_state_messages(self, limit: int = 1) -> List[Dict[str, Any]]:
        """获取并清空最新的状态消息。"""
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
