import logging
import asyncio
from collections import deque
from datetime import datetime
from pathlib import Path
from enum import Enum
from dataclasses import dataclass, field, asdict
from typing import Deque, Dict, List, Any, Optional, Set
from playwright.async_api import Page, BrowserContext

from .browser_types import BrowserEvent,PageRuntimeState,BrowserEventType
logger = logging.getLogger(__name__)


# ... (前置导入保持不变)

class BrowserEventManager:
    def __init__(self, storage_dir: Path):
        self.storage_dir = storage_dir
        # 长期存储：用于生成最终报告或审计日志
        self._history: List[Dict[str, Any]] = []
        # 短期内存：保存最近 20 条原始事件，用于调试
        self._recent_events: Deque[BrowserEvent] = deque(maxlen=20)
        # 消息队列：用于 UI 消费或流式输出当前状态（先进先出）
        self._state_messages: Deque[Dict[str, Any]] = deque(maxlen=10)
        
        self._active_page: Optional[Page] = None
        # 页面状态追踪：key 为 page_id，存储标题、URL、交互计数等
        self._page_runtime_state: Dict[str, PageRuntimeState] = {}
        # 幂等性控制：防止对同一个页面重复注入脚本
        self._instrumented_pages: Set[str] = set()

    # --- 状态访问器 ---
    def _page_id(self, page: Page | None) -> str:
        if page is None:
            return "page:none"
        return f"page:{id(page)}"
    
    def _page_is_closed(self, page: Page | None) -> bool:
        if page is None:
            return True
        is_closed = getattr(page, "is_closed", None)
        if callable(is_closed):
            try:
                return bool(is_closed())
            except Exception:
                return False
        return False
    
    def set_active_page(self, page: Optional[Page]) -> None:
        """
        设置当前正在操作的活跃页面。
        1. 更新内部引用。
        2. 如果该页面是首次被设为活跃，则初始化其运行时状态。
        """
        self._active_page = page
        
        if page is not None:
            page_id = self._page_id(page)
            
            # 确保活跃页面在状态字典中占位
            if page_id not in self._page_runtime_state:
                self._page_runtime_state[page_id] = PageRuntimeState(
                    page_id=page_id,
                    url=page.url,
                    title="Active Page (Initializing...)",
                    last_update=datetime.now().isoformat()
                )
                
            logger.debug(f"Active page set to: {page_id} ({page.url})")
        else:
            logger.debug("Active page cleared (set to None)")

    def update_page_state(self, page: Page, **kwargs) -> None:
        """
        更新特定页面的运行时状态。
        如果页面是首次出现，则初始化 PageRuntimeState 对象。
        """
        page_id = self._page_id(page)
        if page_id not in self._page_runtime_state:
            self._page_runtime_state[page_id] = PageRuntimeState(
                page_id=page_id, 
                url=page.url
            )
        
        state = self._page_runtime_state[page_id]
        # 动态更新属性，例如 update_page_state(page, title="新标题")
        for key, value in kwargs.items():
            if hasattr(state, key) and value is not None:
                setattr(state, key, value)
        
        state.last_update = datetime.now().isoformat()

    # --- 事件记录 ---

    def record_event(self, event_type: BrowserEventType, page: Optional[Page], metadata: Dict[str, Any] = {}) -> None:
        """
        统一事件记录入口。处理 Python 层触发和 JS 层传回的所有事件。
        """
        page_id = self._page_id(page)
        event = BrowserEvent(
            type=event_type,
            timestamp=datetime.now().isoformat(),
            page_id=page_id,
            metadata=metadata
        )
        
        # 1. 存入滚动队列
        self._recent_events.append(event)
        # 2. 包装成消息格式，等待被 drain_state_messages 消费
        self._state_messages.append({
            "type": "event",
            "data": event.to_dict()
        })

        # --- 业务逻辑：根据事件类型自动更新状态 ---
        if event_type in [BrowserEventType.CLICK_INTERCEPTED, BrowserEventType.CLICK]:
            if page_id in self._page_runtime_state:
                # 记录用户在页面上的总交互次数
                self._page_runtime_state[page_id].user_interaction_count += 1
        
        # 自动同步元数据中的基础信息到 RuntimeState
        if "title" in metadata:
            self.update_page_state(page, title=metadata["title"])
        if "url" in metadata:
            self.update_page_state(page, url=metadata["url"])

    def add_to_history(self, history_entry: Dict[str, Any]) -> None:
        """
        添加历史记录条目。
        用于在动作执行后记录历史，与 record_event 分开管理。
        """
        self._history.append(history_entry)

    # --- Playwright 仪器化逻辑 (Core) ---
    async def register_context_instrumentation(self, context: BrowserContext) -> None:
        """
        为整个浏览器上下文注册监听器。
        这是自动化监控的"入水口"，负责捕获所有新开启的标签页。
        """
        # 监听新页面打开事件
        # 当 target="_blank" 点击或 window.open() 触发时，Playwright 会抛出此事件
        context.on("page", self._handle_new_page)
        
        # 监听页面关闭事件（可选，用于实时清理状态）
        # 注意：context.on("page", ...) 返回的是 Page 对象，我们需要在该对象上挂载 close 监听
        
        logger.debug("Registered context-level page listeners.")

    async def _handle_new_page(self, page: Page) -> None:
        """
        当浏览器开启新标签页时自动触发的监听器。
        """
        # 等待 DOM 准备就绪，确保可以执行脚本注入
        await page.wait_for_load_state("domcontentloaded")
        
        # 过滤掉已关闭的页面，统计当前活跃标签数
        page_count = len([p for p in page.context.pages if not self._page_is_closed(p)])
        
        self.record_event(
            BrowserEventType.PAGE_OPENED,
            page=page,
            metadata={"page_count": page_count, "url": page.url},
        )
        # 立即对新页面进行仪器化（注入监控脚本）
        await self.instrument_page(page, source="new_page_event")

    async def instrument_page(self, page: Page, *, source: str) -> None:
        """
        核心：向浏览器页面注入 Observer 脚本，并建立 Python 通信桥梁。
        """
        page_id = self._page_id(page)
        if page_id in self._instrumented_pages:
            return

        # 待注入的 JavaScript 脚本
        BROWSER_OBSERVER_SCRIPT = """
        (function() {
            window.__agiLastTitle = document.title;
            // 封装发送函数，调用 Python 暴露的接口
            function send(type, data) {
                window.__agiRecordBrowserEvent({ type: type, ...data });
            }

            // 1. 监听 DOM 结构变化 (增删改)
            const observer = new MutationObserver(() => {
                send('dom_mutation', { timestamp: Date.now() });
            });
            observer.observe(document, { childList: true, subtree: true });

            // 2. 监听全局点击事件 (捕获阶段)
            document.addEventListener('click', (e) => {
                const rect = e.target.getBoundingClientRect();
                send('click_intercepted', {
                    element_tag: e.target.tagName,
                    element_text: e.target.innerText?.substring(0, 50),
                    clientX: e.clientX,
                    clientY: e.clientY
                });
            }, true);

            // 3. 轮询检测页面标题变化 (针对 SPA 应用)
            setInterval(() => {
                if (document.title !== window.__agiLastTitle) {
                    window.__agiLastTitle = document.title;
                    send('title_changed', { title: document.title });
                }
            }, 1000);

            send('page_ready', { title: document.title, url: window.location.href });
        })();
        """
        try:
            # Step A: 注入脚本。使用 add_init_script 确保在页面加载最初期执行。
            await page.add_init_script(BROWSER_OBSERVER_SCRIPT)
            
            self._instrumented_pages.add(page_id)
            self.update_page_state(page, title=await page.title(), load_state="ready")
            
            self.record_event(BrowserEventType.INSTRUMENTED, page=page, metadata={"source": source})
        except Exception as e:
            logger.error(f"Failed to instrument page {page_id}: {e}")

    async def _on_browser_event(self, source, event_data: Dict[str, Any]) -> None:
        """
        JS 回调的统一 Python 入口。
        source.page 允许我们定位是哪个标签页发出的消息。
        """
        raw_type = event_data.get("type", "unknown")
        try:
            # 尝试将字符串转换为 Python Enum 类型
            etype = BrowserEventType(raw_type)
        except ValueError:
            # 如果是未定义的事件类型则忽略
            return

        # 将 JS 传来的数据转发到记录器
        self.record_event(etype, page=source.page, metadata=event_data)

    
    # --- 状态查询与清理 ---

    def peek_state_messages(self, limit: int = 1) -> List[Dict[str, Any]]:
        """查看队列中最早的消息，但不移除它们（用于监控）"""
        # 转换为 list 以便切片，取最左侧（最早）的 limit 条
        return list(self._state_messages)[:limit]

    def drain_state_messages(self, limit: int = 1) -> List[Dict[str, Any]]:
        """弹出并返回队列中最早的消息（消耗型读取）"""
        messages = []
        while self._state_messages and len(messages) < limit:
            messages.append(self._state_messages.popleft())
        return messages

    def get_recent_events(self, limit: int = 5) -> List[Dict[str, Any]]:
        """获取最近发生的原始事件列表（从最新的开始）"""
        # reversed 是因为 deque.append 是往右边加，右边是最新的
        events = list(self._recent_events)
        events.reverse()
        return [e.to_dict() if hasattr(e, "to_dict") else asdict(e) for e in events[:limit]]

    def get_history(self) -> List[Dict[str, Any]]:
        """获取完整的动作历史记录"""
        return self._history.copy()
    
    def get_page_runtime_state(self, page_id: str) -> Optional[PageRuntimeState]:
        """
        根据页面 ID 获取该页面的实时运行状态。
        如果 page_id 尚未注册或页面已关闭，返回 None。
        """
        # 从初始化的字典中检索
        state = self._page_runtime_state.get(page_id)
        
        if state:
            # 可以在此处进行最后的活性检查（可选）
            # logger.debug(f"Retrieved state for {page_id}: {state.url}")
            return state
            
        return None
