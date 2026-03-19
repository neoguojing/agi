import base64
import uuid
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from asyncio import Lock
from pathlib import Path
from playwright.async_api import async_playwright, Browser, BrowserContext, Page

logger = logging.getLogger(__name__)

@dataclass
class PageInfo:
    url: str
    title: Optional[str]
    html: Optional[str]
    text: Optional[str]
    screenshot_path: Optional[str]
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class QueryMatch:
    selector: str
    text: str
    attributes: Dict[str, Any]

class StatefulBrowserBackend:
    """
    有状态的浏览器后端，专为 Agent 设计。
    维护一个长期的 Page 实例，支持多步交互。
    """

    def __init__(
        self,
        storage_dir: str,
        headless: bool = True,
        timeout: int = 30_000,
        max_content_length: int = 2_000_000, # 2MB limit for HTML
    ):
        self.headless = headless
        self.timeout = timeout
        self.max_content_length = max_content_length
        
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True) # 自动创建目录

        self._init_lock = Lock()
        # 生命周期对象
        self._playwright = None
        self._browser: Optional[Browser] = None
        self._context: Optional[BrowserContext] = None
        self._page: Optional[Page] = None

    # =========================
    # Lifecycle Management
    # =========================

    async def initialize(self):
        """启动浏览器实例 (只调用一次)"""
        async with self._init_lock:
            if self._browser:
                return
            
            logger.info("Launching browser...")
            self._playwright = await async_playwright().start()
            self._browser = await self._playwright.chromium.launch(
                headless=self.headless,
                args=["--no-sandbox", "--disable-setuid-sandbox"] # Docker 友好
            )
            self._context = await self._browser.new_context(
                viewport={"width": 1280, "height": 720},
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            )
            self._page = await self._context.new_page()
            logger.info("Browser ready.")

    async def close(self):
        """关闭浏览器实例"""
        if self._page:
            await self._page.close()
        if self._context:
            await self._context.close()
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()
        logger.info("Browser closed.")

    async def ensure_page(self) -> Page:
        if not self._page:
            await self.initialize()
        return self._page

    # =========================
    # Core Actions (Stateful)
    # =========================

    async def navigate(self, url: str, wait_until: str = "domcontentloaded") -> PageInfo:
        """导航到 URL (复用当前 Page)"""
        page = await self.ensure_page()
        logger.info(f"Navigating to: {url}")
        
        try:
            resp = await page.goto(url, wait_until=wait_until, timeout=self.timeout)
            return await self._capture_page_info(page, url, resp)
        except Exception as e:
            logger.exception(f"Navigation failed: {e}")
            return PageInfo(url=url, title=None, html=None, text=None, screenshot_path=None,metadata={"error": str(e)})

    async def click(self, selector: str) -> PageInfo:
        """点击元素 (在当前页面)"""
        page = await self.ensure_page()
        logger.info(f"Clicking: {selector}")
        
        try:
            await page.click(selector, timeout=5000)
            # 注意：这里可能发生页面跳转，所以 capture 里的 url 会自动更新
            await page.wait_for_load_state("networkidle", timeout=5000)
            return await self._capture_page_info(page, page.url, None)
        except Exception as e:
            # 修复：直接从 page 对象获取当前的 URL
            current_url = page.url if page else "unknown"
            return PageInfo(
                url=current_url, 
                title=None, 
                html=None, 
                text=None, 
                screenshot_path=None,
                metadata={"error": f"Click failed: {str(e)}"}
            )

    async def fill(self, selector: str, text: str) -> PageInfo:
        """填充输入框"""
        page = await self.ensure_page()
        logger.info(f"Filling {selector} with '{text}'")
        
        try:
            await page.fill(selector, text, timeout=5000)
            return await self._capture_page_info(page, page.url, None)
        except Exception as e:
            # 修复：同上
            current_url = page.url if page else "unknown"
            return PageInfo(
                url=current_url, 
                title=None, 
                html=None, 
                text=None, 
                screenshot_path=None,
                metadata={"error": f"Fill failed: {str(e)}"}
            )
    async def get_screenshot(self) -> str:
        """获取当前页面截图，保存为文件并返回绝对路径"""
        page = await self.ensure_page()
        try:
            # 生成唯一文件名
            filename = f"screenshot_{uuid.uuid4().hex[:8]}.png"
            file_path = self.storage_dir / filename
            
            # Playwright 直接保存到路径
            await page.screenshot(path=str(file_path), full_page=False)
            
            logger.info(f"Screenshot saved to: {file_path}")
            return str(file_path.absolute())
        except Exception as e:
            logger.exception("Screenshot failed")
            return ""

    async def find_elements(self, selector: str) -> List[QueryMatch]:
        """在当前页面查找元素"""
        page = await self.ensure_page()
        try:
            elements = await page.query_selector_all(selector)
            results = []
            for el in elements:
                text = await el.inner_text()
                attrs = await el.evaluate("""el => {
                    const obj = {};
                    for (const attr of el.attributes) obj[attr.name] = attr.value;
                    return obj;
                }""")
                results.append(QueryMatch(selector=selector, text=text, attributes=attrs))
            return results
        except Exception:
            return []

    # =========================
    # Helpers
    # =========================
    async def _capture_page_info(self, page: Page, url: str, resp) -> PageInfo:
        """
        捕获当前页面状态，并将截图持久化到文件系统。
        """
        try:
            # 1. 提取基础文本内容
            html = await page.content()
            if len(html) > self.max_content_length:
                html = html[:self.max_content_length] + "\n... [TRUNCATED]"
            
            page_text = await page.inner_text("body")
            page_title = await page.title()

            # 2. 生成唯一的截图文件路径
            # 假设 self.storage_dir 已经在 __init__ 中初始化为 Path 对象
            file_name = f"page_{uuid.uuid4().hex[:10]}.png"
            file_path = self.storage_dir / file_name
            
            # 3. 执行截图并保存到磁盘
            await page.screenshot(path=str(file_path), full_page=False)

            # 4. 返回包含物理路径的 PageInfo
            return PageInfo(
                url=page.url,
                title=page_title,
                html=html,
                text=page_text,
                screenshot_path=str(file_path.absolute()), # 修复：存储绝对路径
                metadata={
                    "status": resp.status if resp else 200,
                    "content_length": len(html),
                    "has_screenshot": True
                }
            )
        except Exception as e:
            logger.error(f"Failed to capture page info for {url}: {e}")
            return PageInfo(
                url=url, 
                title=None, 
                html=None, 
                text=None, 
                screenshot_path=None, 
                metadata={"error": str(e)}
            )