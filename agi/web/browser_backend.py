import base64
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from playwright.async_api import async_playwright, Browser, BrowserContext, Page

logger = logging.getLogger(__name__)

@dataclass
class PageInfo:
    url: str
    title: Optional[str]
    html: Optional[str]
    text: Optional[str]
    screenshot_base64: Optional[str]
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
        headless: bool = True,
        timeout: int = 30_000,
        max_content_length: int = 2_000_000, # 2MB limit for HTML
    ):
        self.headless = headless
        self.timeout = timeout
        self.max_content_length = max_content_length
        
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
            return PageInfo(url=url, title=None, html=None, text=None, metadata={"error": str(e)})

    async def click(self, selector: str) -> PageInfo:
        """点击元素 (在当前页面)"""
        page = await self.ensure_page()
        logger.info(f"Clicking: {selector}")
        
        try:
            await page.click(selector, timeout=5000)
            await page.wait_for_load_state("networkidle", timeout=5000) # 等待跳转或加载
            return await self._capture_page_info(page, page.url, None)
        except Exception as e:
            return PageInfo(url=page.url, title=None, html=None, text=None, metadata={"error": f"Click failed: {e}"})

    async def fill(self, selector: str, text: str) -> PageInfo:
        """填充输入框"""
        page = await self.ensure_page()
        logger.info(f"Filling {selector} with '{text}'")
        
        try:
            await page.fill(selector, text, timeout=5000)
            return await self._capture_page_info(page, page.url, None)
        except Exception as e:
            return PageInfo(url=page.url, title=None, html=None, text=None, metadata={"error": f"Fill failed: {e}"})

    async def get_screenshot(self) -> str:
        """获取当前页面截图 (Base64)"""
        page = await self.ensure_page()
        try:
            img = await page.screenshot(full_page=True, type="png")
            return base64.b64encode(img).decode("utf-8")
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
        """捕获当前页面状态"""
        try:
            html = await page.content()
            if len(html) > self.max_content_length:
                html = html[:self.max_content_length] + "\n... [TRUNCATED]"
            
            text = await page.inner_text("body")
            title = await page.title()
            
            # 截图 (可选，如果太慢可以改为按需调用)
            # 为了节省带宽，这里不默认生成截图，由 middleware 的 screenshot 工具单独调用
            # 但如果需要调试，可以开启
            # img = await page.screenshot() 
            # b64 = base64.b64encode(img).decode()
            
            return PageInfo(
                url=page.url,
                title=title,
                html=html,
                text=text,
                screenshot_base64=None, 
                metadata={
                    "status": resp.status if resp else None,
                    "timestamp": str(page.context.browser.version)
                }
            )
        except Exception as e:
            return PageInfo(url=url, title=None, html=None, text=None, metadata={"error": str(e)})