import base64
import uuid
import logging
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from asyncio import Lock
from pathlib import Path
from playwright.async_api import (
    async_playwright,
    Browser,
    BrowserContext,
    Page,
    TimeoutError as PlaywrightTimeoutError,
)

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
    原子化浏览器操作封装，用于 LLM Agent 模拟人类操作网页。
    特性：
    - 原子操作接口：navigate, click, click_by_text, fill, fill_by_label, find_elements, get_screenshot
    - 智能截图（OCR）
    - 人类行为模拟（逐字输入、随机延迟、滚动）
    - 操作历史记录
    - 自动重试机制
    """

    def __init__(
        self,
        storage_dir: str,
        headless: bool = True,
        timeout: int = 30_000,
        max_content_length: int = 2_000_000,
        max_retry: int = 2,
    ):
        self.headless = headless
        self.timeout = timeout
        self.max_content_length = max_content_length
        self.max_retry = max_retry

        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self._init_lock = Lock()
        self._playwright: Optional[async_playwright] = None
        self._browser: Optional[Browser] = None
        self._context: Optional[BrowserContext] = None
        self._page: Optional[Page] = None
        self._history: List[Dict[str, Any]] = []

        # 截图触发条件
        self.min_text_length = 50
        self.min_html_length = 100
        self.ocr_keywords = ["captcha", "验证", "blocked"]

    # =========================
    # Lifecycle
    # =========================
    async def initialize(self):
        async with self._init_lock:
            if self._browser:
                return
            logger.info("Launching browser...")
            self._playwright = await async_playwright().start()
            self._browser = await self._playwright.chromium.launch(
                headless=self.headless,
                args=["--no-sandbox", "--disable-setuid-sandbox"],
            )
            self._context = await self._browser.new_context(
                viewport={"width": 1280, "height": 720},
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                ),
            )
            self._page = await self._context.new_page()
            logger.info("Browser ready.")

    async def close(self):
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
    # 原子操作接口
    # =========================
    async def navigate(self, url: str, wait_until: str = "domcontentloaded") -> PageInfo:
        """导航到网页"""
        page = await self.ensure_page()
        logger.info(f"Navigating to: {url}")
        for attempt in range(self.max_retry + 1):
            try:
                resp = await page.goto(url, wait_until=wait_until, timeout=self.timeout)
                await self._smart_wait(page)
                return await self._capture_page_info(page, url, resp)
            except PlaywrightTimeoutError:
                logger.warning(f"Navigation timeout, retry {attempt}/{self.max_retry}")
            except Exception as e:
                logger.exception(f"Navigation failed: {e}")
                return PageInfo(url=url, title=None, html=None, text=None, screenshot_path=None, metadata={"error": str(e)})
        return PageInfo(url=url, title=None, html=None, text=None, screenshot_path=None, metadata={"error": "Max retries exceeded"})

    async def click(self, selector: str) -> PageInfo:
        """点击指定选择器，并模拟人类行为"""
        page = await self.ensure_page()
        logger.info(f"Clicking selector: {selector}")
        for attempt in range(self.max_retry + 1):
            try:
                await self._scroll_into_view(page, selector)
                await self._human_delay(100, 400)
                await page.click(selector, timeout=5000)
                await self._smart_wait(page)
                self._history.append({"action": "click", "selector": selector})
                return await self._capture_page_info(page, page.url, None)
            except PlaywrightTimeoutError:
                logger.warning(f"Click timeout, retry {attempt}/{self.max_retry}")
            except Exception as e:
                logger.error(f"Click failed: {e}")
                return PageInfo(url=page.url, title=None, html=None, text=None, screenshot_path=None, metadata={"error": str(e)})
        return PageInfo(url=page.url, title=None, html=None, text=None, screenshot_path=None, metadata={"error": "Max retries exceeded"})

    async def click_by_text(self, text: str) -> PageInfo:
        """通过文本点击元素，并模拟人类行为"""
        page = await self.ensure_page()
        try:
            elements = await page.query_selector_all(f"text={text}")
            if not elements:
                return PageInfo(url=page.url, title=None, html=None, text=None, screenshot_path=None,
                                metadata={"error": f"No element with text '{text}'"})
            await elements[0].scroll_into_view_if_needed()
            await self._human_delay(100, 400)
            await elements[0].click()
            await self._smart_wait(page)
            self._history.append({"action": "click_by_text", "text": text})
            return await self._capture_page_info(page, page.url, None)
        except Exception as e:
            logger.error(f"click_by_text failed: {e}")
            return PageInfo(url=page.url, title=None, html=None, text=None, screenshot_path=None,
                            metadata={"error": str(e)})

    async def fill(self, selector: str, value: str) -> PageInfo:
        """普通填充输入框"""
        page = await self.ensure_page()
        logger.info(f"Filling selector {selector} with '{value}'")
        for attempt in range(self.max_retry + 1):
            try:
                await self._scroll_into_view(page, selector)
                await page.fill(selector, value, timeout=5000)
                self._history.append({"action": "fill", "selector": selector, "value": value})
                await self._smart_wait(page)
                return await self._capture_page_info(page, page.url, None)
            except PlaywrightTimeoutError:
                logger.warning(f"Fill timeout, retry {attempt}/{self.max_retry}")
            except Exception as e:
                logger.error(f"Fill failed: {e}")
                return PageInfo(url=page.url, title=None, html=None, text=None, screenshot_path=None, metadata={"error": str(e)})
        return PageInfo(url=page.url, title=None, html=None, text=None, screenshot_path=None, metadata={"error": "Max retries exceeded"})

    async def fill_by_label(self, label_text: str, value: str) -> PageInfo:
        """通过 label 填充输入框"""
        page = await self.ensure_page()
        try:
            el = await page.query_selector(f"label:has-text('{label_text}') >> input")
            if not el:
                return PageInfo(url=page.url, title=None, html=None, text=None, screenshot_path=None,
                                metadata={"error": f"No input for label '{label_text}'"})
            await el.scroll_into_view_if_needed()
            await el.fill(value)
            self._history.append({"action": "fill_by_label", "label": label_text, "value": value})
            await self._smart_wait(page)
            return await self._capture_page_info(page, page.url, None)
        except Exception as e:
            logger.error(f"fill_by_label failed: {e}")
            return PageInfo(url=page.url, title=None, html=None, text=None, screenshot_path=None,
                            metadata={"error": str(e)})

    async def fill_human_like(self, selector: str, value: str) -> PageInfo:
        """
        模拟逐字输入，降低反爬风险
        """
        page = await self.ensure_page()
        logger.info(f"Human-like filling selector {selector} with '{value}'")
        try:
            await self._scroll_into_view(page, selector)
            await page.focus(selector)
            for char in value:
                await page.keyboard.type(char, delay=random.randint(50, 150))
            await self._human_delay()
            self._history.append({"action": "fill_human_like", "selector": selector, "value": value})
            await self._smart_wait(page)
            return await self._capture_page_info(page, page.url, None)
        except Exception as e:
            logger.error(f"fill_human_like failed: {e}")
            return PageInfo(url=page.url, title=None, html=None, text=None, screenshot_path=None,
                            metadata={"error": str(e)})

    async def find_elements(self, selector: str) -> List[QueryMatch]:
        """查询元素内容"""
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

    async def get_screenshot(self) -> str:
        """原子截图接口"""
        page = await self.ensure_page()
        try:
            filename = f"screenshot_{uuid.uuid4().hex[:8]}.png"
            file_path = self.storage_dir / filename
            await page.screenshot(path=str(file_path), full_page=False)
            logger.info(f"Screenshot saved to: {file_path}")
            return str(file_path.absolute())
        except Exception as e:
            logger.exception("Screenshot failed")
            return ""

    # =========================
    # Helpers
    # =========================
    async def _capture_page_info(self, page: Page, url: str, resp) -> PageInfo:
        """原子抓取页面信息 + 智能截图"""
        try:
            html = await page.content()
            if len(html) > self.max_content_length:
                html = html[:self.max_content_length] + "\n... [TRUNCATED]"
            page_text = await page.inner_text("body")
            page_title = await page.title()

            # 智能截图触发条件
            take_screenshot = (
                len(page_text.strip()) < self.min_text_length
                or len(html.strip()) < self.min_html_length
                or (resp and resp.status != 200)
                or any(keyword in page_text.lower() for keyword in self.ocr_keywords)
            )

            screenshot_path = None
            if take_screenshot:
                file_name = f"page_{uuid.uuid4().hex[:10]}.png"
                file_path = self.storage_dir / file_name
                await page.screenshot(path=str(file_path), full_page=False)
                screenshot_path = str(file_path.absolute())

            return PageInfo(
                url=page.url,
                title=page_title,
                html=html,
                text=page_text,
                screenshot_path=screenshot_path,
                metadata={
                    "status": resp.status if resp else 200,
                    "content_length": len(html),
                    "has_screenshot": screenshot_path is not None
                }
            )
        except Exception as e:
            logger.error(f"Failed to capture page info for {url}: {e}")
            return PageInfo(url=url, title=None, html=None, text=None, screenshot_path=None, metadata={"error": str(e)})

    async def _smart_wait(self, page: Page, delay: int = 300):
        """智能等待，适配 SPA / JS 页面"""
        try:
            await page.wait_for_load_state("networkidle", timeout=5000)
        except PlaywrightTimeoutError:
            pass
        await page.wait_for_timeout(delay)

    async def _scroll_into_view(self, page: Page, selector: str):
        """滚动元素到可视区"""
        try:
            el = await page.query_selector(selector)
            if el:
                await el.scroll_into_view_if_needed(timeout=2000)
        except Exception:
            pass

    async def _human_delay(self, min_ms: int = 200, max_ms: int = 800):
        """模拟人类思考的随机延迟"""
        delay = random.randint(min_ms, max_ms)
        page = await self.ensure_page()
        await page.wait_for_timeout(delay)