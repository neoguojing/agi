import asyncio
import random
import traceback
import re
from typing import List, Optional, Union, Any, Type, Dict

import requests
from bs4 import BeautifulSoup
from pydantic import Field, BaseModel
from langchain_core.tools import BaseTool
from langchain_core.documents import Document
from playwright.async_api import async_playwright, Browser, BrowserContext

from agi.config import log  # 假设你的配置路径

# --- 常量定义 ---
USER_AGENTS = [
    {"ua": "Mozilla/5.0 (iPhone; CPU iPhone OS 15_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.0 Mobile/15E148 Safari/604.1", "viewport": {"width": 375, "height": 812}},
    {"ua": "Mozilla/5.0 (Linux; Android 12; HUAWEI P50) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Mobile Safari/537.36", "viewport": {"width": 390, "height": 844}},
    {"ua": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36", "viewport": {"width": 1920, "height": 1080}},
]

CONTENT_SELECTORS = [
    "#js_content", ".rich_media_content", "article", "main", ".content", 
    ".article-content", ".post-content", ".entry-content", ".post-body",
    ".article-body", "section", "div.article", "div.main-content"
]

class WLInput(BaseModel):
    web_paths: Union[str, List[str], Dict[str, Any]] = Field(description="URLs or a dict containing 'urls' key")

class WebScraper(BaseTool):
    name: str = "web_scraper"
    description: str = "Web scraper that takes one or more URLs as input and extracts web page content such as text, links, and metadata."
    args_schema: Type[BaseModel] = WLInput

    web_paths: List[str] = Field(default_factory=list)
    concurrency: int = 5  # 控制并发数

    def __init__(self, web_paths: Optional[List[str]] = None, **kwargs):
        super().__init__(**kwargs)
        if web_paths:
            self.web_paths = list(web_paths)

    # ------------------- 同步接口 (LangChain 标准) -------------------
    def _run(self, web_paths: Union[str, List[str], Dict[str, Any]], run_manager: Optional[Any] = None) -> List[Document]:
        """同步入口：通过 asyncio 桥接异步实现"""
        urls = self._normalize_urls(web_paths)
        try:
            loop = asyncio.get_running_loop()
            return loop.run_until_complete(self.aload(urls))
        except RuntimeError:
            return asyncio.run(self.aload(urls))

    # ------------------- 异步接口 -------------------
    async def aload(self, web_paths: Optional[Union[str, List[str], Dict[str, Any]]] = None) -> List[Document]:
        """通用异步加载入口"""
        urls = self._normalize_urls(web_paths or self.web_paths)
        if not urls:
            return []

        async with self._managed_browser() as browser:
            # 限制并发，防止被封 IP 或内存溢出
            sem = asyncio.Semaphore(self.concurrency)
            tasks = [self._fetch_and_parse_with_sem(url, browser, sem) for url in urls]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        docs = []
        for res in results:
            if isinstance(res, Exception):
                log.error(f"Error processing URL: {res}")
                continue
            if res:
                docs.append(res)
        return docs

    async def aload2(self, question_url_map: Dict[str, List[str]]) -> Dict[str, List[Document]]:
        """实现原有的问题-URL 映射抓取"""
        # 展平任务
        q_u_pairs = [(q, url) for q, urls in question_url_map.items() for url in urls]
        urls = [pair[1] for pair in q_u_pairs]
        
        # 抓取所有内容
        docs_list = await self.aload(urls)
        
        # 将结果映射回 question
        # 注意：aload 返回的长度可能因为失败而变短，这里需要按顺序重新匹配
        doc_map = {url: doc for doc in docs_list for url in [doc.metadata["source"]]}
        
        out: Dict[str, List[Document]] = {q: [] for q in question_url_map}
        for q, url in q_u_pairs:
            if url in doc_map:
                out[q].append(doc_map[url])
        return out

    # ------------------- 核心逻辑 -------------------
    async def _fetch_and_parse_with_sem(self, url: str, browser: Browser, sem: asyncio.Semaphore) -> Optional[Document]:
        async with sem:
            # 1. 尝试同步 Requests (高效)
            html = await self._fetch_requests(url)
            doc = self._parse_local(html, url) if html else None

            # 2. 如果 Requests 失败或内容太少 (可能是动态加载)，回退到 Playwright
            if not doc or len(doc.page_content) < 200:
                log.warning(f"Fallback to Playwright for {url}")
                html = await self._fetch_playwright(url, browser)
                doc = self._parse_local(html, url) if html else None

            return doc

    async def _fetch_requests(self, url: str) -> Optional[str]:
        ua, _ = self._random_ua()
        try:
            # 在线程池中执行同步请求，避免阻塞 event loop
            resp = await asyncio.to_thread(
                requests.get, url, 
                headers={"User-Agent": ua, "Accept": "text/html"}, 
                timeout=15
            )
            resp.raise_for_status()
            return resp.text
        except Exception as e:
            log.debug(f"Requests fetch failed for {url}: {e}")
            return None

    async def _fetch_playwright(self, url: str, browser: Browser) -> Optional[str]:
        from playwright_stealth import stealth_async
        ua, vp = self._random_ua()
        
        context = await browser.new_context(user_agent=ua, viewport=vp)
        page = await context.new_page()
        await stealth_async(page)
        
        try:
            # 增加对常见正文容器的显式等待
            await page.goto(url, wait_until="networkidle", timeout=30000)
            
            # 模拟原代码中的分批次等待逻辑
            found = False
            for sel in CONTENT_SELECTORS[:5]: # 优先检查高频标签
                try:
                    await page.wait_for_selector(sel, state="attached", timeout=5000)
                    found = True
                    break
                except: continue
            
            if not found:
                await asyncio.sleep(2) # 兜底等待
                
            return await page.content()
        except Exception as e:
            log.warning(f"Playwright failed for {url}: {e}")
            return None
        finally:
            await page.close()
            await context.close()

    # ------------------- 工具函数 -------------------
    @asynccontextmanager
    async def _managed_browser(self):
        """确保浏览器资源始终正确关闭"""
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True, args=["--disable-blink-features=AutomationControlled"])
            try:
                yield browser
            finally:
                await browser.close()

    def _normalize_urls(self, web_paths: Any) -> List[str]:
        if isinstance(web_paths, str):
            return [web_paths]
        if isinstance(web_paths, list):
            return web_paths
        if isinstance(web_paths, dict):
            return web_paths.get("urls") or web_paths.get("url") or []
        return []

    def _random_ua(self):
        choice = random.choice(USER_AGENTS)
        return choice["ua"], choice["viewport"]

    def _parse_local(self, html: str, source: str) -> Optional[Document]:
        """
        基于文本密度（Text Density）的智能解析，减少对选择器的依赖
        """
        if not html: return None
        soup = BeautifulSoup(html, "lxml")
        
        # 移除无意义标签
        for s in soup(["script", "style", "video", "audio", "input", "button", "header", "footer"]):
            s.decompose()

        # 1. 尝试元数据：增加对 OpenGraph 标签的支持 (常见于国际网站)
        title = (
            soup.find("meta", property="og:title") or 
            soup.find("h1") or 
            soup.find("title")
        )
        title_text = self._safe_text(title)

        # 2. 文本块识别：基于行长度和标点符号密度
        # 这种方法比单纯找 <article> 更能应对那些结构混乱的垃圾网页
        blocks = []
        for p in soup.find_all(['p', 'div', 'section']):
            text = p.get_text(strip=True)
            if len(text) > 30: # 过滤掉掉导航栏碎片
                # 计算密度：中文标点符号占比
                punctuation_count = len(re.findall(r'[，。！？、：]', text))
                if punctuation_count > 0 or len(text) > 100:
                    blocks.append(text)

        # 去重并合并
        unique_blocks = []
        seen = set()
        for b in blocks:
            if b not in seen:
                unique_blocks.append(b)
                seen.add(b)

        content = "\n\n".join(unique_blocks).strip()
        
        if not content:
            # 如果密度算法失败，回退到父类的基础提取逻辑
            return super()._parse_local(html, source)

        return Document(
            page_content=content,
            metadata={
                "source": source,
                "title": title_text,
                "type": "text_density_v2",
                "content_hash": hash(content[:100]) # 简单校验码
            }
        )
    
    def _is_noise(self, line: str) -> bool:
        return bool(re.search(r"广告|推荐|相关阅读|copyright|版权所有|扫码关注", line, re.IGNORECASE))

    def _safe_text(self, tag):
        if not tag: return ""
        if hasattr(tag, "get_text"):
            return tag.get_text(strip=True)
        return tag.get("content") or ""