import asyncio
import random
import traceback
from typing import List, Optional, Union, Any, Type
import requests
from bs4 import BeautifulSoup
from pydantic import Field, BaseModel
from langchain_core.tools import BaseTool
from langchain_core.documents import Document
from agi.config import log


class WLInput(BaseModel):
    web_paths: str = Field(description="urls need to be scraped")


class WebScraper(BaseTool):
    name: str = "web_scraper"
    description: str = "Simple web scraper"
    args_schema: Type[BaseModel] = WLInput

    web_paths: Optional[List[str]] = None

    def __init__(self, web_paths: Optional[List[str]] = None, **kwargs):
        super().__init__(**kwargs)
        self.web_paths = list(web_paths) if web_paths else []

    # ------------------- 同步接口 -------------------
    def load(self) -> List[Document]:
        """同步入口，保留 load 和 _run"""
        return self._run(self.web_paths)

    def _run(self, web_paths: Union[str, List[str], dict], run_manager: Optional[Any] = None) -> List[Document]:
        """同步主入口：接受 str / list / dict"""
        if isinstance(web_paths, str):
            web_paths = [web_paths]
        elif isinstance(web_paths, dict):
            web_paths = web_paths.get("urls") or web_paths.get("url") or []

        docs: List[Document] = []

        # 使用异步函数并阻塞获取结果
        async def gather_docs():
            tasks = [self._fetch_and_parse(url) for url in web_paths]
            return await asyncio.gather(*tasks, return_exceptions=True)

        try:
            loop = asyncio.get_running_loop()
            results = loop.run_until_complete(gather_docs())
        except RuntimeError:
            # 没有运行的 loop
            results = asyncio.run(gather_docs())

        for res in results:
            if isinstance(res, Exception):
                log.error("Error processing url: %s", res)
                continue
            if res:
                docs.append(res)
        return docs

    # ------------------- 异步入口 -------------------
    async def aload(self, web_paths: Optional[List[str]] = None) -> List[Document]:
        """异步入口，可并行抓取多个 URL"""
        web_paths = web_paths or self.web_paths
        if isinstance(web_paths, str):
            web_paths = [web_paths]
        elif isinstance(web_paths, dict):
            web_paths = web_paths.get("urls") or web_paths.get("url") or []

        tasks = [self._fetch_and_parse(url) for url in web_paths]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        docs: List[Document] = []
        for res in results:
            if isinstance(res, Exception):
                log.error("Error processing url: %s", res)
                continue
            if res:
                docs.append(res)
        return docs

    # ------------------- 内部抓取和解析 -------------------
    async def _fetch_and_parse(self, url: str) -> Optional[Document]:
        async def try_fetch(fetch_func):
            try:
                html = await fetch_func(url)
                if not html:
                    return None
                doc = self._parse_local(html, url)
                if not doc or len(doc.page_content.strip()) < 50:
                    return None
                return doc, html
            except Exception as e:
                log.warning("Fetch failed for %s: %s", url, e)
                return None

        # 首先尝试 requests（或 _fetch 内封装）
        result = await try_fetch(self._fetch)
        
        # 如果无效且允许 Playwright fallback
        if not result:
            log.warning("Fallback to Playwright for %s", url)
            result = await try_fetch(self._fetch_playwright)

        if not result:
            return None

        doc, html = result
        return doc



    async def _fetch(self, url: str) -> Optional[str]:
        """优先用 requests，同步方式；失败 fallback Playwright"""
        headers = {"User-Agent": self._random_ua(), "Accept": "text/html"}
        try:
            resp = await asyncio.to_thread(requests.get, url, headers=headers, timeout=15)
            resp.raise_for_status()
            return resp.text
        except Exception as e:
            log.warning("Requests fetch failed for %s: %s", url, e)
            return None

    async def _fetch_playwright(self, url: str) -> str:
        """异步 Playwright 抓取"""
        from playwright.async_api import async_playwright
        from playwright_stealth import stealth

        async with async_playwright() as p:
            browser = await p.chromium.launch(
                headless=True,
                args=["--disable-blink-features=AutomationControlled"]
            )
            page = await browser.new_page()
            await stealth(page)
            await page.goto(url, wait_until="domcontentloaded", timeout=20000)
            await page.wait_for_selector("#js_content", timeout=10000)
            await asyncio.sleep(random.uniform(0.5, 1.5))
            html = await page.content()
            await browser.close()
            return html

    # ------------------- 工具函数 -------------------
    def _random_ua(self) -> str:
        choices = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.1 Safari/605.1.15",
            "Mozilla/5.0 (Linux; Android 13; Pixel 6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0 Mobile Safari/537.36",
        ]
        return random.choice(choices)

    def _parse_local(self, html: str, source: str) -> Optional[Document]:
        soup = BeautifulSoup(html, "lxml")
        node = soup.find("article") or soup.find(role="main") or soup.find("main") or soup.body or soup
        for tag in node(["script", "style", "nav", "footer", "aside", "form", "iframe"]):
            tag.decompose()
        text = node.get_text("\n", strip=True)
        clean_lines = [ln.strip() for ln in text.splitlines() if len(ln.strip()) > 40 and not self._is_noise(ln)]
        content = "\n".join(clean_lines).strip()
        if not content:
            return None
        meta = {
            "source": source,
            "title": self._safe_text(soup.find("title")),
            "time": self._safe_text(soup.find("time")) or "",
            "author": self._safe_text(soup.find(attrs={"class": lambda v: v and "author" in v.lower()})) or "",
            "likes": 0,
            "comments": 0,
        }
        return Document(page_content=content, metadata=meta)

    def _is_noise(self, line: str) -> bool:
        import re
        return bool(re.search(r"广告|推荐|相关阅读|copyright", line, re.IGNORECASE))

    def _safe_text(self, tag):
        if not tag:
            return ""
        return tag.get_text(strip=True) if getattr(tag, "get_text", None) else (tag.get("content") or "")

