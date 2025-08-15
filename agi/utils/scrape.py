import asyncio
import random
import traceback
from typing import List, Optional, Union, Any, Type,Dict
import requests
from bs4 import BeautifulSoup
from pydantic import Field, BaseModel
from langchain_core.tools import BaseTool
from langchain_core.documents import Document
from agi.config import log

USER_AGENTS = [
    # 苹果设备
    {
        "ua": "Mozilla/5.0 (iPhone; CPU iPhone OS 15_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.0 Mobile/15E148 Safari/604.1",
        "viewport": {"width": 375, "height": 812}
    },
    {
        "ua": "Mozilla/5.0 (iPhone; CPU iPhone OS 16_4 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.4 Mobile/15E148 Safari/604.1",
        "viewport": {"width": 390, "height": 844}
    },
    {
        "ua": "Mozilla/5.0 (iPad; CPU OS 16_4 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.4 Mobile/15E148 Safari/604.1",
        "viewport": {"width": 768, "height": 1024}
    },


    # 安卓国产设备
    {
        "ua": "Mozilla/5.0 (Linux; Android 12; HUAWEI P50) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Mobile Safari/537.36",
        "viewport": {"width": 390, "height": 844}
    },
    {
        "ua": "Mozilla/5.0 (Linux; Android 13; MI 13) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Mobile Safari/537.36",
        "viewport": {"width": 430, "height": 932}
    },
    {
        "ua": "Mozilla/5.0 (Linux; Android 12; OPPO Find X5 Pro) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Mobile Safari/537.36",
        "viewport": {"width": 412, "height": 915}
    },
    {
        "ua": "Mozilla/5.0 (Linux; Android 12; VIVO X80) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Mobile Safari/537.36",
        "viewport": {"width": 430, "height": 932}
    },
]

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
    
    async def aload2(self, question_url_map: Optional[Dict[str, list[str]]] = None) -> Dict[str, list[Document]]:
        """异步入口，可并行抓取多个 URL，输入是 {question: [url1, url2, ...]}"""

        # 构建 (question, url) 对应列表
        q_u_pairs = [(q, url) for q, urls in question_url_map.items() for url in urls]
        results = await asyncio.gather(*(self._fetch_and_parse(url) for _, url in q_u_pairs), return_exceptions=True)

        out: Dict[str, list[Document]] = {q: [] for q in question_url_map}
        for (q, _), res in zip(q_u_pairs, results):
            if not isinstance(res, Exception) and res is not None:
                out[q].append(res)

        return out



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
                traceback.print_exc()
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
        ua,_ = self._random_ua()
        headers = {"User-Agent": ua, "Accept": "text/html"}
        try:
            resp = await asyncio.to_thread(requests.get, url, headers=headers, timeout=15)
            resp.raise_for_status()
            return resp.text
        except Exception as e:
            log.warning("Requests fetch failed for %s: %s", url, e)
            return None

    async def _fetch_playwright(self, url: str) -> str:
        """异步 Playwright 抓取，优先获取页面内容，正文未出现时分批等待标签"""
        from playwright.async_api import async_playwright
        from playwright_stealth import Stealth

        stealth = Stealth()

        # 全局 selector 列表，中文 + 国际主流网站 + 通用标签
        selectors = [
            "#js_content", ".rich_media_content", "article", "main",
            ".content", ".article-content", ".post-content", ".entry-content",
            ".news-content", ".blog-post-content", ".post-body", ".post-body-content",
            ".article-body", ".article-body__content", ".story-body", ".story-content",
            ".entry-body", ".c-article-body", ".blog-content", ".post-text",
            ".post-article", ".content-body", ".page-content", ".post-entry",
            ".news-article-content", ".content-article", "section", "div.article",
            "div.content", "div.main-content", "div.post"
        ]

        # 高优先级和低优先级分组
        high_priority = selectors[:10]  # 最可能匹配正文的前 10 个
        low_priority = selectors[10:]

        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(
                    headless=True,
                    args=["--disable-blink-features=AutomationControlled"]
                )
                
                ua,vp = self._random_ua()
                context = await browser.new_context(
                    user_agent=ua,
                    viewport=vp,
                )

                await stealth.apply_stealth_async(context)

                page = await context.new_page()
                await page.goto(url, wait_until="networkidle", timeout=30000)

                # 先获取页面内容
                html = await page.content()

                # 判断是否有正文（简单文本长度 + 标签匹配）
                def has_main_content(content: str) -> bool:
                    if len(content.strip()) < 200:  # 文本太短可能无正文
                        return False
                    return any(sel.strip("#.") in content for sel in selectors[:10])

                if not has_main_content(html):
                    # 高优先级并行等待
                    tasks = [page.wait_for_selector(sel, state="attached", timeout=10000) for sel in high_priority]
                    done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                    for t in pending:
                        t.cancel()

                    if done:
                        try:
                            await list(done)[0]
                            html = await page.content()
                        except Exception as e:
                            log.warning(f"High-priority selector wait failed for {url}: {e}")
                    else:
                        # 低优先级顺序等待
                        for sel in low_priority:
                            try:
                                await page.wait_for_selector(sel, state="attached", timeout=5000)
                                html = await page.content()
                                break
                            except Exception:
                                continue
                        else:
                            log.warning(f"No selectors matched for {url}, returning raw page content")

                # 模拟人类浏览延迟
                await asyncio.sleep(random.uniform(0.5, 1.5))

                await browser.close()
                return html

        except Exception as e:
            traceback.print_exc()
            log.warning("_fetch_playwright failed for %s: %s", url, e)
            return None
        
    # ------------------- 工具函数 -------------------
    def _random_ua(self):
        """
        随机返回 User-Agent 和对应视口大小
        """
        choice = random.choice(USER_AGENTS)
        return choice["ua"], choice["viewport"]

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

