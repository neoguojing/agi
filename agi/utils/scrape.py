import time
import random
import traceback
from typing import List, Optional, Union, Any, Type
import requests
from bs4 import BeautifulSoup
from pydantic import Field, BaseModel
from langchain_core.tools import BaseTool
from langchain_core.documents import Document
from agi.config import log

# 非必须：如果要用 LLM pipeline，请传入 llm 且它支持 .invoke({"text": ...})
# 如果不传 llm，则使用本地解析结果。
class WLInput(BaseModel):
    web_paths: str = Field(description="urls need to be scraped")


class WebScraper(BaseTool):
    name: str = "web_scraper"                          # <- 加上类型注解
    description: str = "Simple web scraper"            # <- 加上类型注解
    args_schema: Type[BaseModel] = WLInput

     # **关键**：加上类型注解，使其成为 Pydantic 字段
    web_paths: Optional[List[str]] = None
    chain: Optional[Any] = None
    use_playwright: bool = False
    headless: bool = True

    def __init__(self, web_paths: Optional[List[str]] = None, llm: Any = None, use_playwright: bool = False, headless: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.web_paths = list(web_paths) if web_paths else []
        self.chain = llm  # optional LLM pipeline (callable or object with .invoke)
        self.use_playwright = use_playwright
        self.headless = headless

    # 保留接口
    def load(self) -> List[Document]:
        return self._run(self.web_paths)

    def _run(self, web_paths: Union[str, List[str], dict], run_manager: Optional[Any] = None) -> List[Document]:
        """同步主入口：接受 str / list / dict（兼容旧接口）"""
        docs: List[Document] = []

        if isinstance(web_paths, str):
            web_paths = [web_paths]
        elif isinstance(web_paths, dict):
            web_paths = web_paths.get("urls") or web_paths.get("url") or []

        for url in web_paths:
            try:
                html = self._fetch(url)
                if not html:
                    log.warning("Empty response for %s", url)
                    continue

                # 优先使用链（若提供），否则本地解析
                if self.chain:
                    doc = self._parse_with_llm(html, url)
                else:
                    doc = self._parse_local(html, url)

                if doc:
                    docs.append(doc)

            except Exception:
                log.error("Error processing %s", url)
                log.error(traceback.format_exc())
                # 继续下一个
        return docs

    # ---- 简单抓取（requests）；如需动态页面可启用 Playwright（开关 use_playwright） ----
    def _fetch(self, url: str) -> Optional[str]:
        if self.use_playwright:
            try:
                return self._fetch_playwright(url)
            except Exception as e:
                log.warning("Playwright fetch failed, fallback to requests: %s", e)

        headers = {"User-Agent": self._random_ua(), "Accept": "text/html"}
        try:
            resp = requests.get(url, headers=headers, timeout=15)
            resp.raise_for_status()
            return resp.text
        except Exception as e:
            log.error("requests fetch failed for %s: %s", url, e)
            return None

    def _fetch_playwright(self, url: str) -> str:
        # 轻量版 playwright 抓取（如果未安装或失败会抛异常）
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=self.headless)
            page = browser.new_page()
            # 少量防检测脚本
            page.add_init_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined});")
            page.goto(url, wait_until="networkidle", timeout=20000)
            time.sleep(random.uniform(0.2, 0.8))
            html = page.content()
            browser.close()
            return html

    def _random_ua(self) -> str:
        choices = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.1 Safari/605.1.15",
            "Mozilla/5.0 (Linux; Android 13; Pixel 6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0 Mobile Safari/537.36",
        ]
        return random.choice(choices)

    # ---- 本地解析（轻量） ----
    def _parse_local(self, html: str, source: str) -> Optional[Document]:
        soup = BeautifulSoup(html, "lxml")

        # 提取正文：优先 article/main，其次 body
        node = soup.find("article") or soup.find(role="main") or soup.find("main") or soup.body or soup
        # 去掉脚本样式等
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
        return bool(re_search := __import__("re").search(r"广告|推荐|相关阅读|copyright", line, __import__("re").IGNORECASE))

    def _safe_text(self, tag):
        if not tag:
            return ""
        return tag.get_text(strip=True) if getattr(tag, "get_text", None) else (tag.get("content") or "")

    # ---- 使用 LLM（若提供）解析并返回 Document，失败回退本地解析 ----
    def _parse_with_llm(self, html: str, source: str) -> Optional[Document]:
        # 先做本地抽取作为输入压缩（避免把整页 HTML 送入 LLM）
        summary_input = self._compress_for_llm(html)
        try:
            # 支持两种调用方式：chain.invoke({"text": ...}) 或可调用对象 chain(summary_input)
            if hasattr(self.chain, "invoke"):
                result = self.chain.invoke(summary_input)
            else:
                result = self.chain(summary_input)  # type: ignore
        except Exception as e:
            log.error("LLM invocation failed: %s", e)
            return self._parse_local(html, source)

        # 期望 result 是 dict-like，包含 'content'
        if not result or not isinstance(result, dict):
            return self._parse_local(html, source)

        content = result.get("content") or ""
        if not content:
            return self._parse_local(html, source)

        metadata = {
            "source": source,
            "title": result.get("title", "") or "",
            "time": result.get("time", "") or "",
            "author": result.get("author", "") or "",
            "likes": int(result.get("likes") or 0),
            "comments": int(result.get("comments") or 0),
        }
        return Document(page_content=content, metadata=metadata)

    def _compress_for_llm(self, html: str) -> str:
        """把 HTML 压缩成较短的纯文本给 LLM（去脚本与多余空白）"""
        soup = BeautifulSoup(html, "lxml")
        for tag in soup(["script", "style", "nav", "footer", "aside", "iframe"]):
            tag.decompose()
        text = soup.get_text("\n", strip=True)
        # 取前 2000 字左右作为 LLM 输入（可按需调整）
        return text[:2000]

# End of WebScraper
