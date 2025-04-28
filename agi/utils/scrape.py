import requests
import re
import aiohttp
import asyncio
import traceback
from bs4 import BeautifulSoup
from typing import Iterator, List, Optional
from requests.exceptions import RequestException
import random
import time
from typing import Any, Optional, Type,List
from playwright.sync_api import sync_playwright
from playwright.async_api import async_playwright
from pydantic import Field,BaseModel,PrivateAttr
from langchain_core.tools import BaseTool
from agi.config import log
from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.documents import Document

class WLInput(BaseModel):
    """Input for the web loader tool."""

    urls: List[str] = Field(description="urls need to be scraped")

class WebScraper(BaseTool):
    name: str = "web scraper"
    description: str = (
        "This web scraper tool takes a list of URLs as input and retrieves the content from each webpage."
    )

    args_schema: Type[BaseModel] = WLInput
    use_selenium: bool = False
    web_paths: List[str] = None

    def __init__(self,web_paths:List[str] = None, use_selenium: bool = False,**kwargs):
        super().__init__(**kwargs)
        self.use_selenium = use_selenium  # Flag to use Selenium for dynamic pages
        self.web_paths = web_paths

    def _scrape(self, url: str) -> BeautifulSoup:
        """Scrape content from URL, with option for dynamic content loading via Selenium."""
        if self.use_selenium or "toutiao" in url:
            return self._scrape_dynamic_sync(url)
        else:
            return self._scrape_with_requests(url)

    def _scrape_with_requests(self, url: str) -> BeautifulSoup:
        """Use requests and BeautifulSoup to scrape static content."""
        try:
            headers = self._get_random_headers()
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, "html.parser")
            return soup
        except RequestException as e:
            log.error(f"Error fetching {url}: {e}")
            raise

    async def _scrape_with_aiohttp(self, url: str) -> BeautifulSoup:
        """Use requests and BeautifulSoup to scrape static content."""
        try:
            headers = self._get_random_headers()
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, timeout=10) as response:
                    response.raise_for_status()
                    html = await response.text()
                    # Pass bs_kwargs to BeautifulSoup to customize parsing
                    soup = BeautifulSoup(html, "html.parser")
                    return soup
        except Exception as e:
            log.error(f"Error fetching {url} with aiohttp: {e}")
            raise

    def _scrape_dynamic_sync(self, url: str) -> BeautifulSoup:
        """Use Playwright for scraping dynamic content synchronously."""
        with sync_playwright() as p:
            # 启动 Chromium 浏览器
            browser = p.chromium.launch(headless=True)  # headless=True 表示无头模式
            page = browser.new_page()
            try:
                page.goto(url, wait_until="networkidle", timeout=15000)
                page.wait_for_selector("body", timeout=5000)  # 额外保险
                html_content = page.content()
                # 用 BeautifulSoup 解析 HTML
                soup = BeautifulSoup(html_content, "html.parser")
                return soup
            finally:
                browser.close()

    async def _scrape_dynamic_async(self, url: str) -> BeautifulSoup:
        """Use Selenium for scraping dynamic content."""
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)  # headless=True 表示无头模式
            page = await browser.new_page()

            # 打开网页并获取页面的 HTML 内容
            await page.goto(url)
            html_content = await page.content()  # 获取完整的 HTML 内容，而不是 inner_html('body')

            # 关闭浏览器
            await browser.close()

            # 用 BeautifulSoup 解析 HTML
            soup = BeautifulSoup(html_content, "html.parser")
            return soup

    def _get_random_headers(self) -> dict:
        """Generate random headers to mimic real user requests."""
        user_agents = [
            # Example user agents, can be expanded
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:76.0) Gecko/20100101 Firefox/76.0",
            "Mozilla/5.0 (Windows NT 6.1; WOW64; rv:34.0) Gecko/20100101 Firefox/34.0"
        ]
        return {
            "User-Agent": random.choice(user_agents),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
        }

    def _get_titile(self,soup):
        title = soup.find('h1') or soup.find('meta', {'name': 'title'}) or soup.find('div', class_='question-title')
        return {"title":title.get_text(strip=True) if title else ""}

    def _get_pub_date(self,soup):
        pub_date = soup.find('time') or soup.find('span', class_='publish-time')
        return {"time":pub_date.get_text(strip=True) if pub_date else ""}

    def _get_author(self,soup):
         # 提取作者
        author = soup.find('span', class_='author') or soup.find('meta', {'name': 'author'})
        return {"author":author.get_text(strip=True) if author else ""}
    
    def _get_content(self,soup):
        search_patterns = [
            ('article', None),
            ('div', 'article-content'),
            ('div', 'answer-content'),
            ('div', 'answer-body'),
            ('div','post-content'),
            ('div','.*content.*')
        ]

        content = ""  # 默认值为空字符串

        # 遍历所有的搜索模式
        for tag, cls in search_patterns:
            if cls is None:
                # 如果类名为 None，则只根据标签名查找
                content = soup.find(tag)
            else:
                # 如果类名不为 None，则使用正则表达式匹配类名
                content = soup.find(tag, class_=re.compile(cls)) or soup.find(tag, id=re.compile('.*content.*'))
            
            # 如果找到了匹配的元素，停止查找
            if content:
                break
        
        # 如果找到匹配的元素，提取其文本内容
        if content:
            content = content.get_text(separator='\n', strip=True)

        # 返回提取的文本内容
        return content
        
    
    def _get_comment(self,soup):
        # 提取点赞数
        likes = soup.find('span', class_='like-count')
        likes = likes.get_text(strip=True) if likes else "0"

        # 提取评论数
        comments = soup.find('span', class_='comment-count')
        comments = comments.get_text(strip=True) if comments else "0"
        return {"likes":likes,"comments":comments}
    
    def _get_goods(self,soup):
        data = {}
        # 提取商品名称
        name = soup.find('h1') or soup.find('span', class_='product-name')
        data['name'] = name.get_text(strip=True) if name else ""

        # 提取价格
        price = soup.find('span', class_='price')
        data['price'] = price.get_text(strip=True) if price else "0"

        # 提取销量
        sales = soup.find('div', class_='sales')
        data['sales'] = sales.get_text(strip=True) if sales else "0"

        # 提取评论数
        comments = soup.find('span', class_='comment-count')
        data['comments'] = comments.get_text(strip=True) if comments else "0"
        return data

    def _run(
        self, web_paths: List[str], run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> List[Document]:
        """Use the tool."""
        docs = []
        for path in web_paths:            
            try:
                soup = self._scrape(path)
                # Build metadata
                metadata = {"source": path}
                metadata.update(self._get_author(soup))
                metadata.update(self._get_comment(soup))
                metadata.update(self._get_goods(soup))
                metadata.update(self._get_pub_date(soup))
                metadata.update(self._get_titile(soup))
                docs.append(Document(page_content=self._get_content(soup), metadata=metadata))
            except Exception as e:
                # Log the error and continue with the next URL
                log.error(f"Error processing {path}: {e}")
                print(traceback.format_exc())
        return docs

    async def _arun(
        self,
        web_paths: List[str],
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> List[Document]:
        """Use the tool asynchronously."""
        tasks = []
        for path in web_paths:
            tasks.append(self._scrape(path))

        # Use asyncio.gather to wait for all tasks to complete
        pages = await asyncio.gather(*tasks)
        
        docs = []
        for path, soup in zip(web_paths, pages):            
            try:
                # Build metadata
                metadata = {"source": path}
                metadata.update(self._get_author(soup))
                metadata.update(self._get_comment(soup))
                metadata.update(self._get_goods(soup))
                metadata.update(self._get_pub_date(soup))
                metadata.update(self._get_titile(soup))
                docs.append(Document(page_content=self._get_content(soup), metadata=metadata))

            except Exception as e:
                # Log the error and continue with the next URL
                log.error(f"Error processing {path}: {e}")
        return docs
    
    def load(self) -> List[Document]: 
        return self._run(self.web_paths)