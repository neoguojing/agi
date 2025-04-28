import requests
import aiohttp
import asyncio
from bs4 import BeautifulSoup
from typing import Iterator, List, Optional
from requests.exceptions import RequestException
import random
import time
from typing import Any, Optional, Type,List
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
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
    use_selenium: bool
    web_paths: List[str]

    def __init__(self,web_paths:List[str] = None, use_selenium: bool = False):
        self.use_selenium = use_selenium  # Flag to use Selenium for dynamic pages
        self.web_paths = web_paths

    def _scrape(self, url: str) -> BeautifulSoup:
        """Scrape content from URL, with option for dynamic content loading via Selenium."""
        if self.use_selenium or "toutiao" in url:
            return self._scrape_with_selenium(url)
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

    async def _scrape_with_requests(self, url: str) -> BeautifulSoup:
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

    def _scrape_with_selenium(self, url: str) -> BeautifulSoup:
        """Use Selenium for scraping dynamic content."""
        try:
            options = Options()
            options.headless = True
            driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
            driver.get(url)
            time.sleep(3)  # Wait for JavaScript to load
            html = driver.page_source
            driver.quit()
            soup = BeautifulSoup(html, "html.parser")
            return soup
        except Exception as e:
            log.error(f"Error fetching {url} with Selenium: {e}")
            raise

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
            ('div','post-content')
        ]

        # 使用生成器表达式查找第一个匹配的元素
        content = next(
            (soup.find(tag, class_=cls) for tag, cls in search_patterns if soup.find(tag, class_=cls)),
            ""  # 如果没有找到任何元素，返回 None
        )
        if content:
            content = content.get_text(separator='\n', strip=True)
        
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
        data['name'] = name.get_text(strip=True) if name else "No name found"

        # 提取价格
        price = soup.find('span', class_='price')
        data['price'] = price.get_text(strip=True) if price else "No price found"

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