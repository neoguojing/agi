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
from typing import Any, Optional, Type,List,Union
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
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate

scrape_template = '''
You are an expert HTML analyzer. Given the raw HTML source of a web page, extract the following fields related to a product or article:

1. title – The content of the `<title>` tag or the main page title.
2. time – The publication or update timestamp found on the page (e.g., in `<time>` tags or “Published on …”); normalize to ISO 8601 (YYYY-MM-DD) if possible, else return null.
3. author – The name of the author or seller if present; else null.
4. likes – The number of “likes” or upvotes displayed on the page; parse as an integer, else null.
5. comments – The number of comments or reviews displayed on the page; parse as an integer, else null.
6. name – The product’s name or item title (if this is a product page); else null.
7. price – The product’s price, including currency symbol if present (e.g., “$19.99”); else null.
8. sales – The number of units sold or sales count displayed; parse as an integer, else null.
9. content – The main body of the article or product description, stripped of HTML tags and whitespace-normalized; else null.

Return **only** valid JSON with exactly these keys:
```json
{{
  "title": "...",
  "time": "...",
  "author": "...",
  "likes": 123,
  "comments": 45,
  "name": "...",
  "price": "...",
  "sales": 678,
  "content": "..."
}}
Use null for any field you cannot find or parse.

Do not include any additional keys or commentary.

If a numeric field cannot be parsed, set it to null.

Here is the HTML to analyze:
{text}
'''

parser = JsonOutputParser()

prompt = PromptTemplate(
    template=scrape_template,
    input_variables=["text"],
)


class WLInput(BaseModel):
    """Input for the web loader tool."""

    web_paths: str = Field(description="urls need to be scraped")

class WebScraper(BaseTool):
    name: str = "web scraper"
    description: str = (
        "This web scraper tool takes a list of URLs as input and retrieves the content from each webpage."
    )

    args_schema: Type[BaseModel] = WLInput
    use_selenium: bool = False
    web_paths: List[str] = None
    chain: Any = None

    def __init__(self,web_paths:List[str] = None,llm: Any = None, use_selenium: bool = False,**kwargs):
        super().__init__(**kwargs)
        self.use_selenium = use_selenium  # Flag to use Selenium for dynamic pages
        self.web_paths = web_paths
        if llm:
            self.chain = prompt | llm | parser

    def _scrape(self, url: str) :
        """Scrape content from URL, with option for dynamic content loading via Selenium."""
        if self.use_selenium or "toutiao" in url:
            return self._scrape_dynamic(url)
        else:
            return self._scrape_with_requests(url)

    async def _ascrape(self, url: str):
        """Scrape content from URL, with option for dynamic content loading via Selenium."""
        if self.use_selenium or "toutiao" in url:
            return await self._ascrape_dynamic(url)
        else:
            return await self._scrape_with_aiohttp(url)

    def _scrape_with_requests(self, url: str):
        """Use requests and BeautifulSoup to scrape static content."""
        try:
            headers = self._get_random_headers()
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            return response.content
        except RequestException as e:
            log.error(f"Error fetching {url}: {e}")
            raise

    async def _scrape_with_aiohttp(self, url: str):
        """Use requests and BeautifulSoup to scrape static content."""
        try:
            headers = self._get_random_headers()
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, timeout=10) as response:
                    response.raise_for_status()
                    html = await response.text()

                    return html
        except Exception as e:
            log.error(f"Error fetching {url} with aiohttp: {e}")
            raise

    def _scrape_dynamic(self, url: str):
        """Use Playwright for scraping dynamic content synchronously."""
        with sync_playwright() as p:
            # 启动 Chromium 浏览器
            browser = p.chromium.launch(headless=True)  # headless=True 表示无头模式
            page = browser.new_page()
            try:
                page.goto(url, wait_until="networkidle", timeout=15000)
                page.wait_for_selector("body", timeout=5000)  # 额外保险
                html_content = page.content()

                return html_content
            finally:
                browser.close()

    async def _ascrape_dynamic(self, url: str):
        """优化后的动态内容爬取方法"""
        async with async_playwright() as p:
            browser = await p.chromium.launch(
                headless=True,
                args=["--disable-blink-features=AutomationControlled"]  # 反检测措施:ml-citation{ref="5" data="citationList"}
            )
            page = await browser.new_page()
            
            try:
                # 设置合理的默认超时
                page.set_default_timeout(15000)  # 全局超时15秒:ml-citation{ref="3" data="citationList"}

                # 加载页面并等待关键条件
                await page.goto(
                    url,
                    wait_until="networkidle",  # 或使用 "domcontentloaded" + "load" 组合:ml-citation{ref="8" data="citationList"}
                    timeout=30000
                )
                
                # 双重保险：等待特定动态元素
                await page.wait_for_function(
                    "document.readyState === 'complete'",
                    timeout=5000
                )
                
                # 获取完整HTML
                html_content = await page.content()
                
            except Exception as e:
                await page.screenshot(path="load_fail.png")
                raise
            finally:
                await browser.close()
                
        return html_content


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
    
    def _get_content(self, soup):
        # 三元组：(tag, attr, pattern)
        # attr=None: 只按标签；attr='class' / 'id'：按 class 或 id 正则匹配
        search_patterns = [
            ('article',      None,            None),
            ('section',      None,            None),
            ('main',         None,            None),

            ('div',          'class',     r'.*content.*'),
            ('div',          'id',        r'.*content.*'),
            ('div',          'class',     r'rich-text.*'),
            ('div',          'id',        r'rich-text.*'),
            ('div',          'class',     r'post-body.*'),
            ('div',          'id',        r'post-body.*'),
            ('div',          'class',     r'answer-body.*'),
            ('div',          'id',     r'answer-body.*'),
        ]

        # 如果 class/id 包含下列关键词，就认为是评论、页脚、元信息，不要提取
        blacklist = re.compile(r'comment|reply|footer|meta', re.IGNORECASE)

        for tag, attr, pattern in search_patterns:
            # 找到所有候选元素
            if attr is None:
                candidates = soup.find_all(tag)
            elif attr == 'class':
                candidates = soup.find_all(tag, class_=re.compile(pattern, re.IGNORECASE))
            else:  # attr == 'id'
                candidates = soup.find_all(tag, id=re.compile(pattern, re.IGNORECASE))

            # 过滤掉黑名单元素
            filtered = []
            for el in candidates:
                # 检查 class 和 id
                cls = ' '.join(el.get('class', [])) if el.get('class') else ''
                _id = el.get('id', '') or ''
                if not (blacklist.search(cls) or blacklist.search(_id)):
                    filtered.append(el)

            if not filtered:
                continue

            # 拼接所有符合的元素文本
            texts = [el.get_text(separator='\n', strip=True) for el in filtered]
            return '\n\n'.join(texts)

        # 都没匹配到，返回空字符串
        return ""

        
    
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

    def _analyser(self,html_content,path):
        soup = BeautifulSoup(html_content, "html.parser")
        content = self._get_content(soup)
        if not content:
            return None
        # Build metadata
        metadata = {"source": path}
        metadata.update(self._get_author(soup))
        metadata.update(self._get_comment(soup))
        metadata.update(self._get_goods(soup))
        metadata.update(self._get_pub_date(soup))
        metadata.update(self._get_titile(soup))

        return Document(page_content=content, metadata=metadata)
    
    def _analyser_with_llm(self,html_content,path):
        soup = BeautifulSoup(html_content, "lxml")
        text = soup.get_text(separator="\n")
        json_ret = self.chain.invoke({"text":text})
        print(json_ret)
        if json_ret.get("content"):
            return None
        # Build metadata
        metadata = {
            "source": path,
            "title": json_ret.get("title",""),
            "time": json_ret.get("time",""),
            "author": json_ret.get("author",""),
            "likes": json_ret.get("likes",0),
            "comments": json_ret.get("comments",0),
            "name": json_ret.get("name",""),
            "price": json_ret.get("price",""),
            "sales": json_ret.get("sales",0),      
        }

        return Document(page_content=json_ret.content, metadata=metadata)

    def _run(
        self, web_paths: Union[str,List[str],dict], run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> List[Document]:
        """Use the tool."""
        docs = []
        if isinstance(web_paths,str):
            web_paths = [web_paths]
        elif isinstance(web_paths,dict):
            web_paths = web_paths.get("urls")
            
        for path in web_paths:            
            try:
                html_content = self._scrape(path)
                doc = None
                if self.chain:
                    doc = self._analyser_with_llm(html_content,path)
                else:
                    doc = self._analyser(html_content,path)

                if doc:
                    docs.append(doc)
                
            except Exception as e:
                # Log the error and continue with the next URL
                log.error(f"Error processing {path}: {e}")
                print(traceback.format_exc())
        return docs

    async def _arun(
        self,
        web_paths: Union[str,List[str],dict],
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> List[Document]:
        """Use the tool asynchronously."""
        
        if isinstance(web_paths,str):
            web_paths = [web_paths]
        elif isinstance(web_paths,dict):
            web_paths = [web_paths.get("url")]
        tasks = []
        for path in web_paths:
            tasks.append(self._ascrape(path))

        # Use asyncio.gather to wait for all tasks to complete
        pages = await asyncio.gather(*tasks)
        
        docs = []
        for path, html_content in zip(web_paths, pages):            
            try:
                doc = None
                if self.chain:
                    doc = self._analyser_with_llm(html_content,path)
                else:
                    doc = self._analyser(html_content,path)
                    
                if doc:
                    docs.append(doc)

            except Exception as e:
                # Log the error and continue with the next URL
                log.error(f"Error processing {path}: {e}")
        return docs
    
    def load(self) -> List[Document]: 
        return self._run(self.web_paths)