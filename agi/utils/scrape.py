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
        platforms = ["toutiao","zhihu","baidu"]
        if self.use_selenium or any(phrase in url for phrase in platforms):
            return self._scrape_dynamic(url)
        elif "weixin" in url:
            return self.weixin_article_scrape(url)
        else:
            return self._scrape_with_requests(url)

    async def _ascrape(self, url: str):
        """Scrape content from URL, with option for dynamic content loading via Selenium."""
        platforms = ["toutiao","zhihu","baidu"]
        if self.use_selenium or any(phrase in url for phrase in platforms):
            return await self._ascrape_dynamic(url)
        elif "weixin" in url:
            return await self.aweixin_article_scrape(url)
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
            browser = p.chromium.launch(
                headless=False,  # 非无头模式更隐蔽
                args=[
                    "--disable-blink-features=AutomationControlled",
                    f"--user-agent={self._get_user_agent()}",
                    "--disable-infobars",
                    "--no-first-run",
                    "--disable-extensions",
                    "--disable-web-security"
                ],
                ignore_default_args=["--enable-automation"]
            )
            
            context = browser.new_context(
                locale='zh-CN',
                timezone_id="Asia/Shanghai",
                viewport={'width': 1920, 'height': 1080}
            )
            
            page = context.new_page()
            
            # 关键JS注入
            page.add_init_script("""
                Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
                Object.defineProperty(navigator, 'plugins', {get: () => [1, 2, 3]});
                window.chrome = {runtime: {}};
            """)
            
            try:
                # 模拟人类操作间隔
                time.sleep(random.uniform(1.0, 3.0))
                
                # 带随机滚动的页面加载
                page.goto(url, wait_until="networkidle", timeout=20000)
                page.evaluate("window.scrollBy(0, document.body.scrollHeight * 0.5)")
                time.sleep(random.uniform(0.5, 1.5))
                page.evaluate("window.scrollBy(0, document.body.scrollHeight * 0.8)")
                
                # 多层等待策略
                page.wait_for_selector("body", state="attached", timeout=15000)
                page.wait_for_function("document.readyState === 'complete'")
                
                return page.content()
            finally:
                browser.close()

    async def _ascrape_dynamic(self, url: str):
        """优化后的动态内容爬取方法"""
        async with async_playwright() as p:
            browser = await p.chromium.launch(
                headless=False,  # 非无头模式更隐蔽
                args=[
                    "--disable-blink-features=AutomationControlled",
                    f"--user-agent={self._get_user_agent()}",
                    "--disable-infobars",
                    "--no-first-run",
                    "--disable-extensions",
                    "--disable-web-security"
                ],
                ignore_default_args=["--enable-automation"]
            )
            
            context = await browser.new_context(
                locale='zh-CN',
                timezone_id="Asia/Shanghai",
                viewport={'width': 1920, 'height': 1080}
            )
            
            page = await context.new_page()
            
            # 关键JS注入
            await page.add_init_script("""
                Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
                Object.defineProperty(navigator, 'plugins', {get: () => [1, 2, 3]});
                window.chrome = {runtime: {}};
            """)
            
            try:
                # 模拟人类操作间隔
                await page.wait_for_timeout(random.randint(1000, 3000))
                
                # 带随机滚动的页面加载
                await page.goto(url, wait_until="networkidle", timeout=30000)
                await page.evaluate("window.scrollBy(0, document.body.scrollHeight * 0.5)")
                await page.wait_for_timeout(random.randint(500, 1500))
                await page.evaluate("window.scrollBy(0, document.body.scrollHeight * 0.8)")
                
                # 多层等待策略
                await page.wait_for_selector("body", state="attached", timeout=15000)
                await page.wait_for_function("document.readyState === 'complete'")
                
                return await page.content()
            except Exception as e:
                await page.screenshot(path="load_fail.png")
                raise
            finally:
                await browser.close()

    def weixin_article_scrape(self, url: str):
        _ua_pool = [
            # 安卓微信X5内核
            "Mozilla/5.0 (Linux; Android 10; MI 9 Build/QKQ1.190825.002; wv) AppleWebKit/537.36 (KHTML, like Gecko) Version/4.0 Chrome/89.0.4389.72 MQQBrowser/6.2 TBS/046211 Mobile Safari/537.36 MicroMessenger/8.0.40.2400(0x28002851) WeChat/arm64",
            # iOS微信
            "Mozilla/5.0 (iPhone; CPU iPhone OS 16_5 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Mobile/15E148 MicroMessenger/8.0.40(0x18002831) NetType/WIFI Language/zh_CN"
        ]
        with sync_playwright() as p:
            browser = p.chromium.launch(
                headless=False,
                args=[
                    "--disable-blink-features=AutomationControlled",
                    f"--user-agent={random.choice(_ua_pool)}",
                    "--disable-web-security"
                ],
                ignore_default_args=["--enable-automation"]
            )
            
            context = browser.new_context(
                locale='zh-CN',
                timezone_id="Asia/Shanghai",
                viewport={'width': 414, 'height': 896},
                device_scale_factor=random.choice([2, 3])
            )
            
            page = context.new_page()
            page.add_init_script("""
                Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
                window.__wxjs_environment = 'miniprogram';
                window.chrome = {runtime: {}};
                document.cookie = "wxuin=123456789; domain=.weixin.qq.com";
            """)
            
            try:
                page.wait_for_timeout(random.randint(1000, 3000))
                page.goto(url, wait_until="networkidle", timeout=30000)
                page.wait_for_selector("#img-content", state="attached", timeout=15000)
                page.evaluate("window.scrollBy(0, 500)")
                page.wait_for_timeout(random.randint(800, 1500))
                
                content = page.evaluate("""
                    () => document.getElementById('js_content')?.innerText || ''
                """)
                return content
            except Exception as e:
                page.screenshot(path="debug.png")
                raise
            finally:
                browser.close()

    async def aweixin_article_scrape(self, url: str):
        """爬取微信文章核心方法"""
        _ua_pool = [
            # 安卓微信X5内核
            "Mozilla/5.0 (Linux; Android 10; MI 9 Build/QKQ1.190825.002; wv) AppleWebKit/537.36 (KHTML, like Gecko) Version/4.0 Chrome/89.0.4389.72 MQQBrowser/6.2 TBS/046211 Mobile Safari/537.36 MicroMessenger/8.0.40.2400(0x28002851) WeChat/arm64",
            # iOS微信
            "Mozilla/5.0 (iPhone; CPU iPhone OS 16_5 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Mobile/15E148 MicroMessenger/8.0.40(0x18002831) NetType/WIFI Language/zh_CN"
        ]
        async with async_playwright() as p:
            browser = await p.chromium.launch(
                headless=False,
                args=[
                    "--disable-blink-features=AutomationControlled",
                    f"--user-agent={random.choice(_ua_pool)}",
                    "--disable-web-security",
                    "--disable-infobars"
                ],
                ignore_default_args=["--enable-automation"]
            )
            
            context = await browser.new_context(
                locale='zh-CN',
                timezone_id="Asia/Shanghai",
                viewport={'width': 414, 'height': 896},  # 移动端尺寸
                device_scale_factor=random.choice([2, 3])  # 高DPI设备
            )
            
            page = await context.new_page()
            
            # 关键微信环境注入
            await page.add_init_script("""
                Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
                window.__wxjs_environment = 'miniprogram';
                window.chrome = {runtime: {}};
                document.cookie = "wxuin=123456789; domain=.weixin.qq.com";
            """)
            
            try:
                # 模拟人类操作序列
                await page.wait_for_timeout(random.randint(1000, 3000))
                await page.goto(url, wait_until="networkidle", timeout=30000)
                
                # 微信特有元素检测
                await page.wait_for_selector("#img-content", state="attached", timeout=15000)
                await page.evaluate("window.scrollBy(0, 500)")
                await page.wait_for_timeout(random.randint(800, 1500))
                
                # 获取净化后的内容
                content = await page.evaluate("""
                    () => {
                        const article = document.getElementById('js_content');
                        return article ? article.innerText : '';
                    }
                """)
                
                return content
            except Exception as e:
                await page.screenshot(path="debug.png")
                raise
            finally:
                await browser.close()
                
    def _get_random_headers(self) -> dict:
        """Generate random headers to mimic real user requests."""
        return {
            "User-Agent": self._get_user_agent(),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
        }
    
    
    def _get_user_agent(self) -> str:
        """生成动态更新的用户代理，包含设备指纹特征"""
        platforms = [
            # Windows设备(占60%权重)
            {
                'template': "Mozilla/5.0 (Windows NT {version}; {arch}) AppleWebKit/{webkit} (KHTML, like Gecko) Chrome/{chrome_version} Safari/{webkit}",
                'versions': ['10.0', '11.0'],
                'arches': ['Win64; x64', 'WOW64'],
                'weights': 0.6
            },
            # macOS设备(占25%权重)
            {
                'template': "Mozilla/5.0 (Macintosh; Intel Mac OS X {version}) AppleWebKit/{webkit} (KHTML, like Gecko) Chrome/{chrome_version} Safari/{webkit}",
                'versions': ['10_15_7', '13_5', '14_0'],
                'weights': 0.25
            },
            # 移动设备(占15%权重)
            {
                'template': "Mozilla/5.0 ({device}; {os}) AppleWebKit/{webkit} (KHTML, like Gecko) Chrome/{chrome_version} Mobile Safari/{webkit}",
                'devices': ['iPhone', 'iPad', 'Android Mobile'],
                'oses': ['CPU iPhone OS 16_5 like Mac OS X', 'Linux; Android 13'],
                'weights': 0.15
            }
        ]
        
        # 动态生成版本号
        chrome_version = f"{random.randint(110, 125)}.0.{random.randint(1000, 9999)}.{random.randint(50, 200)}"
        webkit_version = "537.36" if random.random() > 0.3 else "605.1.15"
        
        # 按权重选择平台
        chosen_platform = random.choices(platforms, weights=[p['weights'] for p in platforms])[0]
        
        # 构造最终UA
        if 'Windows' in chosen_platform['template']:
            return chosen_platform['template'].format(
                version=random.choice(chosen_platform['versions']),
                arch=random.choice(chosen_platform['arches']),
                webkit=webkit_version,
                chrome_version=chrome_version
            )
        elif 'Macintosh' in chosen_platform['template']:
            return chosen_platform['template'].format(
                version=random.choice(chosen_platform['versions']),
                webkit=webkit_version,
                chrome_version=chrome_version
            )
        else:
            return chosen_platform['template'].format(
                device=random.choice(chosen_platform['devices']),
                os=random.choice(chosen_platform['oses']),
                webkit=webkit_version,
                chrome_version=chrome_version
            )


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
    
    def extract_main_content(self,html_content):
        """
        通用网页正文提取函数
        参数:
            html_content: 网页HTML内容
        返回:
            净化后的正文文本
        """
        soup = BeautifulSoup(html_content, 'lxml')
        
        # 常见正文容器识别规则
        CONTENT_SELECTORS = [
            {'name': 'article'},  # HTML5标准
            {'class': re.compile(r'content|main|post|article')},
            {'id': re.compile(r'content|main|body|article')},
            {'role': 'main'}
        ]
        
        # 尝试定位正文容器
        content_node = None
        for selector in CONTENT_SELECTORS:
            content_node = soup.find(**selector)
            if content_node: break
        
        # 回退方案：使用body或整个文档
        content_node = content_node or soup.find('body') or soup
        
        # 净化处理
        for element in content_node(['script', 'style', 'nav', 'footer', 
                                'aside', 'iframe', 'form', 'button']):
            element.decompose()
        
        # 高级文本处理
        text = content_node.get_text('\n', strip=True)
        lines = [
            line for line in text.split('\n') 
            if len(line.strip()) > 20  # 过滤短文本行
            and not re.search(r'广告|推荐|相关阅读|copyright', line.lower())
        ]
        return '\n'.join(lines)

    def _analyser_with_llm(self,html_content,path):
        # soup = BeautifulSoup(html_content, "lxml")
        # text = soup.get_text(separator="\n")
        text = self.extract_main_content(html_content)
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

        return Document(page_content=json_ret.get("content"), metadata=metadata)

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