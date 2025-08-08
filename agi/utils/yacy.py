import asyncio
import httpx

YACY_HOST = "http://localhost:8090"
CRAWLER_URL = f"{YACY_HOST}/Crawler_p.html"

async def start_yacy_crawl_async(target_url, username=None, password=None):
    params = {
        "crawlingDomMaxPages": "10000",      # DOM 解析时最多访问页面数，防止过度爬取
        "range": "wide",                     # 爬取范围，'wide'表示广泛爬取
        "intention": "",                     # 爬取意图描述，空表示无特殊意图
        "sitemapURL": "",                   # 网站地图 URL，优先爬取该地址内容
        "crawlingQ": "on",                  # 是否启用爬取队列，允许排队执行
        "crawlingMode": "url",              # 爬取模式，'url'表示按 URL 爬取
        "crawlingURL": target_url,          # 爬取起始 URL
        "crawlingFile": "",                 # 限制爬取文件类型，空表示不限
        "mustnotmatch": "",                 # URL 黑名单正则，匹配的 URL 不爬取
        "crawlingFile$file": "",            # 内部文件参数，空表示无
        "crawlingstart": "Neuen Crawl starten", # 爬虫启动命令（德语“开始新爬取”）
        "mustmatch": ".*",                  # URL 白名单正则，匹配的 URL 才爬取
        "createBookmark": "on",             # 是否为此次爬取创建书签
        "bookmarkFolder": "/crawlStart",   # 书签存放目录路径
        "xsstopw": "on",                   # 是否启用停止词过滤
        "indexMedia": "on",                 # 是否索引媒体内容（图片、音频、视频）
        "crawlingIfOlderUnit": "hour",     # 内容老化判断单位，如小时
        "cachePolicy": "iffresh",           # 缓存策略，'iffresh'表示只用新鲜缓存
        "indexText": "on",                  # 是否索引文本内容
        "crawlingIfOlderCheck": "on",      # 是否启用内容老化检测
        "bookmarkTitle": "",                # 书签标题，空则自动生成
        "crawlingDomFilterDepth": "1",     # DOM 过滤深度，限制 DOM 分析层数
        "crawlingDomFilterCheck": "on",    # 是否启用 DOM 过滤检查
        "crawlingIfOlderNumber": "1",      # 老化判断数值，与单位配合，如1小时内视为新鲜
        "crawlingDepth": "4",               # 爬取深度，从起始 URL 递归跟进链接层数
    }

    auth = (username, password) if username and password else None

    async with httpx.AsyncClient() as client:
        response = await client.get(CRAWLER_URL, params=params, auth=auth)
        if response.status_code == 200:
            print(f"Started crawl for {target_url}")
        else:
            print(f"Failed to start crawl for {target_url}, status code: {response.status_code}")



SEARCH_API = f"{YACY_HOST}/yacysearch.json"

async def yacy_search_async(
    query: str,
    start_record: int = 0,
    maximum_records: int = 5,
    contentdom: str = "text",
    resource: str = "global",
    urlmaskfilter: str = ".*",
    prefermaskfilter: str = "",
    verify: str = "iffresh",
    lr: str = "",
    meancount: int = 3,
    nav: str = "none",
):
    """
    异步调用 YaCy 搜索 API。
    
    参数解释：
    - query: 搜索关键词，支持特殊关键词如 /date（按时间排序）、NEAR（词语近邻提升排名）、LANGUAGE:lang（指定语言）等。
    - start_record: 结果起始序号（分页用），如10表示从第11条结果开始。
    - maximum_records: 本次返回的最大结果数，非认证用户最大10条。
    - contentdom: 过滤内容类型，如 text、image、video、audio 等。
    - resource: 搜索范围，global表示询问整个网络节点，local表示仅本地节点。
    - urlmaskfilter: 用正则表达式限制结果 URL，默认 .* 表示不过滤。
    - prefermaskfilter: 优先显示匹配正则的结果。
    - verify: 结果验证方式，true表示验证并返回摘要，false加快速度，iffresh使用缓存但保证新鲜。
    - lr: 指定语言，如 lr=lang_en。
    - meancount: 返回“你是不是想找”最多备选查询数。
    - nav: 是否显示导航，all显示，none不显示。
    """
    params = {
        "query": query,
        "startRecord": start_record,
        "maximumRecords": maximum_records,
        "contentdom": contentdom,
        "resource": resource,
        "urlmaskfilter": urlmaskfilter,
        "prefermaskfilter": prefermaskfilter,
        "verify": verify,
        "lr": lr,
        "meancount": meancount,
        "nav": nav,
    }

    async with httpx.AsyncClient() as client:
        response = await client.get(SEARCH_API, params=params)
        response.raise_for_status()
        return response.json()


async def main():
    # 示例：搜索包含关键词 "freedom" 并按时间排序
    query = "freedom /date LANGUAGE:en"
    results = await yacy_search_async(query=query, maximum_records=5)
    
    for i, item in enumerate(results.get('searchResult', {}).get('results', []), 1):
        print(f"Result {i}:")
        print(f" Title: {item.get('title')}")
        print(f" URL: {item.get('url')}")
        print(f" Snippet: {item.get('snippet')}")
        print("-" * 40)
