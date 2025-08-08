import pytest
import asyncio

from agi.utils.yacy import yacy_search_async, start_yacy_crawl_async  # 假设函数在your_module.py里

YACY_HOST = "http://localhost:8090"

@pytest.mark.asyncio
async def test_yacy_search_async_real_call():
    # 用一个合理的搜索词测试，限制返回条数避免过大
    query = "俄乌战争 /date LANGUAGE:zh"
    max_records = 5

    result = await yacy_search_async(query=query, maximum_records=max_records)

    # 断言返回的是 dict 且包含预期字段
    assert isinstance(result, dict)
    assert "totalResults" in result
    assert "items" in result
    assert isinstance(result["items"], list)

    # 验证返回条数不超过请求的最大条数
    assert len(result["items"]) <= max_records

    # 验证部分字段存在且非空
    for item in result["items"]:
        assert "title" in item and item["title"]
        assert "link" in item and item["link"]
        assert "description" in item
        assert "pubDate" in item

    print(f"Total results: {result['totalResults']}")
    print("First item:", result["items"][0] if result["items"] else "No items found")


# @pytest.mark.asyncio
# async def test_search_then_crawl_real():
#     # 先搜索，限制返回条数避免过多
#     query = "freedom /date LANGUAGE:en"
#     max_records = 3

#     search_result = await yacy_search_async(query=query, maximum_records=max_records)

#     assert "items" in search_result
#     items = search_result["items"]
#     assert len(items) > 0, "Search returned no items, cannot test crawl"

#     # 取所有链接，依次启动爬虫
#     for item in items:
#         url = item.get("link")
#         assert url, "Item missing link"
#         print(f"Starting crawl for: {url}")

#         # 调用爬虫启动函数，真实调用
#         await start_yacy_crawl_async(url, username="admin", password="yacy_password")

#     print(f"Tested crawling {len(items)} URLs from search results.")
