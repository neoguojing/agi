import pytest
import asyncio

from agi.utils.yacy import yacy_search_async, start_yacy_crawl_async  # 假设函数在your_module.py里

YACY_HOST = "http://localhost:8090"

@pytest.mark.asyncio
async def test_yacy_search_async_basic():
    # 测试简单搜索返回结构
    query = "俄乌战争 /date LANGUAGE:en"
    results = await yacy_search_async(query=query, maximum_records=2)
    print(results)
    assert "searchResult" in results
    assert "results" in results["searchResult"]
    assert isinstance(results["searchResult"]["results"], list)
    assert len(results["searchResult"]["results"]) <= 2
    
    # 断言结果至少包含标题和 URL
    for item in results["searchResult"]["results"]:
        assert "title" in item
        assert "url" in item

# @pytest.mark.asyncio
# async def test_start_yacy_crawl_async_success(monkeypatch):
#     # 模拟异步 HTTP GET 响应，避免真实调用
#     class MockResponse:
#         status_code = 200
#         async def aclose(self): pass

#     class MockClient:
#         async def __aenter__(self): return self
#         async def __aexit__(self, exc_type, exc, tb): pass
#         async def get(self, url, params=None, auth=None):
#             assert url.startswith(YACY_HOST)
#             assert "crawlingURL" in params
#             return MockResponse()

#     async def mock_async_client():
#         return MockClient()

#     # monkeypatch httpx.AsyncClient 为模拟客户端
#     monkeypatch.setattr("httpx.AsyncClient", mock_async_client)

#     # 调用启动爬虫函数
#     await start_yacy_crawl_async("http://example.com", username="admin", password="pass")

# @pytest.mark.asyncio
# async def test_start_yacy_crawl_async_fail(monkeypatch):
#     # 模拟失败返回状态码
#     class MockResponse:
#         status_code = 500
#         async def aclose(self): pass

#     class MockClient:
#         async def __aenter__(self): return self
#         async def __aexit__(self, exc_type, exc, tb): pass
#         async def get(self, url, params=None, auth=None):
#             return MockResponse()

#     async def mock_async_client():
#         return MockClient()

#     monkeypatch.setattr("httpx.AsyncClient", mock_async_client)

#     # 捕获标准输出，确认打印失败信息
#     import io
#     import sys

#     captured_output = io.StringIO()
#     sys.stdout = captured_output

#     await start_yacy_crawl_async("http://example.com")

#     sys.stdout = sys.__stdout__
#     output = captured_output.getvalue()
#     assert "Failed to start crawl" in output
