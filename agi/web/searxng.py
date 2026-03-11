import httpx
import asyncio
from typing import List, Dict, Optional
from agi.config import log

class SearXNGSearch:
    def __init__(self, base_url: str = "http://localhost:8080", max_results: int = 10):
        """
        :param base_url: SearXNG 实例地址
        :param max_results: 每个引擎尝试获取的结果数
        """
        self.base_url = base_url.rstrip("/")
        self.search_url = f"{self.base_url}/search"
        self.max_results = max_results
        self.timeout = httpx.Timeout(15.0, connect=5.0)

    async def search(self, query: str, engines: Optional[List[str]] = None, language: str = "zh-CN", **kwargs) -> List[Dict]:
        """
        异步搜索接口
        :param query: 搜索词
        :param engines: 指定引擎列表（如 ['google', 'bing', 'wikipedia']），不传则使用实例默认配置
        :param language: 语言过滤
        """
        params = {
            "q": query,
            "format": "json",
            "pageno": kwargs.get("pageno", 1),
            "language": language,
            "safesearch": kwargs.get("safesearch", 1), # 0: None, 1: Moderate, 2: Strict
        }
        
        if engines:
            params["engines"] = ",".join(engines)

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(self.search_url, params=params)
                response.raise_for_status()
                data = response.json()
                
                results = data.get("results", [])
                return self._normalize_results(results)

        except Exception as e:
            log.error(f"SearXNG search error: {str(e)}")
            return []

    def _normalize_results(self, results: List[Dict]) -> List[Dict]:
        """标准化输出格式，对齐你的 SearchEngineSelector"""
        normalized = []
        for res in results[:self.max_results]:
            normalized.append({
                "title": res.get("title", ""),
                "link": res.get("url", ""),
                "snippet": res.get("content", ""),
                "source": res.get("engine", "SearXNG"),
                "score": res.get("score", 0.0),
                "category": res.get("category", "general")
            })
        return normalized

# --- 使用示例 ---
async def main():
    # 替换为你自己的 SearXNG 地址
    searx = SearXNGSearch(base_url="http://localhost:8080")
    
    query = "DeepSeek-V3 架构细节"
    print(f"正在搜索: {query}...")
    
    results = await searx.search(query, engines=["google", "bing"])
    
    for i, res in enumerate(results):
        print(f"[{i+1}] {res['title']}")
        print(f"    来源: {res['source']} | 链接: {res['link']}")
        print(f"    摘要: {res['snippet'][:100]}...\n")

if __name__ == "__main__":
    asyncio.run(main())