import asyncio
import random
from collections import defaultdict
from typing import Any, Optional, Type, Dict, List, Set
from pydantic import Field, BaseModel, PrivateAttr
from langchain_core.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun

# 核心配置与日志
from agi.config import EXA_API_KEY, log, TAVILY_API_KEY,SEARXNG_BASE_URL

class SearchEngineSelector(BaseTool):
    name: str = "search"
    description: str = "Search for real-time information. Best for news, facts, and complex queries."
    
    max_results: int = 3
    max_retries: int = 2
    timeout_per_query: float = 10.0  # 单个引擎响应上限

    _engines: Dict[str, Any] = PrivateAttr(default_factory=dict)
    _stats: Dict[str, Dict] = PrivateAttr(default_factory=lambda: defaultdict(lambda: {'success': 1, 'total': 2}))

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._init_engines()

    def _init_engines(self):
        """
        重构后的动态初始化逻辑：
        1. 采用独立尝试机制，确保单一引擎失败不影响整体。
        2. 结构化引擎定义，方便快速增减。
        """
        self._engines = {}
        
        # 引擎定义映射表: {名称: (初始化函数, 依赖项检查)}
        engine_registry = {
            # "YACY": (self._init_yacy, lambda: False),
            # "SearXNG": (self._init_searxng, lambda: bool(SEARXNG_BASE_URL)),
            # "Exa": (self._init_exa, lambda: bool(EXA_API_KEY)),
            "Tavily": (self._init_tavily, lambda: bool(TAVILY_API_KEY)),
        }

        for name, (init_func, check_func) in engine_registry.items():
            if check_func():
                try:
                    engine_instance = init_func()
                    if engine_instance:
                        self._engines[name] = engine_instance
                        log.info(f"Engine '{name}' initialized successfully.")
                except Exception as e:
                    log.warning(f"Engine '{name}' failed to load: {e}")
            else:
                log.debug(f"Engine '{name}' skipped: Missing credentials or URL.")

        if not self._engines:
            log.error("Critical: No search engines were successfully initialized!")

    # --- 具体的引擎实例化逻辑（方便单独维护） ---

    def _init_yacy(self):
        from agi.web.yacy import YaCySearch
        return YaCySearch(max_results=self.max_results)

    def _init_searxng(self):
        # 假设你已经写好了 SearXNGSearch 类
        from agi.web.searxng import SearXNGSearch 
        return SearXNGSearch(base_url=SEARXNG_BASE_URL, max_results=self.max_results)

    def _init_exa(self):
        from exa_py import Exa
        return Exa(EXA_API_KEY)

    def _init_tavily(self):
        from langchain_tavily import TavilySearch
        return TavilySearch(max_results=self.max_results)

    # --- 策略算法：汤普森采样 ---
    def _select_engine(self) -> str:
        names = list(self._engines.keys())
        if not names: return None
        scores = {
            n: random.betavariate(self._stats[n]['success'], 
                                 self._stats[n]['total'] - self._stats[n]['success'] + 1)
            for n in names
        }
        return max(scores, key=scores.get)

    # --- 统一格式化与去重 ---
    def _standardize(self, name: str, raw: Any) -> List[Dict]:
        results = []
        try:
            if name == "Exa":
                for r in getattr(raw, 'results', []):
                    results.append({
                        "title": r.title, "link": r.url, 
                        "content": getattr(r, 'text', "")[:1000], "score": getattr(r, 'score', 0.5)
                    })
            elif name == "Tavily":
                for r in raw.get("results", []):
                    results.append({
                        "title": r.get("title", ""), "link": r.get("url", ""), 
                        "content": r.get("content", ""), "score": r.get("score", 0.5)
                    })
            elif name == "YACY":
                for r in raw:
                    results.append({
                        "title": r.get("title", ""), "link": r.get("link", ""), 
                        "content": r.get("snippet", ""), "score": 0.4 # YaCy 给予较低基础分
                    })
        except Exception as e:
            log.warning(f"Standardization error for {name}: {e}")
        return results

    # --- 核心执行：单次异步搜索 ---
    async def _search_single(self, query: str) -> List[Dict]:
        tried_engines = set()
        for attempt in range(self.max_retries):
            name = self._select_engine()
            if not name or name in tried_engines:
                # 如果引擎全挂了或已试过，尝试随机选一个没试过的
                remaining = set(self._engines.keys()) - tried_engines
                if not remaining: break
                name = random.choice(list(remaining))
            
            tried_engines.add(name)
            try:
                # 关键：增加超时控制，防止慢引擎阻塞
                res = await asyncio.wait_for(
                    self._execute_engine_api(name, query), 
                    timeout=self.timeout_per_query
                )
                if res:
                    self._stats[name]['success'] += 1
                    self._stats[name]['total'] += 1
                    return res
            except Exception as e:
                self._stats[name]['total'] += 1
                log.warning(f"Engine {name} failed for query '{query}': {type(e).__name__}")
        return []

    async def _execute_engine_api(self, name: str, query: str) -> List[Dict]:
        """封装具体的 API 调用"""
        engine = self._engines[name]
        if name == "Exa":
            resp = await asyncio.to_thread(engine.search_and_contents, query, num_results=self.max_results, text=True)
            return self._standardize(name, resp)
        elif name == "Tavily":
            resp = await asyncio.to_thread(engine.invoke, {"query": query})
            return self._standardize(name, resp)
        elif name == "YACY":
            resp = await asyncio.to_thread(engine.search, query=query, maximum_records=self.max_results)
            return self._standardize(name, resp)
        return []

    # --- 外部入口：批量并发搜索 ---
    async def batch_search(self, questions: List[str]) -> Dict[str, List[Dict]]:
        if not questions: return {}
        
        # 并发执行所有问题的搜索
        tasks = [self._search_single(q) for q in questions]
        raw_responses = await asyncio.gather(*tasks)
        
        # 结果聚合与去重 (基于 URL)
        final_results = {}
        for q, docs in zip(questions, raw_responses):
            if not docs: continue
            
            seen_urls = set()
            unique_docs = []
            # 按分数排序，优先保留高质量结果
            sorted_docs = sorted(docs, key=lambda x: x['score'], reverse=True)
            for d in sorted_docs:
                if d['link'] not in seen_urls:
                    unique_docs.append(d)
                    seen_urls.add(d['link'])
            
            final_results[q] = unique_docs[:self.max_results]
            
        log.info(f"Batch search complete. Questions: {len(questions)}, Success: {len(final_results)}")
        return final_results

    # --- 完善同步接口 ---
    def _run(
        self, 
        query: str, 
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> List[Dict]:
        """LangChain 同步接口：在单独的线程中运行异步逻辑"""
        import asyncio
        try:
            # 尝试在现有事件循环中运行（如果是在异步环境中调用同步方法）
            loop = asyncio.get_event_loop()
            if loop.is_running():
                from concurrent.futures import ThreadPoolExecutor
                with ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, self._search_single(query))
                    return future.result()
            return asyncio.run(self._search_single(query))
        except Exception as e:
            log.error(f"Sync execution failed: {e}")
            return []

    # --- 完善异步接口 (LangChain 推荐用法) ---
    async def _arun(
        self, 
        query: str, 
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> List[Dict]:
        """LangChain 异步接口：直接调用核心搜索逻辑"""
        try:
            results = await self._search_single(query)
            return results
        except Exception as e:
            log.error(f"Async execution failed: {e}")
            return []