import random
from collections import defaultdict
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from exa_py import Exa
from typing import Any, Optional, Type
from agi.config import EXA_API_KEY,log

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from typing import DefaultDict,Annotated
from pydantic import Field,BaseModel,PrivateAttr

class SGInput(BaseModel):
    """Input for the search engine tool."""

    query: str = Field(description="search query to look up")

class SearchEngineSelector(BaseTool):
    name: str = "search engine"
    description: str = (
        "Useful for when you need to answer questions about current events. "
        "Input should be a search query."
    )

    args_schema: Type[BaseModel] = SGInput

    max_results: int = 3
    max_retries: int = 3

    __search_engines: dict = PrivateAttr(default_factory=dict)
    __default_engines: list = PrivateAttr(default_factory=list)
    __success_stats: DefaultDict[str, dict] = PrivateAttr(default_factory=lambda: defaultdict(lambda: {'success': 0, 'total': 0}))

    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        # Always add DuckDuckGoSearch
        self._search_engines["DuckDuckGoSearch"] = DuckDuckGoSearchAPIWrapper(region="wt-wt", safesearch="moderate", time="y", max_results=3, source="text")

        if EXA_API_KEY:
            self.__search_engines["Exa"] = Exa(EXA_API_KEY)

        self._default_engines = list(self._search_engines.keys())
        
    def record_result(self, engine_name, success):
        """记录搜索引擎的使用结果"""
        self._success_stats[engine_name]['total'] += 1
        if success:
            self._success_stats[engine_name]['success'] += 1
    
    def get_success_rate(self, engine_name):
        """获取搜索引擎的成功率"""
        stats = self._success_stats[engine_name]
        if stats['total'] == 0:
            return 0.5  # 默认成功率
        return stats['success'] / stats['total']
    
    def select_engine(self):
        """根据成功率选择搜索引擎"""
        # 计算每个引擎的权重（成功率 + 小随机值避免完全排除低成功率引擎）
        weights = {
            name: self.get_success_rate(name) + random.uniform(0, 0.1)
            for name in self._default_engines
        }
        
        # 归一化权重
        total_weight = sum(weights.values())
        if total_weight == 0:
            return random.choice(self._default_engines)
            
        normalized_weights = {
            name: weight / total_weight
            for name, weight in weights.items()
        }
        
        # 根据权重随机选择
        return random.choices(
            list(normalized_weights.keys()),
            weights=list(normalized_weights.values()),
            k=1
        )
    
    def get_engine(self, name):
        """获取指定名称的搜索引擎实例"""
        return self._search_engines.get(name)

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> any:
        """Use the tool."""
        retries = 0
        success = False
        while retries < self.max_retries and not success:
            try:
                random_key = self.select_engine()
                search = self._search_engines[random_key]

                search_results = []
                if isinstance(search,DuckDuckGoSearchAPIWrapper):
                    search_results = search.results(query, max_results=self.max_results)
                elif isinstance(search,Exa):
                    resp = search.search_and_contents(
                        query,
                        type="auto",
                        num_results=self.max_results,
                        text=True,
                    )
                    search_results = []
                    for r in resp.results:
                        search_results.append({
                            "snippet": r.text,
                            "title": r.title,
                            "link": r.url,
                            "date": r.published_date,
                            "source": r.url,
                        })
                
                log.info(f"Search results for query '{q}': {search_results}")
                success = True  # 如果成功，就跳出重试循环
                # 汇报结果
                self.record_result(random_key,success)
                return search_results
            
            except Exception as e:
                retries += 1
                log.error(f"Error with search engine {random_key}, retrying {retries}/{self.max_retries}...")
                log.error(e)
                # 汇报结果
                self.record_result(random_key,success)
                if retries >= self.max_retries:
                    log.error("Max retries reached, skipping this query.")
                else:
                    continue  # 如果还没有达到最大重试次数，则继续尝试其他引擎    