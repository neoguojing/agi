from typing import List, Dict, Any, Sequence,Optional,Callable
from langchain_core.tools import BaseTool
from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
import asyncio

from agi.rag.retriever import MultiCollectionRAGManager

class JITToolRegistry:
    def __init__(self, tools: Sequence[BaseTool]):
        # 将传入的工具数组转化为映射，方便按名索引
        self.tool_map: Dict[str, BaseTool] = {
            (t.name if hasattr(t, 'name') else t['name']): t 
            for t in tools
        }

    def get_tools_by_names(self, names: List[str]) -> List[BaseTool]:
        """根据名称从注册表中提取工具对象"""
        return [self.tool_map[n] for n in names if n in self.tool_map]

    def inject_to_request(self, request_tools: Optional[Sequence[Any]], dynamic_tools: List[BaseTool]) -> List[Any]:
        """
        模仿 deepagents 的注入逻辑：
        将原请求中的工具与动态工具合并，并进行去重。
        """
        existing_tools = list(request_tools or [])
        existing_names = set()
        
        # 提取已有工具的名称
        for t in existing_tools:
            name = t.name if hasattr(t, 'name') else t.get('name')
            if name: existing_names.add(name)
            
        # 仅添加不存在于原请求中的工具
        for dt in dynamic_tools:
            if dt.name not in existing_names:
                existing_tools.append(dt)
                existing_names.add(dt.name)
        
        return existing_tools
    

class JITOrchestratorMiddleware(AgentMiddleware):
    def __init__(self,tools: Sequence[BaseTool], manager: MultiCollectionRAGManager, top_k: int = 3):
        super().__init__()
        self.registry = JITToolRegistry(tools)
        self.manager = manager
        self.top_k = top_k
        self.sync_vector_store()

    def sync_vector_store(self):
        """将工具元数据一次性同步至向量库"""
        texts = []
        metas = []
        for tool in self.tools.values():
            # 提取工具描述和 Schema 作为检索文本
            schema = tool.args_schema.schema() if tool.args_schema else "No schema"
            text = f"Tool: {tool.name}\nDescription: {tool.description}\nArgs: {schema}"
            texts.append({"text": text})
            metas.append({"name": tool.name})
        
        # 假设 manager.ingest 处理文档存入 "tools" 集合
        self.manager.ingest_text("tools", texts,metas)
        print(f"✅ Vectorized {len(self.tools)} tools.")

    def wrap_model_call(self, request: ModelRequest, handler: Callable) -> ModelResponse:
        """
        拦截模型请求，执行语义检索并注入工具
        """
        # 1. 提取最后一条用户消息（意图）
        user_query = ""
        for msg in reversed(request.messages):
            if msg.type == "human":
                user_query = msg.content
                break
        
        if not user_query:
            return handler(request)

        # 2. 语义检索相关工具的名称 (Sync over Async)
        try:
            loop = asyncio.get_event_loop()
            # 假设 manager 返回匹配的工具名列表
            relevant_names = loop.run_until_complete(
                self.manager.query("tools",question=user_query, top_k=self.top_k)
            )
        except Exception:
            relevant_names = []

        # 3. 从注册表获取工具实例
        dynamic_tools = self.registry.get_tools_by_names(relevant_names)

        # 4. 执行注入逻辑：合并 + 去重
        final_tools = self.registry.inject_to_request(request.tools, dynamic_tools)

        # 5. 使用 override 触发下层 handler，这与 deepagents 的逻辑一致
        return handler(request.override(tools=final_tools))