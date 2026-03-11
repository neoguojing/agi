from typing import List, Callable, Dict, Any,Union
from langchain_core.tools import tool, BaseTool
from langchain_core.documents import Document

from agi.rag.retriever import KnowledgeManager

class ToolRegistryManager:
    def __init__(self, km: KnowledgeManager, collection_name: str = "agent_tools"):
        self.km = km
        self.collection_name = collection_name
        # 运行时缓存，用于根据名称还原函数对象
        self._tool_instances: Dict[str, BaseTool] = {}

    async def register_tools(self, tools: List[Union[Callable, BaseTool]], tenant: str = "system"):
        """
        1. 提取 Tool 的元数据（名称、描述、参数描述）
        2. 转化为 Document 存入 KnowledgeManager
        3. 缓存实例用于后续还原
        """
        docs_to_index = []
        for t in tools:
            # 确保是 LangChain 的 Tool 对象
            instance = t if isinstance(t, BaseTool) else tool(t)
            self._tool_instances[instance.name] = instance
            
            # 构造工具的“特征描述”供向量检索
            content = f"Tool Name: {instance.name}\nDescription: {instance.description}"
            # 获取参数 schema 的字符串表示，增加检索维度
            args_info = str(instance.args)
            
            doc = Document(
                page_content=f"{content}\nArguments: {args_info}",
                metadata={
                    "source": f"tool://{instance.name}",
                    "tool_name": instance.name,
                    "type": "tool_definition"
                }
            )
            docs_to_index.append(doc)

        # 调用底层 KnowledgeManager 存储工具定义
        await self.km.store(
            collection_name=self.collection_name,
            content=docs_to_index,
            tenant=tenant
        )

    async def retrieve_and_restore(self, query: str, tenant: str = "system", k: int = 5) -> List[BaseTool]:
        """
        1. 根据用户意图 query 从向量库检索匹配的工具 ID
        2. 从缓存中还原 Tool 实例
        """
        # 利用 KM 的混合检索能力查找工具描述
        docs = await self.km.query_doc(
            collection_name=self.collection_name,
            query=query,
            tenant=tenant,
            k=k
        )
        
        restored_tools = []
        for d in docs:
            t_name = d.metadata.get("tool_name")
            if t_name in self._tool_instances:
                restored_tools.append(self._tool_instances[t_name])
        
        return restored_tools