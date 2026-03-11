from typing import List
from langchain.tools import tool, ToolRuntime
from langgraph.types import Command
from langchain.messages import ToolMessage
from agi.rag.retriever import KnowledgeManager  # 假设原始逻辑已封装在此
from agi.config import log

# 假设定义的上下文 Schema
from dataclasses import dataclass
@dataclass
class TenantContext:
    tenant_id: str
    collection_name: str



def _run_sync(coro):
    import asyncio

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    # Called from an active event loop (e.g. notebook/framework runtime).
    # Run in a dedicated loop to keep the sync tool interface.
    import threading

    result = {}

    def _runner():
        result["value"] = asyncio.run(coro)

    t = threading.Thread(target=_runner, daemon=True)
    t.start()
    t.join()
    return result.get("value")


class KnowledgeTools:
    def __init__(self, manager: KnowledgeManager):
        self.manager = manager

    @tool
    def store_document(
        self, 
        file_paths: List[str], 
        runtime: ToolRuntime[TenantContext]
    ) -> Command:
        """
        Index local files into the knowledge base for future retrieval.

        Args:
            file_paths: A list of local file paths to be processed and indexed.
        """
        # 从 Runtime Context 自动获取租户和集合信息，无需 LLM 传入
        tenant = runtime.context.tenant_id
        collection = runtime.context.collection_name
        
        _run_sync(self.manager.store(collection, file_paths, tenant=tenant))

        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=f"Successfully indexed {len(file_paths)} files to {collection}.",
                        tool_call_id=runtime.tool_call_id
                    )
                ],
                "last_action": "document_stored" # 更新自定义 Graph State
            }
        )

    @tool
    def query_knowledge_base(
        self, 
        query: str, 
        runtime: ToolRuntime[TenantContext]
    ) -> str:
        """
        Retrieve relevant document snippets from the knowledge base based on a semantic query.

        Args:
            query: The natural language question or search terms to look up in the indexed documents.
        """
        tenant = runtime.context.tenant_id
        collection = runtime.context.collection_name
        
        docs = _run_sync(self.manager.query_doc(collection, query, tenant=tenant, k=4))
        
        if not docs:
            return "No relevant information found in the knowledge base."
            
        # 格式化输出给 LLM 读
        formatted_docs = []
        for i, d in enumerate(docs):
            formatted_docs.append(f"Source [{d.metadata.get('source')}]:\n{d.page_content}")
            
        return "\n\n".join(formatted_docs)

    def get_tools(self):
        return [self.store_document, self.query_knowledge_base]