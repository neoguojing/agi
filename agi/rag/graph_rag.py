import os
import asyncio
from typing import Literal, Dict, List, Any, Optional
from llama_index.core.indices.property_graph import SchemaLLMPathExtractor, PropertyGraphIndex
from llama_index.graph_stores.memgraph import MemgraphPropertyGraphStore
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.schema import Document

# --- 1. 定义认知本体 (同前) ---
entities = Literal["USER", "PROJECT", "TECH_STACK", "USER_PREFERENCE", "GOAL", "ORGANIZATION"]
relations = Literal["SKILLED_IN", "WORKS_ON", "INTERESTED_IN", "MEMBER_OF", "PART_OF", "HAS_GOAL", "PREFERS", "DEPENDS_ON"]

validation_schema = {
    "USER": ["SKILLED_IN", "WORKS_ON", "INTERESTED_IN", "MEMBER_OF", "HAS_GOAL", "PREFERS"],
    "PROJECT": ["PART_OF", "DEPENDS_ON"],
    "TECH_STACK": ["PART_OF"],
    "GOAL": ["PART_OF"],
}

llm = Ollama(model="qwen3.5:9b", temperature=0.0)
embed_model = OllamaEmbedding(model_name="embeddinggemma:latest")

kg_extractor = SchemaLLMPathExtractor(
    llm=llm,
    possible_entities=entities,
    possible_relations=relations,
    kg_validation_schema=validation_schema,
    strict=True,
    num_workers=4
)

# --- 2. 数据库管理器：负责多库调度 ---

class MemgraphDBManager:
    """负责数据库层面的隔离逻辑"""
    def __init__(self, url="bolt://localhost:7687", username="memgraph", password="password"):
        self.url = url
        self.username = username
        self.password = password

    def get_user_db_name(self, user_id: str) -> str:
        # 转换 user_id 为合法的数据库名称（只允许字母数字下划线，且不能以数字开头）
        safe_id = "".join([c if c.isalnum() else "_" for c in user_id])
        return f"db_{safe_id}"

    def ensure_database(self, db_name: str):
        """确保用户的数据库已创建"""
        # 连接到系统默认库执行管理命令
        admin_store = MemgraphPropertyGraphStore(
            url=self.url, username=self.username, password=self.password, database="memgraph"
        )
        try:
            # Memgraph Cypher 创建库命令
            admin_store.execute(f"CREATE DATABASE {db_name};")
        except Exception:
            # 如果已存在会报错，忽略即可
            pass
        finally:
            admin_store.close()

    def get_user_store(self, user_id: str) -> MemgraphPropertyGraphStore:
        db_name = self.get_user_db_name(user_id)
        self.ensure_database(db_name)
        return MemgraphPropertyGraphStore(
            url=self.url,
            username=self.username,
            password=self.password,
            database=db_name
        )

# --- 3. 重构后的组件 ---

class GraphMemoryReflector:
    """异步反思器：将信息存入用户专属库"""
    def __init__(self, db_manager: MemgraphDBManager):
        self.db_manager = db_manager

    async def distill_and_persist(self, user_id: str, user_msg: str, ai_msg: str):
        store = self.db_manager.get_user_store(user_id)
        try:
            content = f"User said: {user_msg}\nAssistant replied: {ai_msg}"
            doc = Document(text=content, metadata={"user_id": user_id})
            
            # 为当前用户构建/更新索引
            PropertyGraphIndex.from_documents(
                [doc],
                property_graph_store=store,
                kg_extractors=[kg_extractor],
                embed_model=embed_model,
                show_progress=False
            )
        finally:
            store.close() # 必须关闭，防止连接堆积



class GraphContextProvider(ContextProvider):
    """上下文提供者：从用户专属库读取信息"""
    def __init__(self, db_manager: MemgraphDBManager):
        self.db_manager = db_manager

    async def load(self, runtime, state) -> dict:
        user_id = runtime.context.user_id
        query_text = state["messages"][-1].content if state["messages"] else ""
        if not query_text:
            return {}

        store = self.db_manager.get_user_store(user_id)
        try:
            # 加载该用户专属库的现有索引
            index = PropertyGraphIndex.from_existing(
                property_graph_store=store,
                llm=llm,
                embed_model=embed_model
            )
            query_engine = index.as_query_engine(include_text=True)
            response = await query_engine.aquery(query_text)
            
            return {"knowledge_graph_context": str(response)}
        except Exception as e:
            print(f"User {user_id} retrieval failed: {e}")
            return {}
        finally:
            store.close()

---

### 4. 关键点说明

# 1.  **物理安全隔离**：每个 `user_id` 拥有一个独立的 `db_xxxx`。即使用户通过 Prompt Injection 试图绕过应用层逻辑，他也无法在数据库底层访问到 `db_other_user` 的数据。
# 2.  **动态连接分配**：在 `load` 和 `persist` 方法中，我们不再预存 `index` 实例，而是通过 `db_manager` 实时获取。这解决了多租户场景下数据库实例切换的问题。
# 3.  **资源管理**：
#     * **连接泄露**：重构代码在 `finally` 块中显式调用了 `store.close()`。在多用户并发时，这是防止 Memgraph 连接数达到上限的关键。
#     * **冷启动**：第一次请求时会触发 `CREATE DATABASE`，可能会有秒级的延迟，后续请求将直接命中已有库。
# 4.  **本体一致性**：虽然数据库隔离了，但所有用户库共享同一套 `validation_schema`。这确保了系统在全局抽象层的一致性。

# ---

# **下一步建议：**
# 这种方案在用户量较少（如数百人）时表现完美。如果你的系统需要支持数万名用户，频繁的 `CREATE DATABASE` 和过多的数据库实例可能会导致 Memgraph 元数据管理压力增大。届时，你是否需要了解如何引入**数据库连接池**来优化高并发下的响应速度？