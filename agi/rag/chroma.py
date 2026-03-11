import chromadb
from chromadb import Settings
from chromadb.utils.embedding_functions.ollama_embedding_function import OllamaEmbeddingFunction
from langchain_core.documents import Document
from agi.utils.nlp import TextProcessor
from agi.config import log,EMBEDDING_BASE_URL,RAG_EMBEDDING_MODEL,OLLAMA_API_BASE_URL
from typing import List,Dict,Tuple,Optional
import asyncio
import uuid
import math
import os
from tqdm import tqdm  # 可选：用于进度显示
from threading import Lock

class ChromaClientFactory:
    """
    ChromaDB 客户端工厂：
    1. 负责多租户与多数据库的自动初始化。
    2. 提供基于 (tenant, database) 的实例缓存，避免频繁创建连接。
    3. 线程安全，支持并发调用。
    """
    def __init__(self, data_path: str, allow_reset: bool = False):
        self.data_path = data_path
        self.settings = Settings(
            chroma_api_impl="chromadb.api.segment.SegmentAPI",
            is_persistent=True,
            persist_directory=data_path,
            allow_reset=allow_reset,
            anonymized_telemetry=False,
        )
        # 管理端 Client：专门用于创建租户和数据库
        self.admin_client = chromadb.AdminClient(self.settings)
        # 客户端缓存池：{(tenant, database): PersistentClient}
        self._client_pool: Dict[Tuple[str, str], chromadb.PersistentClient] = {}
        self._lock = Lock()

    def _ensure_tenant_and_db(self, tenant: str, database: str):
        """确保指定的租户和数据库在底层存储中已存在"""
        try:
            # 检查租户，不存在则创建
            try:
                self.admin_client.get_tenant(tenant)
            except Exception:
                log.info(f"Creating new tenant: {tenant}")
                self.admin_client.create_tenant(tenant)
            
            # 检查数据库，不存在则创建
            try:
                self.admin_client.get_database(database, tenant)
            except Exception:
                log.info(f"Creating new database: {database} for tenant: {tenant}")
                self.admin_client.create_database(database, tenant)
        except Exception as e:
            log.error(f"Failed to initialize tenant/db structure: {e}")
            raise

    def get_client(self, tenant: str = "default_tenant", database: str = "default_database") -> chromadb.PersistentClient:
        """获取特定租户和数据库的持久化客户端"""
        pool_key = (tenant, database)
        
        # 第一轮检查（无锁）
        if pool_key in self._client_pool:
            return self._client_pool[pool_key]

        with self._lock:
            # 第二轮检查（有锁），防止并发创建
            if pool_key not in self._client_pool:
                self._ensure_tenant_and_db(tenant, database)
                
                client = chromadb.PersistentClient(
                    path=self.data_path,
                    settings=self.settings,
                    tenant=tenant,
                    database=database,
                )
                self._client_pool[pool_key] = client
                log.debug(f"Client initialized for {pool_key}")
                
            return self._client_pool[pool_key]

class CollectionHandle:
    def __init__(self, collection, embedding_model,text_processor):
        self.collection = collection
        self.embedding = embedding_model
        self.text_proc = text_processor

    # --- Create / Update (CRUD: C/U) ---

    async def upsert(self, documents: List[Document], batch_size: int = 50, auto_process: bool = True):
        """
        异步批量 Upsert。
        补充：使用 TextProcessor 进行关键词自动增强和文本清洗。
        """
        if not documents: return

        # 1. 文本预处理与关键词提取 (如果 auto_process 为 True)
        processed_docs = documents
        if auto_process:
            texts = [doc.page_content for doc in documents]
            # 批量提取关键词用于元数据增强
            kw_results = await self.text_proc.abatch_process(texts, method="textrank")
            
            for i, doc in enumerate(processed_docs):
                # 将 TextProcessor 提取的关键词注入元数据，方便后续 build_query
                if "keywords" not in doc.metadata:
                    doc.metadata["keywords"] = kw_results[i]

        # 2. 核心元数据清洗 (保留源码细节：列表转字符串)
        ids = []
        metadatas = []
        for doc in processed_docs:
            ids.append(doc.metadata.get("doc_id") or str(uuid.uuid4()))
            meta = doc.metadata.copy()
            for k, v in meta.items():
                if isinstance(v, list):
                    # 格式化：keyword1:0.9,keyword2:0.8
                    if k == "keywords" and v and isinstance(v[0], tuple):
                        meta[k] = ",".join([f"{kw}:{round(sc, 3)}" for kw, sc in v])
                    else:
                        meta[k] = ",".join(map(str, v))
            metadatas.append(meta)

        # 3. 异步并发生成 Embedding
        texts = [doc.page_content for doc in processed_docs]
        embeddings = await asyncio.gather(*[
            asyncio.to_thread(self.embedding.embed_query, t) for t in texts
        ])

        # 4. 执行写入
        for i in range(0, len(ids), batch_size):
            end = i + batch_size
            self.collection.upsert(
                ids=ids[i:end],
                embeddings=list(embeddings[i:end]),
                metadatas=metadatas[i:end],
                documents=texts[i:end]
            )
        log.info(f"Successfully processed and upserted {len(ids)} docs.")

    # --- 混合搜索 (Search with TextProcessor) ---
    async def hybrid_search(self, texts: List[str], k: int = 10, **kwargs) -> Dict[str, List[Document]]:
        """
        补充：利用 TextProcessor 提取搜索关键词进行 where_document 过滤。
        """
        # 利用 TextProcessor 提取查询文本的关键词
        kw_results = await self.text_proc.abatch_process(texts, method="textrank")

        async def _single_query(content, kws):
            # 提取权重最高的前 5 个关键词
            search_kws = [kw[0] for kw in kws[:5]]
            
            # 构建 Chroma 专用的 $contains 过滤器
            doc_filter = self._build_where_document(search_kws)
            
            # 属性过滤逻辑
            where_attr = {k: v for k, v in kwargs.items() if k in ["cluster_id", "source"]}

            # 并发执行向量编码
            emb = await asyncio.to_thread(self.embedding.embed_query, content)

            return self.collection.query(
                query_embeddings=[emb],
                n_results=k,
                where_document=doc_filter,
                where=where_attr if where_attr else None
            )

        tasks = [_single_query(texts[i], kw_results[i]) for i in range(len(texts))]
        raw_results = await asyncio.gather(*tasks)

        # 结果映射封装
        return {
            text: [
                Document(page_content=d, metadata={**m, "score": s})
                for d, m, s in zip(res['documents'][0], res['metadatas'][0], res['distances'][0])
            ]
            for text, res in zip(texts, raw_results)
        }

    def _build_where_document(self, kws: List[str]):
        if not kws: return None
        filters = [{"$contains": kw} for kw in kws]
        return {"$or": filters} if len(filters) > 1 else filters[0]
    
    # --- Read (CRUD: R) ---

    def get_documents(self, source: str = None, limit: int = 10, offset: int = 0) -> List[Document]:
        """
        Read: 分页查询文档。
        保留源码细节：支持基于 source 文件名的过滤。
        """
        where_cond = {"source": os.path.basename(source)} if source else None
        result = self.collection.get(limit=limit, offset=offset, where=where_cond)
        
        return [
            Document(page_content=doc, metadata=meta) 
            for doc, meta in zip(result['documents'], result['metadatas'])
        ]

    # --- Delete (CRUD: D) ---

    def delete(self, ids: Optional[List[str]] = None, where: Optional[dict] = None):
        """
        Delete: 删除指定文档。
        支持按 ID 列表删除或按元数据条件删除。
        """
        return self.collection.delete(ids=ids, where=where)

class CollectionManager:
    def __init__(self, factory: ChromaClientFactory, embedding_model=None):
        self.factory = factory
        self.embedding = embedding_model
        # 预定义 EF
        self.qwen_ef = OllamaEmbeddingFunction(model_name="qwen", url=EMBEDDING_BASE_URL)
        self.ollama_ef = OllamaEmbeddingFunction(model_name="bge-m3:latest", url=OLLAMA_API_BASE_URL)
        self.text_proc = TextProcessor()


    def _get_ef(self):
        """策略模式：根据配置选择 Embedding 函数"""
        if RAG_EMBEDDING_MODEL == "qwen": return self.qwen_ef
        if RAG_EMBEDDING_MODEL == "bge": return self.ollama_ef
        return self.embedding

    # --- Collection CRUD ---

    def create_collection(self, name: str, tenant=chromadb.DEFAULT_TENANT, database=chromadb.DEFAULT_DATABASE):
        client = self.factory.get_client(tenant, database)
        return client.create_collection(name=name, embedding_function=self._get_ef())

    def list_collections(self, tenant=chromadb.DEFAULT_TENANT, database=chromadb.DEFAULT_DATABASE) -> List[str]:
        client = self.factory.get_client(tenant, database)
        return [c.name for c in client.list_collections()]

    def delete_collection(self, name: str, tenant=chromadb.DEFAULT_TENANT, database=chromadb.DEFAULT_DATABASE):
        client = self.factory.get_client(tenant, database)
        client.delete_collection(name)

    def get_handle(self, name: str, tenant=chromadb.DEFAULT_TENANT, database=chromadb.DEFAULT_DATABASE) -> "CollectionHandle":
        """获取句柄，如果不存在则报错或自动创建（由内部逻辑决定）"""
        client = self.factory.get_client(tenant, database)
        collection = client.get_or_create_collection(name=name, embedding_function=self._get_ef())
        return CollectionHandle(collection, self.embedding,self.text_proc)
