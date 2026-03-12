import uuid
import asyncio
from typing import Any, Iterable, List, Optional, Type,Sequence,Tuple,Union
import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import (
    PointStruct, VectorParams, Distance, Filter,PayloadSchemaType, 
    FieldCondition, MatchValue,PointIdsList,ScoredPoint, Record
)

class QdrantCustomStore(VectorStore):
    def __init__(
        self,
        client: QdrantClient,
        collection_name: str,
        embeddings: Embeddings,
    ):
        self._client = client
        self._collection_name = collection_name
        self._embeddings = embeddings

        self._ensure_collection()

    @property
    def embeddings(self) -> Embeddings:
        return self._embeddings

    def _ensure_collection(self):
        if self._client.collection_exists(self._collection_name):
            return

        # 1. 获取向量维度（仅在创建时调用一次）
        dim = len(self._embeddings.embed_query("test"))

        # 2. 创建集合
        self._client.create_collection(
            collection_name=self._collection_name,
            vectors_config=VectorParams(
                size=dim,
                distance=Distance.COSINE
            )
        )

        # 3. 创建常用的 Payload 索引（提升过滤性能）
        # 为 doc_id 创建索引（用于精确匹配和去重）
        self._client.create_payload_index(
            collection_name=self._collection_name,
            field_name="doc_id",
            field_schema=PayloadSchemaType.KEYWORD
        )

        # 为常见的 'source' 字段创建索引（RAG 中常用的过滤维度）
        self._client.create_payload_index(
            collection_name=self._collection_name,
            field_name="source",
            field_schema=PayloadSchemaType.KEYWORD
        )

    def _point_to_document(self, point: Union[ScoredPoint, Record]) -> Document:
        """
        统一转换函数：将 Qdrant 的点对象转换为 LangChain Document。
        
        Args:
            point: Qdrant 搜索返回的 ScoredPoint 或 Record 对象。
        """
        payload = point.payload or {}
        
        # 1. 提取核心内容，同时确保不破坏原 payload 字典
        page_content = payload.get("text", "")
        
        # 2. 构造元数据：过滤掉内部使用的 text 字段
        metadata = {k: v for k, v in payload.items() if k != "text"}
        
        # 3. 注入系统级信息
        metadata["_id"] = point.id
        if hasattr(point, "score"):
            metadata["_score"] = point.score
            
        return Document(page_content=page_content, metadata=metadata)
    
    # --- 核心写入方法 ---

    def _run_in_batches(self, items: List[Any], batch_size: int):
        """通用分批生成器"""
        for i in range(0, len(items), batch_size):
            yield items[i : i + batch_size]

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        *,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        list_texts = list(texts)
        n = len(list_texts)
        pids = ids or [str(uuid.uuid4()) for _ in range(n)]
        metadatas = metadatas or [{} for _ in range(n)]

        # 1. 分批处理 Embedding 
        # 解决 API 超时和 Payload 过大问题
        embedding_batch_size = kwargs.get("embedding_batch_size", 32)
        all_embeddings = []
        
        for batch in self._run_in_batches(list_texts, embedding_batch_size):
            # 这里可以轻松加入重试逻辑 (retry)
            batch_encodings = self._embeddings.embed_documents(batch)
            all_embeddings.extend(batch_encodings)

        # 2. 组装数据
        points = []
        for text, vector, metadata, point_id in zip(
            list_texts, all_embeddings, metadatas, pids
        ):
            # 修复 1: 非破坏性提取 payload
            payload = {k: v for k, v in metadata.items() if k != "text"}
            payload["text"] = text
            
            points.append(PointStruct(id=point_id, vector=vector, payload=payload))

        # 3. 分批写入 Qdrant
        # 解决 Qdrant 写入 body 大小限制问题
        qdrant_batch_size = kwargs.get("qdrant_upload_batch_size", 100)
        for batch_points in self._run_in_batches(points, qdrant_batch_size):
            self._client.upsert(
                collection_name=self._collection_name,
                points=batch_points
            )

        return pids

    # --- 核心搜索方法 ---

    def _build_filter(self, filter: Optional[Union[dict, Filter]]) -> Optional[Filter]:
        if filter is None:
            return None

        if isinstance(filter, Filter):
            return filter

        if isinstance(filter, dict):
            conditions = [
                FieldCondition(
                    key=k,
                    match=MatchValue(value=v)
                )
                for k, v in filter.items()
            ]

            return Filter(must=conditions)

        raise ValueError("Unsupported filter type")

    def _search(
        self,
        vector: List[float],
        k: int,
        filter: Optional[Union[dict, Filter]] = None,
        with_vectors: bool = False,
        **kwargs
    ):
        # 确保 kwargs 中不包含已显式定义的参数
        kwargs.pop("collection_name", None)
        kwargs.pop("query_vector", None)
        kwargs.pop("query_filter", None)
        kwargs.pop("limit", None)

        filter = self._build_filter(filter)
        
        return self._client.search(
            collection_name=self._collection_name,
            query_vector=vector,
            query_filter=filter,
            limit=k,
            with_vectors=with_vectors,
            **kwargs
        )

    def search(
        self,
        query: str,
        search_type: str,
        **kwargs: Any,
    ) -> List[Document]:
        """
        根据指定的 search_type 执行搜索。
        
        Args:
            query: 查询字符串。
            search_type: 搜索类型，支持 "similarity" 或 "mmr"。
            **kwargs: 传递给具体搜索方法的参数（如 k, fetch_k, lambda_mult, filter）。
            
        Returns:
            List[Document]: 检索到的文档列表。
        """
        if search_type == "similarity":
            return self.similarity_search(query, **kwargs)
        
        elif search_type == "mmr":
            # 如果你实现了 max_marginal_relevance_search
            return self.max_marginal_relevance_search(query, **kwargs)
        
        else:
            raise ValueError(
                f"不支持的 search_type: {search_type}. "
                f"当前支持: 'similarity', 'mmr'"
            )

    def similarity_search(
        self, query: str, k: int = 4, filter: Optional[dict] = None, **kwargs: Any
    ) -> List[Document]:
        query_vector = self._embeddings.embed_query(query)
        
        results = self._search(vector=query_vector,k=k,filter=filter,**kwargs)

        return [self._point_to_document(r) for r in results]

    # -----------------------------------------------------
    # Similarity Search
    # -----------------------------------------------------

    def similarity_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        **kwargs: Any
    ) -> List[Document]:
        # 修复 2: 弹出已被显式定义的参数，防止 **kwargs 冲突
        search_filter = kwargs.pop("filter", None)
        # 允许用户通过 kwargs 传递 search_params (如 hnsw_ef)
        

        results = self._search(vector= embedding, k=k, filter=search_filter,**kwargs)

        return [self._point_to_document(r) for r in results]
    
    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        """返回文档及其归一化后的分数 [0, 1]"""
        embedding = self._embeddings.embed_query(query)
        # 修复 1 & 2: 使用 pop 避免冲突，并处理 filter 类型
        search_filter = kwargs.pop("filter", None)
        
        results = self._search(
            vector=embedding, 
            k=k, 
            filter=search_filter, 
            **kwargs
        )
        # 归一化逻辑：Cosine 距离下，Qdrant 返回的是相似度
        # 通常相似度 score = (score + 1) / 2 或直接使用 score 取决于 Qdrant 配置
        return [
            (self._point_to_document(r), r.score)
            for r in results
        ]

    def similarity_search_with_relevance_scores(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        """LangChain 标准接口，要求分数必须在 [0, 1] 之间"""
        # 这里可以直接复用 similarity_search_with_score
        # 如果 Qdrant 配置为 COSINE，分数已经在相似度区间
        return self.similarity_search_with_score(query, k, **kwargs)

    # -----------------------------------------------------
    # MMR (Max Marginal Relevance)
    # -----------------------------------------------------

    def max_marginal_relevance_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        **kwargs: Any
    ) -> List[Document]:
        search_filter = kwargs.pop("filter", None)

        results = self._search(
            vector=embedding, 
            k=fetch_k, 
            filter=search_filter, 
            with_vectors=True, 
            **kwargs
        )

        if not results:
            return []

        # 修复 4: 兼容 Named Vectors (dict) 和普通 Vector (list)
        embeddings = []
        for r in results:
            if isinstance(r.vector, list):
                embeddings.append(r.vector)
            elif isinstance(r.vector, dict):
                # 取第一个命名向量，或根据业务逻辑取特定 key
                embeddings.append(list(r.vector.values())[0])
            else:
                raise ValueError(f"Unexpected vector format: {type(r.vector)}")

        from langchain_core.vectorstores import utils as vectorstore_utils
        mmr_selected = vectorstore_utils.maximal_marginal_relevance(
            np.array(embedding, dtype=np.float32),
            embeddings,
            k=k,
            lambda_mult=lambda_mult
        )

        # 同样应用修复 1 的非破坏性 payload 提取
        return [self._point_to_document(results[i]) for i in mmr_selected]

    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        **kwargs: Any
    ) -> List[Document]:
        embedding = self._embeddings.embed_query(query)
        return self.max_marginal_relevance_search_by_vector(
            embedding, k, fetch_k, lambda_mult, **kwargs
        )

    # -----------------------------------------------------
    # Factory Methods
    # -----------------------------------------------------

    @classmethod
    def from_documents(
        cls,
        documents: List[Document],
        embedding: Embeddings,
        **kwargs: Any
    ) -> "QdrantCustomStore":
        """工厂方法：从文档直接初始化 Store"""
        host = kwargs.get("host", "localhost")
        port = kwargs.get("port", 6333)
        collection_name = kwargs.get("collection_name", str(uuid.uuid4()))
        
        client = QdrantClient(host=host, port=port)
        
        # 预先获取向量维度
        sample_vec = embedding.embed_query("test")
        
        client.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=len(sample_vec), distance=Distance.COSINE)
        )
        
        store = cls(client, collection_name, embedding)
        store.add_documents(documents)
        return store
    # --- 类方法：初始化 ---

    @classmethod
    def from_texts(
        cls: Type["QdrantCustomStore"],
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        collection_name: str = "langchain_collection",
        host: str = "localhost",
        apiKey: Optional[str] = None,
        **kwargs: Any,
    ) -> "QdrantCustomStore":
        client = QdrantClient(host=host, api_key=apiKey)
        
        # 简单演示：自动创建 collection
        vector_size = len(embedding.embed_query("test"))
        client.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
        )
        
        instance = cls(client, collection_name, embedding)
        instance.add_texts(texts, metadatas=metadatas)
        return instance

    def delete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> Optional[bool]:
        if not ids:
            return False
            
        self._client.delete(
            collection_name=self._collection_name,
            # 修复 3: 使用 PointIdsList 包装器
            points_selector=PointIdsList(points=ids)
        )
        return True

    def get_by_ids(
        self,
        ids: Sequence[str],
        /,
    ) -> List[Document]:
        """
        根据 ID 列表直接从 Qdrant 获取文档。
        
        Args:
            ids: 要检索的文档 ID 序列（位置参数）。
            
        Returns:
            List[Document]: 对应的 Document 对象列表。
        """
        if not ids:
            return []

        # Qdrant 的 retrieve 接口可以直接根据 ID 列表获取点
        results = self._client.retrieve(
            collection_name=self._collection_name,
            ids=list(ids),
            with_payload=True,
            with_vectors=False  # 通常 get_by_ids 不需要返回向量，节省带宽
        )
        
        return [self._point_to_document(r) for r in results]
    

    def add_documents(
        self,
        documents: List[Document],
        **kwargs: Any,
    ) -> List[str]:
        """
        将 Document 对象列表添加到向量数据库中。
        
        Args:
            documents: 要添加的 Document 对象列表。
            **kwargs: 额外参数。支持 'ids' 用于指定 ID 列表。
            
        Returns:
            List[str]: 成功插入的 ID 列表。
        """
        # 1. 提取 ID（优先从 kwargs 获取，其次尝试从 Document metadata 获取）
        ids = kwargs.get("ids")
        if ids is None:
            # 尝试从 metadata 中寻找预设的 doc_id
            ids = [doc.metadata.get("doc_id") for doc in documents]
            # 如果 metadata 中也不全，则由 add_texts 统一生成 UUID
            if any(id is None for id in ids):
                ids = None

        # 2. 提取文本和元数据
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]

        # 3. 委派给 add_texts 处理向量化和上传
        return self.add_texts(
            texts=texts,
            metadatas=metadatas,
            ids=ids,
            **kwargs
        )