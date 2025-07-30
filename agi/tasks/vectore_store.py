import chromadb
from chromadb import Settings
from langchain_core.documents import Document
from langchain_chroma import Chroma
from agi.utils.nlp import TextProcessor
from agi.config import log
from typing import List
import asyncio
import uuid
import math
import os
from tqdm import tqdm  # 可选：用于进度显示
from threading import Lock
class CollectionManager:
    def __init__(self, data_path, embedding, allow_reset=True, anonymized_telemetry=False):
        self.data_path = data_path
        self.embedding = embedding
        self.allow_reset = allow_reset
        self.anonymized_telemetry = anonymized_telemetry

        self.text_proc = TextProcessor()

        self.settings = Settings(
            chroma_api_impl="chromadb.api.segment.SegmentAPI",
            is_persistent=True,
            persist_directory=self.data_path,
            allow_reset=self.allow_reset,
            anonymized_telemetry=self.anonymized_telemetry,
        )

        self.admin_client = chromadb.AdminClient(self.settings)

        self._client_lock = Lock()
        self._persistent_client = None
        
    def get_or_create_tenant_for_user(self,tenant, database=chromadb.DEFAULT_DATABASE):
        try:
            self.admin_client.get_tenant(tenant)
        except Exception as e:
            self.admin_client.create_tenant(tenant)
            self.admin_client.create_database(database, tenant)
        return tenant, database
    
    def _get_persistent_client(self, tenant, database):
        with self._client_lock:
            if self._persistent_client is None:
                self._persistent_client = chromadb.PersistentClient(
                    path=self.data_path,
                    settings=self.settings,
                    tenant=tenant,
                    database=database,
                )
            else:
                # 仅切换 tenant/database 无需新建 client 实例
                # 因为 SharedSystemClient 会缓存，不允许不同设置
                self._persistent_client.set_tenant(tenant)
                self._persistent_client.set_database(database)
            return self._persistent_client

    def client(self,tenant=chromadb.DEFAULT_TENANT, database=chromadb.DEFAULT_DATABASE):
        if tenant is None:
            tenant = chromadb.DEFAULT_TENANT
        else:
            _,database = self.get_or_create_tenant_for_user(tenant,database)
        return self._get_persistent_client(tenant, database)
    
    def get_or_create_collection(self, collection_name,tenant=chromadb.DEFAULT_TENANT, database=chromadb.DEFAULT_DATABASE):
        """Get or create a collection by name."""
        return self.client(tenant,database).get_or_create_collection(name=collection_name)
      

    def delete_collection(self, collection_name,tenant=chromadb.DEFAULT_TENANT, database=chromadb.DEFAULT_DATABASE):
        """Delete the collection by name."""
        self.client(tenant,database).delete_collection(name=collection_name)

    def list_collections(self, limit=None, offset=None,tenant=chromadb.DEFAULT_TENANT, database=chromadb.DEFAULT_DATABASE):
        """List collections with optional pagination."""
        collections = self.client(tenant,database).list_collections(limit, offset)
        if collections is None or len(collections) == 0:
            collection = self.get_or_create_collection("default",tenant,database)
            collections = [collection]

        return collections
    
    def get_vector_store(self, collection_name,tenant=chromadb.DEFAULT_TENANT, database=chromadb.DEFAULT_DATABASE) -> Chroma:
        """Get or create a vector store for the given collection name."""
        self.get_or_create_collection(collection_name,tenant=tenant,database=database)
        return Chroma(client=self.client(tenant,database), 
                      embedding_function=self.embedding, 
                      collection_name=collection_name)

    def get_documents(self, collection_name,source=None,limit=10,offset=0,tenant=chromadb.DEFAULT_TENANT, database=chromadb.DEFAULT_DATABASE) -> list[Document]:
        """Retrieve all documents and their metadata from the collection."""
        collection = self.get_or_create_collection(collection_name,tenant,database)
        where_cond = None
        if source:
            where_cond = {"source": os.path.basename(source)}
        result = collection.get(limit=limit,offset=offset,where=where_cond)
        
        return [Document(page_content=document, metadata=metadata) 
                for document, metadata in zip(result['documents'], result['metadatas'])]
    
    def get_sources(self, collection_name,tenant=chromadb.DEFAULT_TENANT, database=chromadb.DEFAULT_DATABASE) -> list[str]:
        """Retrieve all sources from the collection."""
        collection = self.get_or_create_collection(collection_name,tenant=tenant,database=database)
        result = collection.get()
        
        return [ metadata["source"]
                for metadata in result['metadatas']]
    
    async def add_documents(
        self,
        documents: List[Document],
        collection_name: str,
        embeddings: List[List[float]] =None,
        ids :List[str] = None,
        batch_size: int = 10,
        tenant=chromadb.DEFAULT_TENANT,
        database=chromadb.DEFAULT_DATABASE,
    ):
        """异步批量添加 Document 对象到 Chroma Collection"""
        collection = self.get_or_create_collection(
            collection_name,
            tenant=tenant,
            database=database
        )
        texts = [doc.page_content for doc in documents]
        metadatas = [
            {
                **doc.metadata,
                "keywords": ",".join([f"{kw}:{round(score, 3)}" for kw, score in doc.metadata.get("keywords", [])]),
                "related_docs": ",".join([docid for docid in doc.metadata.get("related_docs", [])])
            }
            for doc in documents
        ]

        if ids is None:
            ids = [doc.metadata.get("doc_id") for doc in documents]
            if ids is None:
                ids = [str(uuid.uuid4()) for _ in documents]  # 每条文档生成唯一 id

        log.info(f"texts={len(texts)},metadatas={len(metadatas)},ids={len(ids)}")
        # 分批添加到 Chroma Collection
        num_batches = math.ceil(len(documents) / batch_size)
        for i in tqdm(range(num_batches), desc="Adding document batches"):
            start = i * batch_size
            end = start + batch_size
            try:
                # 异步并发生成嵌入（嵌套列表需要解包）
                if embeddings is None:
                    embeddings = await asyncio.gather(*[
                        asyncio.to_thread(self.embedding.embed_query, text)
                        for text in texts[start:end]
                    ])
                collection.add(
                    documents=texts[start:end],
                    embeddings=embeddings,
                    metadatas=metadatas[start:end],
                    ids=ids[start:end]
                )
            except Exception as e:
                log.error(f"Failed to add batch {i + 1}: {e}")
    
    async def embedding_search(
        self,
        texts: List[str],
        collection_name: str,
        k: int = 10,
        cluster_id: str = None,
        tenant=chromadb.DEFAULT_TENANT,
        database=chromadb.DEFAULT_DATABASE
    ):
        collection = self.get_or_create_collection(
            collection_name,
            tenant=tenant,
            database=database
        )

        async def query_single(text: str):
            embedding = self.embedding.embed_query(text)
            where_cond = None
            if cluster_id:
                where_cond = {"cluster_id":cluster_id}
            log.info(f"text: {text} where:{where_cond}")
            return collection.query(
                query_embeddings=[embedding],
                n_results=k,
                where=where_cond,         
            )

        tasks = [query_single(text) for text in texts]

        # 3. 并发执行所有查询任务
        results = await asyncio.gather(*tasks)
        ret = []
        for result in results:
            for docs, metas, scores in zip(result['documents'], result['metadatas'], result['distances']):
                for document, metadata, score in zip(docs, metas, scores):
                    metadata['score'] = score
                    ret.append(Document(page_content=document, metadata=metadata))
        return ret
    
    async def full_search(
        self,
        texts: List[str],
        collection_name: str,
        cluster_id: str = None,
        k: int = 10,
        tenant=chromadb.DEFAULT_TENANT,
        database=chromadb.DEFAULT_DATABASE
    ):
        collection = self.get_or_create_collection(
            collection_name,
            tenant=tenant,
            database=database
        )

        # 1. 异步批量关键词提取
        processed_results = await self.text_proc.abatch_process(texts, method="textrank")

        # 2. 构建并发查询任务
        async def query_single(text: str, keywords: list):
            keywords = [kw[0] for kw in keywords]
            query = self.build_query(contains_list=keywords)
            where_cond = None
            if cluster_id:
                where_cond = {"cluster_id":cluster_id}
            log.info(f"text: {text} query_single：{query},where:{where_cond}")
            return collection.query(
                query_texts=[text],
                n_results=k,
                where_document=query,
                where=where_cond,         
            )

        tasks = [query_single(texts[i], processed_results[i]) for i in range(len(texts))]

        # 3. 并发执行所有查询任务
        results = await asyncio.gather(*tasks)
        ret = []
        for result in results:
            for docs, metas, scores in zip(result['documents'], result['metadatas'], result['distances']):
                for document, metadata, score in zip(docs, metas, scores):
                    metadata['score'] = score
                    ret.append(Document(page_content=document, metadata=metadata))
        return ret

    
    def build_query(self,contains_list=None, not_contains_list=None):
        query_or = []
        if contains_list:
            for s in contains_list:
                query_or.append({"$contains": s})

        if not_contains_list:
            for s in not_contains_list:
                query_or.append({"$not_contains": s})

        return {"$or": query_or} if query_or else None

