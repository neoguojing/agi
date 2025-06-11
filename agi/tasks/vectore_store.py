import chromadb
from chromadb import Settings
from langchain_core.documents import Document
from langchain_chroma import Chroma
from agi.utils.nlp import TextProcessor
from agi.config import log
from typing import List
import asyncio

# TODO 多租户改造
class CollectionManager:
    def __init__(self, data_path, embedding, allow_reset=True, anonymized_telemetry=False):
        self.data_path = data_path
        self.embedding = embedding
        self.allow_reset = allow_reset
        self.anonymized_telemetry = anonymized_telemetry

        self.embedding = embedding
        self.text_proc = TextProcessor()
        # db_path = f"{self.data_path}/{tenant}/{database}"
        self.adminClient = chromadb.AdminClient(Settings(
            chroma_api_impl="chromadb.api.segment.SegmentAPI",
            is_persistent=True,
            persist_directory=self.data_path,
        ))
        
    def get_or_create_tenant_for_user(self,tenant, database=chromadb.DEFAULT_DATABASE):
        try:
            self.adminClient.get_tenant(tenant)
        except Exception as e:
            self.adminClient.create_tenant(tenant)
            self.adminClient.create_database(database, tenant)
        return tenant, database

    def client(self,tenant=chromadb.DEFAULT_TENANT, database=chromadb.DEFAULT_DATABASE):
        if tenant is None:
            tenant = chromadb.DEFAULT_TENANT
        else:
            _,database = self.get_or_create_tenant_for_user(tenant)
        # db_path = f"{self.data_path}/{tenant}/{database}"
        return chromadb.PersistentClient(
            path=self.data_path,
            # settings=Settings(allow_reset=self.allow_reset, anonymized_telemetry=self.anonymized_telemetry),
            database=database,
            tenant=tenant
        )
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

    def get_documents(self, collection_name,tenant=chromadb.DEFAULT_TENANT, database=chromadb.DEFAULT_DATABASE) -> list[Document]:
        """Retrieve all documents and their metadata from the collection."""
        collection = self.get_or_create_collection(collection_name,tenant,database)
        result = collection.get(limit=3,offset=0)
        
        return [Document(page_content=document, metadata=metadata) 
                for document, metadata in zip(result['documents'], result['metadatas'])]
    
    def get_sources(self, collection_name,tenant=chromadb.DEFAULT_TENANT, database=chromadb.DEFAULT_DATABASE) -> list[str]:
        """Retrieve all sources from the collection."""
        collection = self.get_or_create_collection(collection_name,tenant=tenant,database=database)
        result = collection.get()
        
        return [ metadata["source"]
                for metadata in result['metadatas']]
    
    async def full_search(
        self,
        texts: List[str],
        collection_name: str,
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
        processed_results = await self.text_proc.batch_process(texts, method="textrank")

        # 2. 构建并发查询任务
        async def query_single(item: dict):
            keywords = [kw for kw, _ in item["keywords"]]
            query = self.build_query(contains_list=keywords)
            return collection.query(
                query_texts=[item["text"]],
                n_results=k,
                where_document=query
            )

        tasks = [query_single(item) for item in processed_results]

        # 3. 并发执行所有查询任务
        results = await asyncio.gather(*tasks)
        return results

    
    def build_query(self,contains_list=None, not_contains_list=None):
        query = {"$or": []}
        if contains_list:
            for s in contains_list:
                query["$or"].append({"$contains": s})

        if not_contains_list:
            for s in not_contains_list:
                query["$or"].append({"$not_contains": s})

        return query
