import asyncio
import uuid
from typing import List, Dict, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import (
    PointStruct,
    VectorParams,
    Distance,
    Filter,
    FieldCondition,
    MatchValue,
    MatchAny
)
from langchain_core.documents import Document


# =========================================================
# Qdrant Client Factory
# =========================================================

class QdrantClientFactory:
    """
    Qdrant 客户端工厂
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        api_key: Optional[str] = None
    ):
        self.client = QdrantClient(
            host=host,
            port=port,
            api_key=api_key
        )

    def get_client(self) -> QdrantClient:
        return self.client


# =========================================================
# Collection Manager
# =========================================================

class CollectionManager:

    def __init__(
        self,
        factory: QdrantClientFactory,
        embedding_model,
        vector_size: int = 1024
    ):
        self.factory = factory
        self.embedding = embedding_model
        self.vector_size = vector_size

    def _collection_name(self, name, tenant, database):
        return f"{tenant}_{database}_{name}"

    def create_collection(
        self,
        name: str,
        tenant: str = "default",
        database: str = "default"
    ):

        client = self.factory.get_client()

        cname = self._collection_name(name, tenant, database)

        exists = [c.name for c in client.get_collections().collections]

        if cname not in exists:

            client.create_collection(
                collection_name=cname,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE
                )
            )

        return cname

    def list_collections(self):

        client = self.factory.get_client()

        return [
            c.name
            for c in client.get_collections().collections
        ]

    def delete_collection(self, name):

        client = self.factory.get_client()

        client.delete_collection(name)

    def get_handle(
        self,
        name: str,
        tenant="default",
        database="default"
    ):

        cname = self.create_collection(name, tenant, database)

        return CollectionHandle(
            self.factory.get_client(),
            cname,
            self.embedding
        )


# =========================================================
# Collection Handle
# =========================================================

class CollectionHandle:

    def __init__(
        self,
        client: QdrantClient,
        collection_name: str,
        embedding_model
    ):

        self.client = client
        self.collection = collection_name
        self.embedding = embedding_model

    # -----------------------------------------------------
    # Upsert
    # -----------------------------------------------------

    async def upsert(
        self,
        documents: List[Document],
        batch_size: int = 64
    ):

        if not documents:
            return

        texts = [doc.page_content for doc in documents]

        embeddings = await asyncio.gather(*[
            asyncio.to_thread(
                self.embedding.embed_query,
                text
            )
            for text in texts
        ])

        points = []

        for doc, vec in zip(documents, embeddings):

            pid = doc.metadata.get("doc_id") or str(uuid.uuid4())

            payload = {
                "text": doc.page_content,
                **doc.metadata
            }

            points.append(
                PointStruct(
                    id=pid,
                    vector=vec,
                    payload=payload
                )
            )

        for i in range(0, len(points), batch_size):

            self.client.upsert(
                collection_name=self.collection,
                points=points[i:i + batch_size]
            )

    # -----------------------------------------------------
    # Vector Search
    # -----------------------------------------------------

    async def search(
        self,
        query: str,
        k: int = 10
    ) -> List[Document]:

        emb = await asyncio.to_thread(
            self.embedding.embed_query,
            query
        )

        results = self.client.search(
            collection_name=self.collection,
            query_vector=emb,
            limit=k
        )

        docs = []

        for r in results:

            docs.append(
                Document(
                    page_content=r.payload.get("text"),
                    metadata={
                        **r.payload,
                        "score": r.score
                    }
                )
            )

        return docs

    # -----------------------------------------------------
    # Metadata Filter Search
    # -----------------------------------------------------

    async def search_with_filter(
        self,
        query: str,
        source: Optional[str] = None,
        k: int = 10
    ):

        emb = await asyncio.to_thread(
            self.embedding.embed_query,
            query
        )

        query_filter = None

        if source:

            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="source",
                        match=MatchValue(value=source)
                    )
                ]
            )

        results = self.client.search(
            collection_name=self.collection,
            query_vector=emb,
            query_filter=query_filter,
            limit=k
        )

        docs = []

        for r in results:

            docs.append(
                Document(
                    page_content=r.payload.get("text"),
                    metadata={
                        **r.payload,
                        "score": r.score
                    }
                )
            )

        return docs

    # -----------------------------------------------------
    # Hybrid Search
    # -----------------------------------------------------

    async def hybrid_search(
        self,
        query: str,
        keywords: Optional[List[str]] = None,
        k: int = 10
    ):

        emb = await asyncio.to_thread(
            self.embedding.embed_query,
            query
        )

        query_filter = None

        if keywords:

            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="keywords",
                        match=MatchAny(any=keywords)
                    )
                ]
            )

        results = self.client.search(
            collection_name=self.collection,
            query_vector=emb,
            query_filter=query_filter,
            limit=k
        )

        docs = []

        for r in results:

            docs.append(
                Document(
                    page_content=r.payload.get("text"),
                    metadata={
                        **r.payload,
                        "score": r.score
                    }
                )
            )

        return docs

    # -----------------------------------------------------
    # Get Documents
    # -----------------------------------------------------

    def get_documents(
        self,
        limit: int = 10
    ):

        results = self.client.scroll(
            collection_name=self.collection,
            limit=limit
        )

        docs = []

        for r in results[0]:

            docs.append(
                Document(
                    page_content=r.payload.get("text"),
                    metadata=r.payload
                )
            )

        return docs

    # -----------------------------------------------------
    # Delete
    # -----------------------------------------------------

    def delete(
        self,
        ids: Optional[List[str]] = None
    ):

        if ids:

            self.client.delete(
                collection_name=self.collection,
                points_selector=ids
            )