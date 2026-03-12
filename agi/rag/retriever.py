import threading
from typing import List, Optional,Dict
from qdrant_client import QdrantClient

from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    SimpleDirectoryReader,
    Document,
)

from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.retrievers import QueryFusionRetriever
from langchain_core.documents import Document as LangchainDocument




class QdrantRAGManager:

    def __init__(
        self,
        collection_name: str,
        qdrant_url: str = "http://localhost:6333",
        ollama_embedding_model: Optional[str] = None,
        ollama_base_url: str = "http://localhost:11434",
        embed_model=None,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
    ):
        """
        Parameters
        ----------
        collection_name : Qdrant collection
        qdrant_url : Qdrant server url
        ollama_embedding_model : 使用 Ollama embedding
        embed_model : 自定义 embedding
        """

        self._lock = threading.Lock()

        # -----------------------------
        # Qdrant Client
        # -----------------------------

        self.client = QdrantClient(url=qdrant_url)

        self.vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=collection_name,
        )

        self.storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store
        )

        # -----------------------------
        # Embedding Model
        # -----------------------------

        if embed_model:
            self.embed_model = embed_model

        elif ollama_embedding_model:
            self.embed_model = OllamaEmbedding(
                model_name=ollama_embedding_model,
                base_url=ollama_base_url,
            )

        else:
            raise ValueError(
                "必须提供 embed_model 或 ollama_embedding_model"
            )

        # -----------------------------
        # Node Parser
        # -----------------------------

        self.node_parser = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        self._index: Optional[VectorStoreIndex] = None

    # --------------------------------
    # 初始化 Index
    # --------------------------------

    def _ensure_index(self):

        if self._index is None:

            self._index = VectorStoreIndex.from_vector_store(
                vector_store=self.vector_store,
                storage_context=self.storage_context,
                embed_model=self.embed_model,
            )

    # --------------------------------
    # 文本入库
    # --------------------------------

    def ingest_text(
        self,
        text: str,
        metadata: Optional[dict] = None,
    ):

        doc = Document(
            text=text,
            metadata=metadata or {},
        )

        self._add_to_index([doc])

    # --------------------------------
    # 文件入库
    # --------------------------------

    def ingest_files(self, file_paths: List[str]):

        reader = SimpleDirectoryReader(
            input_files=file_paths
        )

        documents = reader.load_data()

        self._add_to_index(documents)

    # --------------------------------
    # 目录入库
    # --------------------------------

    def ingest_directory(self, path: str):

        reader = SimpleDirectoryReader(
            input_dir=path,
            recursive=True,
        )

        documents = reader.load_data()

        self._add_to_index(documents)

    # --------------------------------
    # 入库核心逻辑
    # --------------------------------

    def _add_to_index(self, documents: List[Document]):

        if not documents:
            return

        self._ensure_index()

        nodes = self.node_parser.get_nodes_from_documents(
            documents
        )

        with self._lock:
            self._index.insert_nodes(nodes)

        print(f"✅ 成功入库 {len(nodes)} 个节点")

    # --------------------------------
    # Query Engine
    # --------------------------------

    def get_query_engine(
        self,
        mode: str = "default",
        top_k: int = 5,
        alpha: float = 0.5,
    ):

        self._ensure_index()

        query_mode = "default"

        if mode == "hybrid":
            query_mode = "hybrid"

        query_engine = self._index.as_query_engine(
            similarity_top_k=top_k,
            vector_store_query_mode=query_mode,
            alpha=alpha,
        )

        return query_engine

    # --------------------------------
    # 查询
    # --------------------------------

    def query(
        self,
        question: str,
        mode: str = "default",
        top_k: int = 5,
    ):

        engine = self.get_query_engine(
            mode=mode,
            top_k=top_k,
        )

        response = engine.query(question)

        docs = []

        for node in response.source_nodes:
            docs.append(
                LangchainDocument(
                    page_content=node.node.text,
                    metadata=node.node.metadata or {},
                )
            )

        return docs

    # --------------------------------
    # Retriever
    # --------------------------------

    def get_retriever(
        self,
        mode: str = "default",
        top_k: int = 5,
    ):

        self._ensure_index()

        query_mode = "default"

        if mode == "hybrid":
            query_mode = "hybrid"

        retriever = self._index.as_retriever(
            similarity_top_k=top_k,
            vector_store_query_mode=query_mode,
        )

        return retriever
    

class MultiCollectionRAGManager:

    def __init__(
        self,
        qdrant_url: str = "http://localhost:6333",
        ollama_embedding_model: Optional[str] = "bge:m3",
        ollama_base_url: str = "http://localhost:11434",
        embed_model=None,
    ):

        self.qdrant_url = qdrant_url
        self.ollama_embedding_model = ollama_embedding_model
        self.ollama_base_url = ollama_base_url
        self.embed_model = embed_model

        self._collections: Dict[str, QdrantRAGManager] = {}

    # -----------------------------
    # 获取 collection manager
    # -----------------------------

    def get_collection(self, collection_name: str) -> QdrantRAGManager:

        if collection_name not in self._collections:

            self._collections[collection_name] = QdrantRAGManager(
                collection_name=collection_name,
                qdrant_url=self.qdrant_url,
                ollama_embedding_model=self.ollama_embedding_model,
                ollama_base_url=self.ollama_base_url,
                embed_model=self.embed_model,
            )

        return self._collections[collection_name]

    # -----------------------------
    # 文本入库
    # -----------------------------

    def ingest_text(
        self,
        collection_name: str,
        text: str,
        metadata: Optional[dict] = None,
    ):

        manager = self.get_collection(collection_name)

        manager.ingest_text(
            text=text,
            metadata=metadata,
        )

    # -----------------------------
    # 文件入库
    # -----------------------------

    def ingest_files(
        self,
        collection_name: str,
        files,
    ):

        manager = self.get_collection(collection_name)

        manager.ingest_files(files)

    # -----------------------------
    # 查询
    # -----------------------------

    def query(
        self,
        collection_name: str,
        question: str,
        mode: str = "default",
        top_k: int = 5,
    ):

        manager = self.get_collection(collection_name)

        return manager.query(
            question=question,
            mode=mode,
            top_k=top_k,
        )

    # -----------------------------
    # 删除collection
    # -----------------------------

    def drop_collection(self, collection_name: str):

        if collection_name in self._collections:

            manager = self._collections[collection_name]

            manager.client.delete_collection(
                collection_name
            )

            del self._collections[collection_name]


    def multi_query(
        self,
        collections: list,
        question: str,
        top_k: int = 5,
    ):

        retrievers = []

        for c in collections:

            retriever = self.get_collection(c).get_retriever(
                top_k=top_k
            )

            retrievers.append(retriever)

        fusion = QueryFusionRetriever(
            retrievers=retrievers,
            similarity_top_k=top_k,
            num_queries=1,   # 不做query expansion
            mode="reciprocal_rerank",
        )

        nodes = fusion.retrieve(question)

        return nodes