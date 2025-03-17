import chromadb
from chromadb.config import Settings
from langchain_core.documents import Document
from langchain_chroma import Chroma

# TODO 多租户改造
class CollectionManager:
    def __init__(self, data_path, embedding, allow_reset=True, anonymized_telemetry=False):
        self.data_path = data_path
        self.embedding = embedding
        self.allow_reset = allow_reset
        self.anonymized_telemetry = anonymized_telemetry

        self.embedding = embedding

    def client(self,tenant=chromadb.DEFAULT_TENANT, database=chromadb.DEFAULT_DATABASE):
        if tenant is None:
            tenant = chromadb.DEFAULT_TENANT
        return chromadb.PersistentClient(
            path=self.data_path,
            settings=Settings(allow_reset=self.allow_reset, anonymized_telemetry=self.anonymized_telemetry),
            database=database,
            tenant=tenant
        )
    def get_or_create_collection(self, collection_name,tenant=chromadb.DEFAULT_TENANT, database=chromadb.DEFAULT_DATABASE):
        """Get or create a collection by name."""
        try:
            return self.client(tenant,database).get_collection(name=collection_name)
        except Exception as e:
            return self.create_collection(collection_name,tenant,database)

    def delete_collection(self, collection_name,tenant=chromadb.DEFAULT_TENANT, database=chromadb.DEFAULT_DATABASE):
        """Delete the collection by name."""
        self.client(tenant,database).delete_collection(name=collection_name)

    def create_collection(self, collection_name,tenant=chromadb.DEFAULT_TENANT, database=chromadb.DEFAULT_DATABASE):
        """Create a new collection."""
        return self.client(tenant,database).create_collection(name=collection_name)

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
        result = collection.get()
        
        return [Document(page_content=document, metadata=metadata) 
                for document, metadata in zip(result['documents'], result['metadatas'])]
    
    def get_sources(self, collection_name,tenant=chromadb.DEFAULT_TENANT, database=chromadb.DEFAULT_DATABASE) -> list[str]:
        """Retrieve all sources from the collection."""
        collection = self.get_or_create_collection(collection_name,tenant=tenant,database=database)
        result = collection.get()
        
        return [ metadata["source"]
                for metadata in result['metadatas']]