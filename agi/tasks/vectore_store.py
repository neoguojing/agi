import chromadb
from chromadb.config import Settings
from langchain_core.documents import Document
from langchain_chroma import Chroma

class CollectionManager:
    def __init__(self, data_path, embedding,tenant=chromadb.DEFAULT_TENANT, database=chromadb.DEFAULT_DATABASE, allow_reset=True, anonymized_telemetry=False):
        self.client = chromadb.PersistentClient(
            path=data_path,
            settings=Settings(allow_reset=allow_reset, anonymized_telemetry=anonymized_telemetry),
            database=database
        )
        self.embedding = embedding

    def get_or_create_collection(self, collection_name):
        """Get or create a collection by name."""
        try:
            return self.client.get_collection(name=collection_name)
        except Exception as e:
            return self.create_collection(collection_name)

    def delete_collection(self, collection_name):
        """Delete the collection by name."""
        self.client.delete_collection(name=collection_name)

    def create_collection(self, collection_name):
        """Create a new collection."""
        return self.client.create_collection(name=collection_name)

    def list_collections(self, limit=None, offset=None):
        """List collections with optional pagination."""
        return self.client.list_collections(limit, offset)
    
    def get_vector_store(self, collection_name) -> Chroma:
        """Get or create a vector store for the given collection name."""
        return Chroma(client=self.client, 
                      embedding_function=self.embedding, 
                      collection_name=collection_name)

    def get_documents(self, collection_name) -> list[Document]:
        """Retrieve all documents and their metadata from the collection."""
        collection = self.get_or_create_collection(collection_name)
        result = collection.get()
        
        return [Document(page_content=document, metadata=metadata) 
                for document, metadata in zip(result['documents'], result['metadatas'])]
    
    def get_sources(self, collection_name) -> list[str]:
        """Retrieve all sources from the collection."""
        collection = self.get_or_create_collection(collection_name)
        result = collection.get()
        
        return [ metadata["source"]
                for metadata in result['metadatas']]
