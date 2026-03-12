from qdrant_client import QdrantClient
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from agi.rag.vector_store import QdrantCustomStore


def build_store():

    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    client = QdrantClient(host="localhost", port=6333)

    store = QdrantCustomStore(
        client=client,
        collection_name="test_qdrant_store",
        embeddings=embedding
    )

    return store


def test_insert(store):

    docs = [
        Document(
            page_content="LangChain helps build LLM apps",
            metadata={"source": "langchain"}
        ),
        Document(
            page_content="Qdrant is a high performance vector database",
            metadata={"source": "qdrant"}
        ),
        Document(
            page_content="Python is widely used for AI",
            metadata={"source": "python"}
        ),
        Document(
            page_content="Tokyo is the capital of Japan",
            metadata={"source": "tokyo"}
        ),
    ]

    ids = store.add_documents(docs)

    print("Inserted IDs:", ids)

    return ids


def test_similarity_search(store):

    print("\n--- similarity_search ---")

    docs = store.similarity_search(
        query="vector database",
        k=2
    )

    for d in docs:
        print(d.page_content, d.metadata)


def test_similarity_with_score(store):

    print("\n--- similarity_search_with_score ---")

    docs = store.similarity_search_with_score(
        query="AI programming",
        k=2
    )

    for doc, score in docs:
        print(score, doc.page_content)


def test_mmr(store):

    print("\n--- MMR search ---")

    docs = store.max_marginal_relevance_search(
        query="programming language",
        k=2,
        fetch_k=4
    )

    for d in docs:
        print(d.page_content)


def test_filter(store):

    print("\n--- filter search ---")

    docs = store.similarity_search(
        query="database",
        k=3,
        filter={"source": "qdrant"}
    )

    for d in docs:
        print(d.page_content, d.metadata)


def test_get_by_ids(store, ids):

    print("\n--- get_by_ids ---")

    docs = store.get_by_ids(ids[:2])

    for d in docs:
        print(d.page_content)


def test_delete(store, ids):

    print("\n--- delete ---")

    store.delete(ids[:1])

    docs = store.get_by_ids(ids[:1])

    print("After delete:", docs)


def main():

    store = build_store()

    ids = test_insert(store)

    test_similarity_search(store)

    test_similarity_with_score(store)

    test_mmr(store)

    test_filter(store)

    test_get_by_ids(store, ids)

    test_delete(store, ids)


if __name__ == "__main__":
    main()