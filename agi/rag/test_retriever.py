import pytest
import tempfile
import os

from agi.rag.retriever import QdrantRAGManager


@pytest.fixture(scope="module")
def rag():

    manager = QdrantRAGManager(
        collection_name="test_rag_collection",
        ollama_embedding_model="nomic-embed-text"
    )

    return manager


# --------------------------------
# 初始化测试
# --------------------------------

def test_init(rag):

    assert rag.client is not None
    assert rag.vector_store is not None
    assert rag.embed_model is not None


# --------------------------------
# 文本入库测试
# --------------------------------

def test_ingest_text(rag):

    rag.ingest_text(
        "Qdrant is a high performance vector database",
        metadata={"source": "tech"}
    )

    rag.ingest_text(
        "Python is a programming language",
        metadata={"source": "programming"}
    )

    assert True


# --------------------------------
# 查询测试
# --------------------------------

def test_query(rag):

    result = rag.query("What is Qdrant?")

    assert result is not None
    assert len(str(result)) > 0


# --------------------------------
# Hybrid 查询
# --------------------------------

def test_hybrid_query(rag):

    result = rag.query(
        "vector database",
        mode="hybrid"
    )

    assert result is not None


# --------------------------------
# Metadata filter
# --------------------------------

def test_metadata_filter(rag):

    result = rag.query_with_filter(
        question="programming language",
        filter_dict={"source": "programming"}
    )

    assert result is not None


# --------------------------------
# 文件入库测试
# --------------------------------

def test_ingest_files(rag):

    with tempfile.TemporaryDirectory() as tmp:

        file_path = os.path.join(tmp, "test.txt")

        with open(file_path, "w") as f:
            f.write("LlamaIndex is a framework for RAG systems")

        rag.ingest_files([file_path])

    assert True


# --------------------------------
# 目录入库测试
# --------------------------------

def test_ingest_directory(rag):

    with tempfile.TemporaryDirectory() as tmp:

        for i in range(3):

            with open(os.path.join(tmp, f"doc{i}.txt"), "w") as f:
                f.write(f"document {i}")

        rag.ingest_directory(tmp)

    assert True