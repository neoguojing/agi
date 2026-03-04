import pytest
from unittest.mock import MagicMock
from agi.tasks.vectore_store import CollectionManager
from langchain_core.documents import Document
from agi.tasks.task_factory import TaskFactory
class DummyEmbedding:
    def embed_query(self, text):
        return [0.1] * 768  # mock embedding

@pytest.fixture
def collection_manager(tmp_path):
    # 创建临时目录作为数据库路径
    return CollectionManager(
        data_path=str(tmp_path),
        embedding=TaskFactory.get_embedding()
    )

@pytest.mark.asyncio
async def test_embedding_search(collection_manager):
    texts = ["北京是中国的首都", "苹果公司是一家科技公司"]
    results = await collection_manager.embedding_search(
        texts,
        collection_name="test_collection"
    )

    assert isinstance(results, list)
    for doc in results:
        assert isinstance(doc, Document)
        assert "page_content" in doc.__dict__
        assert "metadata" in doc.__dict__

@pytest.mark.asyncio
async def test_full_search(collection_manager):
    texts = ["人工智能正在改变世界", "大数据和云计算是热门技术"]
    results = await collection_manager.full_search(
        texts,
        collection_name="test_collection"
    )

    assert isinstance(results, list)
    for doc in results:
        assert isinstance(doc, Document)
        assert "page_content" in doc.__dict__
        assert "metadata" in doc.__dict__

def test_build_query():
    manager = CollectionManager("/tmp", DummyEmbedding())
    query = manager.build_query(["北京", "科技"], ["垃圾"])
    assert "$or" in query
    assert {"$contains": "北京"} in query["$or"]
    assert {"$not_contains": "垃圾"} in query["$or"]
