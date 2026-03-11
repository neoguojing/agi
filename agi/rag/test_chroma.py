import pytest
import asyncio
from langchain_core.documents import Document

# 假设上述代码保存在 chroma_framework.py 中
from agi.rag.chroma import ChromaClientFactory, CollectionManager

@pytest.fixture
def manager():
    """初始化管理器单例"""
    # 使用临时目录进行测试，防止污染生产数据
    factory = ChromaClientFactory(data_path="./test_chroma_db", allow_reset=True)
    # 模拟一个简单的 Embedding Model
    from langchain_community.embeddings import FakeEmbeddings
    fake_ef = FakeEmbeddings(size=384) 
    return CollectionManager(factory, fake_ef)

@pytest.mark.asyncio
async def test_tenant_isolation(manager):
    """测试多租户物理隔离"""
    # 1. 租户 A 写入数据
    handle_a = manager.get_handle("shared_col", tenant="tenant_A")
    doc_a = [Document(page_content="Apple is a fruit.", metadata={"doc_id": "a1"})]
    await handle_a.upsert(doc_a)

    # 2. 租户 B 写入同名 Collection 但内容不同
    handle_b = manager.get_handle("shared_col", tenant="tenant_B")
    doc_b = [Document(page_content="Boeing is a plane.", metadata={"doc_id": "b1"})]
    await handle_b.upsert(doc_b)

    # 3. 验证隔离性：从 A 的 Handle 搜不到 B 的内容
    results_a = await handle_a.hybrid_search(texts=["plane"], k=5)
    # 即使搜索关键词 "plane"，由于 Tenant A 只有 "Apple"，向量距离会非常远或者不匹配关键词
    assert all("Boeing" not in d.page_content for d in results_a["plane"])
    
    # 4. 验证数量
    docs_a = handle_a.get_documents(limit=10)
    docs_b = handle_b.get_documents(limit=10)
    assert len(docs_a) == 1
    assert len(docs_b) == 1
    assert docs_a[0].page_content != docs_b[0].page_content

@pytest.mark.asyncio
async def test_crud_operations(manager):
    """测试标准的增删改查"""
    handle = manager.get_handle("crud_test", tenant="tester")
    
    # Create / Upsert
    doc_id = "test_doc_001"
    test_docs = [
        Document(page_content="Deep learning is part of machine learning.", 
                 metadata={"doc_id": doc_id, "category": "AI"})
    ]
    await handle.upsert(test_docs)
    
    # Read
    read_docs = handle.get_documents(limit=1)
    assert read_docs[0].metadata["doc_id"] == doc_id
    
    # Update (通过 upsert 相同 ID)
    updated_docs = [
        Document(page_content="Updated content about AI.", 
                 metadata={"doc_id": doc_id, "category": "AI"})
    ]
    await handle.upsert(updated_docs)
    final_docs = handle.get_documents(limit=1)
    assert "Updated" in final_docs[0].page_content
    
    # Delete
    handle.delete(ids=[doc_id])
    after_delete = handle.get_documents(limit=1)
    assert len(after_delete) == 0

@pytest.mark.asyncio
async def test_hybrid_search_with_textprocessor(manager):
    """测试 TextProcessor 提取关键词并进行混合搜索"""
    handle = manager.get_handle("search_test", tenant="tester")
    
    # 写入带有显著关键词的文档
    docs = [
        Document(page_content="The Great Wall is a famous landmark in China.", metadata={"doc_id": "wall"}),
        Document(page_content="Pyramids are historical structures in Egypt.", metadata={"doc_id": "pyramid"})
    ]
    await handle.upsert(docs)
    
    # 执行搜索
    # TextProcessor 应该能提取出 "China" 或 "Great Wall"
    search_results = await handle.hybrid_search(texts=["Tell me about landmarks in China"], k=1)
    
    found_docs = search_results["Tell me about landmarks in China"]
    assert len(found_docs) > 0
    assert "Great Wall" in found_docs[0].page_content
    # 检查 score 是否被正确注入
    assert "score" in found_docs[0].metadata

def test_collection_management(manager):
    """测试库管理逻辑"""
    t, db = "mgt_tenant", "mgt_db"
    col_name = "to_be_deleted"
    
    # 创建
    manager.create_collection(col_name, tenant=t, database=db)
    assert col_name in manager.list_collections(tenant=t, database=db)
    
    # 删除
    manager.delete_collection(col_name, tenant=t, database=db)
    assert col_name not in manager.list_collections(tenant=t, database=db)