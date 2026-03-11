import asyncio
import os
import shutil
from langchain_community.embeddings import FakeEmbeddings # 用于测试，节省算力

# 假设你的代码保存在 knowledge_manager.py 中
from agi.rag.retriever import KnowledgeManager

async def test_knowledge_base_flow():
    # --- 1. 初始化 ---
    # 使用一个临时目录作为测试数据库
    test_db_path = "./test_chroma_db"
    if os.path.exists(test_db_path):
        shutil.rmtree(test_db_path)
    
    # 使用 768 维的模拟向量
    fake_embedding = FakeEmbeddings(size=768)
    km = KnowledgeManager(data_path=test_db_path, embedding=fake_embedding)
    
    collection_name = "test_collection"
    tenant_id = "user_001"
    
    # 模拟一个测试文件（实际测试时请确保该路径存在或 mock loader）
    # 这里我们假设 LoaderFactory 能够处理 .txt 文件
    test_file = "test_document.txt"
    with open(test_file, "w", encoding="utf-8") as f:
        f.write("人工智能是计算机科学的一个分支。它企图了解本质，并生产出一种新的能以相似的方式做出反应的智能机器。")

    print(f"--- 步骤 1: 首次入库 ---")
    await km.store(collection_name, test_file, tenant=tenant_id)
    
    # 验证是否成功入库
    handle = km.collection_manager.get_handle(collection_name, tenant=tenant_id)
    count = handle.collection.count()
    print(f"库中分块数量: {count}")
    assert count > 0, "入库失败，数量不应为0"

    # --- 2. 测试幂等性 (去重) ---
    print(f"\n--- 步骤 2: 重复入库校验 ---")
    # 再次调用 store，由于 _is_file_indexed 的存在，应该跳过处理
    await km.store(collection_name, test_file, tenant=tenant_id)
    
    new_count = handle.collection.count()
    print(f"重复入库后数量: {new_count}")
    assert new_count == count, "去重逻辑失效，检测到重复入库"

    # --- 3. 测试检索 ---
    print(f"\n--- 步骤 3: 检索功能校验 ---")
    query = "什么是人工智能？"
    results = await km.query_doc(collection_name, query, tenant=tenant_id, k=2)
    
    print(f"检索到 {len(results)} 个片段")
    for i, doc in enumerate(results):
        print(f"片段 {i+1} 来源: {doc.metadata.get('source')}")
        print(f"片段 {i+1} 面包屑: {doc.metadata.get('breadcrumb')}")
        print(f"内容摘要: {doc.page_content[:50]}...")
    
    assert len(results) > 0, "检索未返回结果"
    assert "source" in results[0].metadata, "元数据缺失 source 字段"

    # --- 4. 清理 ---
    if os.path.exists(test_file):
        os.remove(test_file)
    # shutil.rmtree(test_db_path) # 正式环境建议清理
    print("\n✅ 所有测试项通过!")

if __name__ == "__main__":
    asyncio.run(test_knowledge_base_flow())