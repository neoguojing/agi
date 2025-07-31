import pytest
from httpx import AsyncClient
from agi.apps.embding.fast_api_embding import app  # 替换为你的实际模块路径
from agi.tasks.task_factory import TaskFactory
from langchain_openai import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings


api_key = "123"
headers={
    "Authorization": f"Bearer {api_key}"
}

@pytest.mark.asyncio
async def test_embding():
    client = TaskFactory.get_embedding()
    assert isinstance(client,OllamaEmbeddings)
    ret = client.embed_query("我爱北京天安门")
    assert isinstance(ret,list)
    print(ret)
    ret1 = client.embed_query("good morning")
    assert isinstance(ret1,list)
    print(ret)
    assert ret != ret1




@pytest.mark.asyncio
async def test_rerank():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        payload = {
            "query": "What is deep learning?",
            "documents": [
                "Deep learning is a subset of machine learning.",
                "The capital of France is Paris.",
                "Neural networks are used in deep learning models."
            ],
            "model": "bge-reranker-base",
            "top_k": 2
        }
        response = await ac.post("/v1/rerank", json=payload)

    assert response.status_code == 200, response.text
    data = response.json()

    # 验证 response 格式
    assert data["object"] == "list"
    assert isinstance(data["data"], list)
    assert len(data["data"]) == 2  # 因为 top_k=2
    assert data["model"] == "qwen"

    # 验证每一项结构
    for item in data["data"]:
        assert item["object"] == "rerank"
        assert isinstance(item["index"], int)
        assert isinstance(item["document"], str)
        assert isinstance(item["score"], float)