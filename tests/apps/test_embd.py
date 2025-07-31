import pytest
from httpx import AsyncClient
from agi.apps.embding.fast_api_embding import app  # 替换为你的实际模块路径
from agi.tasks.task_factory import TaskFactory
from langchain_openai import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings
from agi.llms.rerank import rerank_with_batching

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
    ret1 = client.embed_query("good morning")
    assert isinstance(ret1,list)
    assert ret != ret1




@pytest.mark.asyncio
async def test_rerank():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        payload = {
            "query": "What are the effects of global warming?",
            "documents": [
                "Global warming has led to rising sea levels due to melting glaciers and polar ice caps. Coastal cities around the world are at risk of flooding, and small island nations face existential threats. Additionally, increased ocean temperatures contribute to more frequent and severe hurricanes.",
                
                "The Mona Lisa is a famous portrait painted by Leonardo da Vinci during the Italian Renaissance. It is housed in the Louvre Museum in Paris and is considered one of the most iconic pieces of art in history.",
                
                "As global temperatures rise, ecosystems are disrupted, leading to habitat loss and species extinction. For example, coral reefs suffer from bleaching events caused by warmer waters. Changes in temperature and rainfall patterns also affect agriculture, threatening food security.",
                
                "The process of photosynthesis allows plants to convert sunlight into energy. This process is essential for producing the oxygen we breathe and forms the basis of the food chain. It occurs primarily in the chloroplasts of plant cells.",
                
                "Global warming contributes to more frequent and intense heatwaves, droughts, and wildfires. These events put pressure on human health systems, increase energy demand for cooling, and exacerbate existing inequalities by disproportionately affecting vulnerable populations."
            ],
            "model": "qwen",
            "top_k": 3
        }
        response = await ac.post("/v1/rerank", json=payload)

    assert response.status_code == 200, response.text
    data = response.json()
    print(data)
    # 验证 response 格式
    assert data["object"] == "list"
    assert isinstance(data["data"], list)
    assert len(data["data"]) == 3  # 因为 top_k=2
    assert data["model"] == "qwen"

    # 验证每一项结构
    for item in data["data"]:
        assert item["object"] == "rerank"
        assert isinstance(item["index"], int)
        assert isinstance(item["document"], str)
        assert isinstance(item["score"], float)
