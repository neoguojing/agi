import pytest
from httpx import AsyncClient
from agi.apps.image.fast_api_image import app  # 替换为你的实际模块路径
import base64

api_key = "123"
headers={
    "Authorization": f"Bearer {api_key}"
}

@pytest.mark.asyncio
async def test_generate_image():
    payload = {
        "model": "your-model-name",
        "prompt": "a cat riding a bicycle",
        "n": 1,
        "size": "256x256",
        "response_format": "url",
        "user": "test",
        "moderation": "auto"
    }

    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.post("/v1/images/generations", json=payload,headers=headers)

    assert response.status_code == 200
    json_data = response.json()
    print(json_data)
    assert "created" in json_data
    assert isinstance(json_data["data"], list)
    assert "url" in json_data["data"][0]


# 工具函数：读取本地图片并编码为 base64
def encode_image_base64(path):
    with open(path, "rb") as f:
        return "data:image/jpg;base64," + base64.b64encode(f.read()).decode("utf-8")

@pytest.mark.asyncio
async def test_image_edit_with_base64_input():
    # 模拟请求体（image_url 是 base64 编码）
    base64_image = encode_image_base64("tests/cat.jpg")

    request_body = {
        "model": "test-model",
        "messages": [
            {
                "role": "user",
                "content": [
                    { "type": "text", "text": "把这张图片变成黑白的" },
                    { "type": "image_url", "image_url": { "url": base64_image } }
                ]
            }
        ]
    }
    async with AsyncClient(app=app, base_url="http://test") as ac:
        resp = await ac.post("/v1/chat/completions", json=request_body,headers=headers)

        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "chat.completion"
        assert data["choices"][0]["message"]["content"] == "http://fake.com/edited.png"

