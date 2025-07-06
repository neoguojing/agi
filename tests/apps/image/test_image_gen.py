import pytest
from httpx import AsyncClient
from agi.apps.image.fast_api_image import app  # 替换为你的实际模块路径
from fastapi.testclient import TestClient
from PIL import Image
import io

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

@pytest.mark.asyncio
async def test_edit_image():
    # 构造一张临时图片（RGB 白底）
    img = Image.open("tests/cat.jpg")
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="JPEG")
    img_bytes.seek(0)

    files = {
        "image": ("cat.jpg", img_bytes, "image/jpeg")
    }
    data = {
        "prompt": "Add a hat on the top of cat"
    }

    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.post("/v1/images/edits", data=data, files=files,headers=headers)

    assert response.status_code == 200
    json_data = response.json()
    assert "created" in json_data
    assert isinstance(json_data["data"], list)
    assert "url" in json_data["data"][0]
