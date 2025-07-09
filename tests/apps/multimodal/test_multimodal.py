import pytest
from httpx import AsyncClient
from unittest.mock import AsyncMock, patch
from agi.apps.multimodal.fast_api_multimodal import app  # 导入你的 FastAPI 应用
import base64

api_key = "123"
headers={
    "Authorization": f"Bearer {api_key}"
}

def load_base64_image(path: str) -> str:
    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    ext = path.split('.')[-1]
    return f"data:image/{ext};base64,{encoded}"

@pytest.mark.asyncio
async def test_chat_completion_with_image():
    # 构造测试请求体
    request_body = {
        "model": "test-model",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "请描述这张图片"},
                    {"type": "image", "image": load_base64_image("tests/cat.jpg")}
                ]
            }
        ]
    }

    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.post(
            "/v1/chat/completions",
            json=request_body,
            headers=headers
        )
        assert response.status_code == 200
        data = response.json()
        print(data)
        assert data["object"] == "chat.completion"
        assert data["choices"][0]["message"]["content"] == "图中是一只猫"
