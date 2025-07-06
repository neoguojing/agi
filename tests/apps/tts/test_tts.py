from agi.apps.tts.fast_api_audio import app
from agi.apps.tts.tts import SENTINEL
from fastapi.testclient import TestClient
import pytest
import httpx
import asyncio

client = TestClient(app)
api_key = "123"
def test_generate_speech():
    response = client.post(
        "/v1/audio/speech",
        headers={
            "Authorization": f"Bearer {api_key}"
        },
        json={
            "input": "你好，这是一个测试。",
            "voice": "test_voice",
            "response_format": "wav",
            "speed": 1.0
        }
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("audio/wav")
    assert len(response.content) > 100  # 基础校验数据大小

@pytest.mark.asyncio
async def test_generate_speech_streaming():
    async with httpx.AsyncClient(base_url="http://test", app=app) as client:
        response = await client.post(
            "/v1/audio/speech/streaming",
            headers={
                "Authorization": f"Bearer {api_key}"
            },
            json={"input": "测试流式语音", "response_format": "pcm"}
        )

        assert response.status_code == 200
        assert response.headers["content-type"] == "audio/pcm"

        # 读取流
        data = b""
        async for chunk in response.aiter_bytes():
            data += chunk
            if len(data) > 1024:
                break  # 足够判断是否有效流式数据
            print(len(data))
            if data == SENTINEL:
                print(str(data))

        assert len(data) > 0