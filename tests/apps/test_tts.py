import pytest
import httpx

# 远端服务地址
REMOTE_BASE_URL = "http://localhost:8002"
api_key = "123"

def test_generate_speech_remote():
    response = httpx.post(
        f"{REMOTE_BASE_URL}/v1/audio/speech",
        headers={
            "Authorization": f"Bearer {api_key}"
        },
        json={
            "input": "你好，这是一个测试。",
            "voice": "test_voice",
            "response_format": "wav",
            "speed": 1.0,
            "user": "test"
        },
        timeout=60.0
    )
    assert response.status_code == 200, f"错误返回: {response.text}"
    assert response.headers["content-type"].startswith("audio/wav")
    assert len(response.content) > 100  # 基础校验数据大小


@pytest.mark.asyncio
async def test_generate_speech_streaming_remote():
    async with httpx.AsyncClient(base_url=REMOTE_BASE_URL, timeout=60.0) as client:
        response = await client.post(
            "/v1/audio/speech/streaming",
            headers={
                "Authorization": f"Bearer {api_key}"
            },
            json={
                "input": "测试流式语音",
                "response_format": "pcm",
                "user": "test_stream"
            }
        )

        assert response.status_code == 200, f"错误返回: {response.text}"
        assert response.headers["content-type"] == "audio/pcm"

        # 读取流
        data = b""
        async for chunk in response.aiter_bytes():
            print(f"收到 chunk 大小: {len(chunk)}")
            data += chunk
            if len(data) > 1024:
                break  # 收到足够数据就停止

        assert len(data) > 0
