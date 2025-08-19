import pytest
import os
from httpx import AsyncClient
from agi.apps.whisper.fast_api_whisper import app
api_key = "123"
headers={
    "Authorization": f"Bearer {api_key}"
}
REMOTE_BASE_URL = "http://localhost:8003"
@pytest.mark.asyncio
async def test_transcribe_audio_remote():
    audio_path = "tests/zh-cn-sample.wav"  # 本地准备好的测试音频
    assert os.path.exists(audio_path), "测试音频不存在"

    with open(audio_path, "rb") as f:
        files = {"file": ("zh-cn-sample.wav", f, "audio/wav")}
        data = {
            "model_name": "whisper-1",
            "response_format": "json"
        }

        async with AsyncClient(base_url=REMOTE_BASE_URL, timeout=60.0) as ac:
            response = await ac.post(
                "/v1/audio/transcriptions",
                data=data,
                files=files,
                headers=headers
            )
            assert response.status_code == 200, f"服务返回错误: {response.text}"
            res = response.json()
            print(res)
            assert "text" in res
            assert "测试" in res["text"]
