import pytest
import os
from httpx import AsyncClient
from agi.apps.whisper.fast_api_whisper import app
api_key = "123"
headers={
    "Authorization": f"Bearer {api_key}"
}
@pytest.mark.asyncio
async def test_transcribe_audio():
    # 构造上传文件
    audio_path = "tests/zh-cn-sample.wav"  # 准备一段小音频
    assert os.path.exists(audio_path), "测试音频不存在"

    with open(audio_path, "rb") as f:
        files = {"file": ("zh-cn-sample.wav", f, "audio/wav")}
        data = {
            "model_name": "whisper-1",
            "response_format": "json"
        }


        async with AsyncClient(app=app, base_url="http://test") as ac:
            response = await ac.post("/v1/audio/transcriptions", data=data, files=files,headers=headers)
            assert response.status_code == 200
            res = response.json()
            assert "text" in res
            assert "测试" in res["text"]
