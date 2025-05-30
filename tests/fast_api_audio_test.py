
from fastapi.testclient import TestClient
from agi.fastapi_agi import app
from agi.config import log
from agi.llms.tts import TextToSpeech 
import numpy as np

# 创建测试客户端
client = TestClient(app)

# 测试列出文件 API
def test_audio_stream():
    tenant_id = "test_tenant"

    dummy_pcm = np.ones(1024, dtype=np.int16)
    TextToSpeech.get_queue(tenant_id).put(dummy_pcm.tobytes())


    response = client.get(f"/v1/audio_stream/{tenant_id}")
    assert response.status_code == 200
    assert response.headers["Content-Type"] == "audio/L16; rate=24000; channels=1"

    # 读取几块 PCM 数据
    chunk_count = 0
    for chunk in response.iter_content(chunk_size=2048):  # 1024 samples x 2 bytes
        print(len(chunk))
        print(chunk)

        assert isinstance(chunk, bytes)
        assert len(chunk) == 2048
        chunk_count += 1
        if chunk_count >= 2:
            break

    assert chunk_count >= 1
