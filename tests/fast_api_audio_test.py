import pytest
import asyncio
import json
from fastapi.testclient import TestClient
from websockets.connect import connect
from starlette.websockets import WebSocketDisconnect

from agi.fastapi_agi import app  # 替换为你的实际 FastAPI 实例名

client = TestClient(app)

@pytest.mark.asyncio
async def test_audio_stream_ws(monkeypatch):
    tenant_id = "test_tenant"

    # 模拟 TTS 模块的 get_queue 方法，返回一个假的队列
    class FakeQueue:
        def get(self, block=False, timeout=0):
            return b"\x00\x01\x02\x03"  # 假的 PCM 数据

    from agi.llms.tts import TextToSpeech  # 替换为实际模块
    monkeypatch.setattr(TextToSpeech, "get_queue", lambda tid: FakeQueue())

    # 启动 FastAPI 测试服务器
    import websockets

    async with websockets.connect(f"ws://localhost:8000/v1/ws/audio_stream/{tenant_id}") as websocket:
        # 发送初始配置请求
        await websocket.send(json.dumps({"type": "config_request"}))
        config_resp = await websocket.recv()
        config_data = json.loads(config_resp)
        
        assert config_data["type"] == "config"
        assert config_data["rate"] in (16000, 24000)
        assert config_data["channels"] == 1

        # 尝试接收一帧音频数据
        audio_frame = await websocket.recv()
        assert isinstance(audio_frame, bytes)
        assert len(audio_frame) > 0