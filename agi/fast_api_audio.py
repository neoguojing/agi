from queue import Empty
from fastapi import APIRouter
from fastapi import WebSocket
from agi.llms.tts import TextToSpeech
from agi.config import log,TTS_MODEL_DIR
import asyncio
router_audio = APIRouter(prefix="/v1")

@router_audio.websocket("/ws/audio_stream/{tenant_id}")
async def audio_stream_ws(websocket: WebSocket, tenant_id: str):
    await websocket.accept()
    pcm_queue = TextToSpeech.get_queue(tenant_id)
    
    try:
        # 等待客户端初始配置请求
        init_msg = await websocket.receive_json()
        if init_msg.get("type") == "config_request":
            await websocket.send_json({
                "type": "config",
                "rate": 24000 if "cosyvoice" in TTS_MODEL_DIR else 16000,
                "channels": 1
            })

        # 主音频流循环
        while True:
            try:
                frame = pcm_queue.get(block=False, timeout=0.01)
                if frame:
                    await websocket.send_bytes(frame)
            except Empty:  # 显式捕获空队列异常
                # await websocket.send_json({
                #     "type": "event",
                #     "data": "empty"
                # })
                pass
            except Exception as e:
                log.error(f"帧处理异常: {e}")
                break

            await asyncio.sleep(0.01)
                    
    except Exception as e:
        print(f"WebSocket错误: {e}")
    finally:
        await websocket.close()