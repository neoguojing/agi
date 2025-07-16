from queue import Empty
import time
from fastapi import WebSocket
from agi.config import log,TTS_MODEL_DIR
from agi.apps.tts.tts import TTS,SENTINEL
from pydantic import BaseModel, Field
from typing import Literal, Optional, List, Union
from agi.apps.common import verify_api_key,SpeechRequest
from fastapi import Depends, HTTPException,FastAPI
from fastapi.responses import StreamingResponse,FileResponse
from typing import AsyncGenerator,Generator
from fastapi.concurrency import run_in_threadpool
import asyncio
import traceback

tts = TTS()
# 初始化 FastAPI 应用
app = FastAPI(
    title="AGI TTS API",
    description="兼容 OpenAI API 的 AGI 接口",
    version="1.0.0",
    # lifespan=lifespan
)

@app.websocket("/v1/ws/audio_stream/{tenant_id}")
async def audio_stream_ws(websocket: WebSocket, tenant_id: str):
    await websocket.accept()
    pcm_queue = TTS.get_queue(tenant_id)
    
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
                    if frame is SENTINEL:
                        # 接到结束信号，退出 generator
                        continue
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


@app.post("/v1/audio/speech",summary="文本转语音")
async def generate_speech(request: SpeechRequest, api_key: str = Depends(verify_api_key)):
    """
    接收文本并生成语音文件。
    """
    log.info(request)
    try:
        if request.stream:
            return await generate_speech_streaming(request,api_key=api_key)
        else:
            _ ,file_path = tts.invoke(request.input,user_id=request.user,save_file=True,model_name=request.model)
            return FileResponse(file_path, media_type=f"audio/{request.response_format}", filename=file_path)
    
    except Exception as e:
        log.error(e)
        print(traceback.format_exc())

        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/audio/speech/streaming",summary="文本转语音")
async def generate_speech_streaming(request: SpeechRequest, api_key: str = Depends(verify_api_key)):
    """
    接收文本并生成语音文件。
    """
    log.info(request)
    try:
        # 非阻塞线程池执行
        asyncio.create_task(run_in_threadpool(tts.invoke, input_str=request.input,user_id=request.user,model_name=request.model))
        return StreamingResponse(
            audio_generator(request.user),
            media_type="audio/pcm"
        )
    
    except Exception as e:
        log.error(e)
        print(traceback.format_exc())

        raise HTTPException(status_code=500, detail=str(e))

def audio_generator(tenant_id: str = "default") -> Generator[bytes, None, None]:
    """
    同步生成器：从PCM队列取chunk并立即yield，
    最终触发 StreamingResponse 的 chunked 传输。
    """
    pcm_queue = TTS.get_queue(tenant_id)
    while True:
        frame = pcm_queue.get(block=True)
        if frame is SENTINEL:
            # 接到结束信号，退出 generator
            break
        yield frame

        time.sleep(0.01)

