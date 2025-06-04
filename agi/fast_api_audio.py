import threading
import time
from queue import Queue, Empty
from fastapi import APIRouter
from fastapi import WebSocket
from fastapi.responses import StreamingResponse
from agi.llms.tts import TextToSpeech,END_TAG
from agi.config import log,TTS_MODEL_DIR
import numpy as np
from typing import Generator, Optional
import asyncio
import json

def generate_pcm(
    pcm_queue: Queue[bytes],
    sample_rate: int = 24000,
    chunk_size: int = 480,
    wait_timeout: float = 0.1
) -> Generator[bytes, None, None]:
    """生成PCM音频流的生成器函数
    
    参数:
        pcm_queue: 输入PCM数据队列
        sample_rate: 采样率(Hz)
        chunk_size: 每帧采样点数
        wait_timeout: 队列获取超时时间(s)
        
    返回:
        生成器，每次yield一个PCM数据块
        
    注意:
        修改了时间控制逻辑，避免wait_timeout过长导致的节奏问题
    """
    frame_duration = chunk_size / sample_rate  # 每帧持续时间(秒)
    next_frame_time = time.monotonic()  # 使用单调时钟避免时间回退
    
    while True:
        # 计算最大等待时间不超过帧持续时间
        max_wait = min(wait_timeout, frame_duration * 1.5)  # 最多等待1.5帧时间
        
        # 尝试获取数据（限制等待时间）
        try:
            # pcm_chunk = pcm_queue.get(timeout=max_wait)
            pcm_chunk = pcm_queue.get(block=True)
        except Empty:
            # 生成静音帧(16bit PCM)
            pcm_chunk = bytes(chunk_size * 2)
        
        # 计算处理后的时间
        # current_time = time.monotonic()
        # elapsed = current_time - next_frame_time
        
        # # 如果滞后超过3帧，重置时间基准
        # if elapsed > frame_duration * 3:
        #     next_frame_time = current_time + frame_duration
        #     yield pcm_chunk
        #     continue
        
        # 正常节奏控制
        yield pcm_chunk
        
        time.sleep(0.001)
        # 计算下一帧时间并睡眠
        # next_frame_time += frame_duration
        # sleep_time = next_frame_time - time.monotonic()
        
        # # 最小睡眠阈值避免过度CPU占用
        # if sleep_time > 0.001:
        #     time.sleep(sleep_time)


router_audio = APIRouter(prefix="/v1")

@router_audio.get("/audio_stream/{tenant_id}")
def audio_stream(tenant_id: str):
    pcm_queue = TextToSpeech.get_queue(tenant_id)
    
    content_type = f"audio/L16; rate=16000; channels=1"
    if "cosyvoice" in TTS_MODEL_DIR:
        content_type = f"audio/L16; rate=24000; channels=1"
        
    headers = {
        "Content-Type": content_type
    }
    return StreamingResponse(generate_pcm(pcm_queue), headers=headers)

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
                frame = pcm_queue.get(block=False,timeout=0.01)
                if frame:
                    await websocket.send_bytes(frame)
            except asyncio.QueueEmpty:
                # try:
                #     ctrl_msg = await websocket.receive_json()
                #     if ctrl_msg.get("type") == "pause":
                #         # pause logic here
                #         pass
                # except asyncio.TimeoutError:
                pass
            finally:
                await asyncio.sleep(0.01)
                    
    except Exception as e:
        print(f"WebSocket错误: {e}")
    finally:
        await websocket.close()