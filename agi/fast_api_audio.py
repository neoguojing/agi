import threading
import time
from queue import Queue, Empty
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from agi.llms.tts import TextToSpeech
import numpy as np



def generate_pcm(pcm_queue: Queue, wait_timeout=1.0, idle_sleep=0.1):
    while True:
        try:
            pcm_chunk = pcm_queue.get(timeout=wait_timeout)  # 等待数据
            yield pcm_chunk
        except Empty:
            # 没拿到数据，发送空包保持连接
            yield b''
            # 避免忙等
            time.sleep(idle_sleep)

router_audio = APIRouter(prefix="/v1")
@router_audio.get("/audio_stream/{tenant_id}")
def audio_stream(tenant_id: str):
    pcm_queue = TextToSpeech.get_queue(tenant_id)
    headers = {
        "Content-Type": "audio/L16; rate=24000; channels=1"
    }
    return StreamingResponse(generate_pcm(pcm_queue), headers=headers)
