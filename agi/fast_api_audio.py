import threading
import time
from queue import Queue, Empty
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from agi.llms.tts import TextToSpeech,END_TAG
from agi.config import log
import numpy as np

def generate_pcm(pcm_queue: Queue, wait_timeout=0.2, sample_rate=24000, chunk_size=1024):
    # frame_duration = chunk_size / sample_rate  # 每帧的播放时长（秒）
    wait_empty_times = 0
    while True:
        # start_time = time.time()
        try:
            # 空5次之后，发送END_TAG，让客户端主动结束当前
            if wait_empty_times >= 5:
                wait_empty_times = 0
                log.info(f"generate_pcm: end tag send")
                yield END_TAG
                
            pcm_chunk = pcm_queue.get(timeout=wait_timeout)
            # pcm_chunk = pcm_queue.get(block=True)
            log.debug(f"generate_pcm: {len(pcm_chunk)} bytes")
        except Empty:
            pcm_chunk = b''  # 发空包保持连接
            wait_empty_times += 1
            # pcm_chunk = b'\x00' * chunk_size * 2

        yield pcm_chunk

        # 控制频率：保证下一帧在正确时间发送
        # elapsed = time.time() - start_time
        # sleep_time = frame_duration - elapsed
        # if sleep_time > 0:
        #     time.sleep(sleep_time)
        # else:
        #     log.debug(f"Processing lagging behind by {-sleep_time:.4f}s")


router_audio = APIRouter(prefix="/v1")
@router_audio.get("/audio_stream/{tenant_id}")
def audio_stream(tenant_id: str):
    pcm_queue = TextToSpeech.get_queue(tenant_id)
    headers = {
        "Content-Type": "audio/L16; rate=24000; channels=1"
    }
    return StreamingResponse(generate_pcm(pcm_queue), headers=headers)
