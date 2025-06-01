import threading
import time
from queue import Queue, Empty
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from agi.llms.tts import TextToSpeech,END_TAG
from agi.config import log,TTS_MODEL_DIR
import numpy as np

def generate_pcm(pcm_queue: Queue, wait_timeout=0.5, sample_rate=24000, chunk_size=480):
    frame_duration = chunk_size / sample_rate
    expected_next_time = time.time()

    while True:
        try:
            pcm_chunk = pcm_queue.get(block=True)
        except Empty:
            pcm_chunk = b''
            log.warning("generate_pcm: TTS 未及时产出帧，发送空帧防断")

        now = time.time()
        
        # 如果和节奏时间相差太久，说明被阻塞了，直接重置节奏轨
        if abs(now - expected_next_time) > frame_duration * 2:
            expected_next_time = now
            log.warning(f"[generate_pcm] 长时间阻塞，重置节奏时间轨道")
            
        drift = now - expected_next_time
        if abs(drift) > 0.005:
            log.warning(f"[generate_pcm] 节奏抖动 {drift:+.4f}s")

        # 控制节奏
        sleep_time = expected_next_time + frame_duration - time.time()
        if sleep_time > 0:
            time.sleep(sleep_time)
        else:
            log.warning(f"[generate_pcm] 滞后 {-sleep_time:.4f}s")

        expected_next_time += frame_duration
        yield pcm_chunk


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
