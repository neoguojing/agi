import threading
import time
from queue import Queue, Empty
from fastapi import APIRouter
from fastapi.responses import StreamingResponse

import numpy as np

router_audio = APIRouter(prefix="/v1")

pcm_queue = Queue(maxsize=100)

def tts_pcm_producer():
    sample_rate = 16000
    freq = 440
    chunk_size = 1024
    t = 0
    while True:
        t_vals = np.arange(t, t + chunk_size)
        wave = 0.1 * np.sin(2 * np.pi * freq * t_vals / sample_rate)
        pcm_chunk = (wave * 32767).astype('int16').tobytes()
        try:
            pcm_queue.put(pcm_chunk, timeout=1)
        except:
            pass
        t += chunk_size
        time.sleep(chunk_size / sample_rate)

def generate_pcm():
    while True:
        try:
            pcm_chunk = pcm_queue.get(block=True)
        except Empty:
            yield b''  # 队列空，发空包防断流
            continue
        yield pcm_chunk

@router_audio.get("/audio_stream")
def audio_stream():
    headers = {
        "Content-Type": "audio/L16; rate=16000; channels=1"
    }
    return StreamingResponse(generate_pcm(), headers=headers)
