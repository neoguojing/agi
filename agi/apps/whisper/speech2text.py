import threading
import time
import os
from typing import Union
import numpy as np
from agi.config import WHISPER_GPU_ENABLE,log,COMPUTE_TYPE,MODEL_PATH
from agi.config import WHISPER_MODEL_DIR as model_root
from agi.utils.common import Media,Timer
from dataclasses import asdict
from io import BytesIO


style = 'style="width: 100%; max-height: 100vh;"'

class Speech2Text:
    def __init__(self, model_path: str=model_root, timeout: int = 300,compute_type: str = COMPUTE_TYPE):
        """
        model_path: 模型本地路径或 huggingface 名称
        timeout: 超过多少秒未使用就自动卸载（默认10分钟）
        """
        self.model_size = model_path
        self.timeout = timeout
        self.model = None
        self.whisper = None
        self.last_used = 0
        self.lock = threading.Lock()
        self.monitor_thread = threading.Thread(target=self._monitor, daemon=True)
        self.monitor_thread.start()
        self.local_files_only = True

        self.beam_size = 5
        self.compute_type = compute_type
        if WHISPER_GPU_ENABLE:
            self.device = "cuda"
            if not os.path.exists(self.model_size):
                self.model_size = os.path.join(MODEL_PATH,"models--Systran--faster-whisper-large-v3")
                # self.local_files_only=False
            log.info(self.model_size)
        else:
            self.device = "cpu"
            log.info(self.model_size)

    def get_model(self,device:str):
        """访问模型，如果未加载则自动加载"""
        with self.lock:
            self.last_used = time.time()
            if self.model is None:
                self.device = device
                self._load()
            else:
                if self.device != device:
                    self._unload()
                    self.device = device
                    self._load()

            return self.model

    def _load(self):
        from faster_whisper import WhisperModel
        whisper = None
        if self.device == "cpu":
            self.model_size = os.path.join(MODEL_PATH,"models--Systran--faster-whisper-base")
            if not os.path.exists(self.model_size):
                self.model_size = "base"
            print(f"[Model] Loading model from {self.model_size}")
            whisper = WhisperModel(self.model_size, device=self.device, compute_type="int8",local_files_only=self.local_files_only)
        else:
            if not os.path.exists(self.model_size):
                self.model_size = os.path.join(MODEL_PATH,"models--Systran--faster-whisper-large-v3")
                # self.local_files_only=False
            print(f"[Model] Loading model from {self.model_size}")
            whisper = WhisperModel(self.model_size, device=self.device, compute_type=self.compute_type,local_files_only=self.local_files_only)

        self.whisper = whisper
        self.model = whisper.model

    def invoke(self, audio_input: Union[str, np.ndarray,BytesIO],device="cuda") -> str:
        """Generate an image from the input text."""
        self.get_model(device)

        audio_input= Media.from_data(audio_input,media_type="audio")
        
        if audio_input is None:
            return "No valid audio input found."
        with Timer():
            # Transcribe the audio input
            segments, info = self.whisper.transcribe(audio_input.data, beam_size=self.beam_size)
            content = "".join(segment.text for segment in segments)

            log.info(f"speech to text:{content}")
            return content, asdict(info)

    def _unload(self):
        print(f"[Model] Unloading model from {self.model_size}")
        del self.model
        self.model = None
        self.whisper = None
        # 非torch模型，不依赖pytorch，使用的是onnx
        # torch.cuda.empty_cache()

    def _monitor(self):
        """后台线程定期检查是否应卸载模型"""
        while True:
            time.sleep(30)
            with self.lock:
                if self.model and (time.time() - self.last_used > self.timeout):
                    self._unload()
