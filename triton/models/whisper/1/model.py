import triton_python_backend_utils as pb_utils
import numpy as np
import time
import threading
import os
from io import BytesIO

from faster_whisper import WhisperModel

try:
    import torch
except:
    torch = None


class TritonPythonModel:
    def initialize(self, args):
        """
        Triton 启动时调用（只执行一次）
        """
        self.model_repo = args["model_repository"]
        self.model_name = args["model_name"]
        self.model_version = args["model_version"]

        # ===== 配置 =====
        self.device = "cuda"
        self.compute_type = "float16"
        self.timeout = 300
        self.beam_size = 5

        # 模型路径（推荐放 weights）
        self.model_path = os.path.join(
            self.model_repo,
            self.model_name,
            self.model_version,
            "weights"
        )

        if not os.path.exists(self.model_path):
            # fallback
            self.model_path = "large-v3"

        self.whisper = None
        self.last_used = 0

        self.lock = threading.RLock()

        # 启动回收线程
        self._stop_event = threading.Event()
        self.monitor_thread = threading.Thread(
            target=self._monitor, daemon=True
        )
        self.monitor_thread.start()

    # =========================
    # 模型加载 / 卸载
    # =========================
    def _load(self):
        if self.whisper is not None:
            return

        print(f"[Triton] Loading Whisper: {self.model_path}")

        self.whisper = WhisperModel(
            self.model_path,
            device=self.device,
            compute_type=self.compute_type,
        )

    def _unload(self):
        if self.whisper is None:
            return

        print("[Triton] Unloading Whisper")

        try:
            del self.whisper
            self.whisper = None

            if torch and self.device.startswith("cuda"):
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"Unload error: {e}")

    def _get_model(self):
        with self.lock:
            self.last_used = time.time()

            if self.whisper is None:
                self._load()

            return self.whisper

    # =========================
    # 音频处理
    # =========================
    def _decode_audio(self, raw):
        """
        输入支持：
        - bytes (wav)
        - float32 ndarray
        - int16 ndarray
        """
        import soundfile as sf

        if isinstance(raw, bytes):
            audio, _ = sf.read(BytesIO(raw))
            return audio.astype(np.float32)

        if isinstance(raw, np.ndarray):
            if raw.dtype == np.int16:
                raw = raw.astype(np.float32) / 32768.0
            return raw

        raise ValueError("Unsupported audio format")

    # =========================
    # 推理入口
    # =========================
    def execute(self, requests):
        responses = []

        model = self._get_model()

        for request in requests:
            in_tensor = pb_utils.get_input_tensor_by_name(
                request, "AUDIO_DATA"
            )
            batch = in_tensor.as_numpy()

            results = []

            for item in batch:
                try:
                    audio = self._decode_audio(item)

                    segments, info = model.transcribe(
                        audio,
                        beam_size=self.beam_size
                    )

                    text = "".join(s.text for s in segments)

                except Exception as e:
                    text = f"ERROR: {str(e)}"

                results.append(text)

            out_tensor = pb_utils.Tensor(
                "TRANSCRIPTION",
                np.array(results, dtype=np.object_)
            )

            responses.append(
                pb_utils.InferenceResponse(
                    output_tensors=[out_tensor]
                )
            )

        return responses

    # =========================
    # 自动卸载
    # =========================
    def _monitor(self):
        while not self._stop_event.is_set():
            time.sleep(30)

            with self.lock:
                if (
                    self.whisper is not None
                    and time.time() - self.last_used > self.timeout
                ):
                    self._unload()

    def finalize(self):
        self._stop_event.set()
        self.monitor_thread.join(timeout=1)

        with self.lock:
            self._unload()