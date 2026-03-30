import numpy as np
import soundfile as sf
from io import BytesIO
from typing import Union, List

# Triton client
import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient

class TritonASRClient:
    def __init__(
        self,
        model_name: str,
        url: str = "localhost:8001",
        protocol: str = "grpc",  # "grpc" or "http"
        verbose: bool = False,
    ):
        self.model_name = model_name
        self.protocol = protocol.lower()

        if self.protocol == "grpc":
            self.client = grpcclient.InferenceServerClient(
                url=url, verbose=verbose
            )
        else:
            self.client = httpclient.InferenceServerClient(
                url=url, verbose=verbose
            )

    # -----------------------------
    # 音频处理
    # -----------------------------
    def _read_audio(self, audio: Union[str, bytes, BytesIO, np.ndarray]) -> np.ndarray:
        """
        输出 float32 waveform
        """
        if isinstance(audio, str):
            audio, _ = sf.read(audio)
        elif isinstance(audio, (bytes, BytesIO)):
            audio, _ = sf.read(BytesIO(audio))
        elif isinstance(audio, np.ndarray):
            audio = audio.astype(np.float32) if audio.dtype != np.float32 else audio
        else:
            raise ValueError("Unsupported audio input")
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        return audio

    # -----------------------------
    # 推理
    # -----------------------------
    def infer(self, audio_inputs: Union[Union[str, bytes, np.ndarray], List[Union[str, bytes, np.ndarray]]]):
        """
        支持单条或批量
        返回 List[str]
        """
        if not isinstance(audio_inputs, list):
            audio_inputs = [audio_inputs]

        # 转换所有 audio
        batch = [self._read_audio(a) for a in audio_inputs]

        # Triton 输入 tensor
        if self.protocol == "grpc":
            inputs = [
                grpcclient.InferInput("AUDIO_DATA", [len(b)], "UINT8")
                for b in batch
            ]
            for i, b in enumerate(batch):
                inputs[i].set_data_from_numpy(b.tobytes(), binary_data=True)
            outputs = [grpcclient.InferRequestedOutput("TRANSCRIPTION")]
            response = self.client.infer(
                model_name=self.model_name,
                inputs=inputs,
                outputs=outputs,
            )
            # 解析输出
            result = [
                r.decode("utf-8") for r in response.as_numpy("TRANSCRIPTION")
            ]

        else:  # http
            inputs = [
                httpclient.InferInput("AUDIO_DATA", [len(b)], "UINT8")
                for b in batch
            ]
            for i, b in enumerate(batch):
                inputs[i].set_data_from_numpy(b.tobytes(), binary_data=True)
            outputs = [httpclient.InferRequestedOutput("TRANSCRIPTION")]
            response = self.client.infer(
                model_name=self.model_name,
                inputs=inputs,
                outputs=outputs,
            )
            result = [
                r.decode("utf-8") for r in response.as_numpy("TRANSCRIPTION")
            ]

        return result if len(result) > 1 else result[0]