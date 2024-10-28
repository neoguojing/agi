import os
import time
from datetime import date
from pathlib import Path
import torch
from TTS.api import TTS
from agi.config import MODEL_PATH as model_root, CACHE_DIR, TTS_SPEAKER_WAV
from agi.llms.base import CustomerLLM, MultiModalMessage, Audio
from langchain_core.runnables import RunnableConfig
from typing import Any, Optional
from pydantic import BaseModel, Field


class TextToSpeech(CustomerLLM):
    tts: Optional[Any] = Field(default=None)
    speaker_wav: str = Field(default=TTS_SPEAKER_WAV)
    is_gpu: bool = Field(default=False)
    language: str = Field(default="zh-cn")
    save_file: bool = Field(default=True)
    
    def __init__(self,is_gpu = False, save_file: bool = True):
        
        tts = self.initialize_tts(is_gpu)
        super().__init__(llm=tts.synthesizer)
        self.tts = tts
        self.save_file = save_file
        self.is_gpu = is_gpu
       

        
    def initialize_tts(self,is_gpu) -> TTS:
        """Initialize the TTS model based on the available hardware."""
        if is_gpu:
            model_path = os.path.join(model_root, "tts_models--multilingual--multi-dataset--xtts_v2")
            config_path = os.path.join(model_path, "config.json")
            return TTS(model_path=model_path, config_path=config_path).to(torch.device("cuda"))
        else:
            return TTS(model_name="tts_models/zh-CN/baker/tacotron2-DDC-GST")

    def list_available_models(self):
        """Return a list of available TTS models."""
        return self.tts.list_models()

    def invoke(
        self, input: MultiModalMessage, config: Optional[RunnableConfig] = None, **kwargs: Any
    ) -> MultiModalMessage:
        """Invoke the TTS to generate audio from text."""
        if self.save_file:
            file_path = self.save_audio_to_file(text=input.content)
            return MultiModalMessage(content=input.content, audio=Audio(file_path=file_path))
        
        samples = self.generate_audio_samples(input.content)
        return MultiModalMessage(content=input.content, audio=Audio(samples=samples))

    def generate_audio_samples(self, text: str) -> Any:
        """Generate audio samples from text."""
        if self.is_gpu:
            return self.tts.tts(text=text, speaker_wav=self.speaker_wav, language=self.language)
        return self.tts.tts(text=text, speaker_wav=self.speaker_wav)

    def save_audio_to_file(self, text: str, file_path: str = "") -> str:
        """Save the generated audio to a file."""
        if not file_path:
            file_name = f'{date.today().strftime("%Y_%m_%d")}/{int(time.time())}.wav'
            file_path = os.path.join(CACHE_DIR, file_name)
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
        return self.tts.tts_to_file(
            text=text,
            speaker_wav=self.speaker_wav,
            language=self.language if self.is_gpu else None,
            file_path=file_path
        )
