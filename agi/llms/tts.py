import os
import time
from datetime import date
from pathlib import Path
import torch
from TTS.api import TTS
from agi.config import MODEL_PATH as model_root, CACHE_DIR, TTS_SPEAKER_WAV
from agi.llms.base import CustomerLLM, Audio,AudioType
from langchain_core.runnables import RunnableConfig
from typing import Any, Optional
from pydantic import BaseModel, Field
import logging
from langchain_core.messages import AIMessage, HumanMessage

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
            logging.info("use ts_models--multilingual--multi-dataset--xtts_v2")
            return TTS(model_path=model_path, config_path=config_path).to(torch.device("cuda"))
        else:
            logging.info("use tts_models/zh-CN/baker/tacotron2-DDC-GST")
            return TTS(model_name="tts_models/zh-CN/baker/tacotron2-DDC-GST").to(torch.device("cpu"))
            # model_dir = os.path.join(model_root, "tts_models--zh-CN--baker--tacotron2-DDC-GST")
            # model_path = os.path.join(model_dir, "model_file.pth")
            # config_path = os.path.join(model_dir, "config.json")
            # return TTS(model_path=model_path, config_path=config_path)
    def list_available_models(self):
        """Return a list of available TTS models."""
        return self.tts.list_models()
    
    def invoke(self, input: HumanMessage, config: Optional[RunnableConfig] = None, **kwargs: Any) -> AIMessage:
        """Generate speech audio from input text."""
        if not input.content.strip():
            return AIMessage(content="No input text provided.")
        
        if self.save_file:
            file_path = self.save_audio_to_file(text=input.content)
            return AIMessage(content=[
                {"type": "text", "text": input.content},
                {"type": AudioType.FILE_PATH, AudioType.FILE_PATH: file_path}
            ])
        
        # Generate audio samples and return as ByteIO
        samples = self.generate_audio_samples(input.content)
        return AIMessage(content=[
            {"type": "text", "text": input.content},
            {"type": AudioType.BYTE_IO, AudioType.BYTE_IO: samples}
        ])

    def generate_audio_samples(self, text: str) -> Any:
        """Generate audio samples from the input text."""
        try:
            if self.is_gpu:
                return self.tts.tts(text=text, speaker_wav=self.speaker_wav, language=self.language)
            else:
                return self.tts.tts(text=text, speaker_wav=self.speaker_wav)
        except Exception as e:
            logging.error(f"Error generating audio samples: {e}")
            raise RuntimeError("Failed to generate audio samples.")

    def save_audio_to_file(self, text: str, file_path: str = "") -> str:
        """Save the generated audio to a file and return the file path."""
        if not file_path:
            file_name = f'{date.today().strftime("%Y_%m_%d")}/{int(time.time())}.wav'
            file_path = os.path.join(CACHE_DIR, file_name)
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
        try:
            self.tts.tts_to_file(
                text=text,
                speaker_wav=self.speaker_wav,
                language=self.language if self.is_gpu else None,
                file_path=file_path
            )
        except Exception as e:
            logging.error(f"Error saving audio to file: {e}")
            raise RuntimeError("Failed to save audio to file.")
        
        return file_path