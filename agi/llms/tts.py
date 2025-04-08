import os
import time
from datetime import date
from pathlib import Path
import torch
from TTS.api import TTS
from TTS.utils.radam import RAdam 
from TTS.tts.configs.xtts_config import XttsConfig 
from TTS.tts.models.xtts import XttsAudioConfig,XttsArgs
from TTS.config.shared_configs import BaseDatasetConfig
from collections import defaultdict
from agi.config import TTS_MODEL_DIR as model_root, CACHE_DIR, TTS_SPEAKER_WAV,TTS_GPU_ENABLE,TTS_FILE_SAVE_PATH
from agi.llms.base import CustomerLLM,parse_input_messages,path_to_preview_url
from langchain_core.runnables import RunnableConfig
from typing import Any, Optional,Union
from pydantic import BaseModel, Field

from langchain_core.messages import AIMessage, HumanMessage
import base64

from torch.serialization import add_safe_globals
from agi.config import log

audio_style = "width: 300px; height: 50px;"  # 添加样式
# for torch 2.6
add_safe_globals([RAdam,defaultdict,dict,XttsConfig,XttsAudioConfig,BaseDatasetConfig,XttsArgs])
class TextToSpeech(CustomerLLM):
    tts: Optional[Any] = Field(default=None)
    speaker_wav: str = Field(default=TTS_SPEAKER_WAV)
    is_gpu: bool = Field(default=TTS_GPU_ENABLE)
    language: str = Field(default="zh-cn")
    save_file: bool = Field(default=True)
    
    def __init__(self,save_file: bool = False,**kwargs):
        super().__init__(**kwargs)

        self.save_file = save_file
        self.model = None
       
    def _load_model(self):
        """Initialize the TTS model based on the available hardware."""
        if self.tts is None:
            if self.is_gpu:
                # GPU：2739MB
                log.info("loading TextToSpeech model(GPU)...")
                # model_path = os.path.join(model_root, "tts_models--multilingual--multi-dataset--xtts_v2")
                model_path = model_root
                config_path = os.path.join(model_path, "config.json")
                logging.info("use ts_models--multilingual--multi-dataset--xtts_v2")
                self.tts = TTS(model_path=model_path, config_path=config_path).to(torch.device("cuda"))
            else:
                log.info("loading TextToSpeech model(CPU)...")
                # self.tts = TTS(model_name="tts_models/zh-CN/baker/tacotron2-DDC-GST").to(torch.device("cpu"))
                self.tts = TTS(model_name=model_root).to(torch.device("cpu"))
                # model_dir = os.path.join(model_root, "tts_models--zh-CN--baker--tacotron2-DDC-GST")
                # model_path = os.path.join(model_dir, "model_file.pth")
                # config_path = os.path.join(model_dir, "config.json")
                # return TTS(model_path=model_path, config_path=config_path)
            self.model = self.tts.synthesizer

    def list_available_models(self):
        """Return a list of available TTS models."""
        return self.tts.list_models()
    
    def invoke(self, input: Union[list[HumanMessage],HumanMessage,str], config: Optional[RunnableConfig] = None, **kwargs: Any) -> AIMessage:
        """Generate speech audio from input text."""
        self._load_model()

        input_str = None
        if isinstance(input,str):
            input_str = input
        else:
            _,input_str = parse_input_messages(input)
            
        if self.save_file:
            file_path = self.save_audio_to_file(text=input_str)
            return AIMessage(content=[
                # {"type": "text", "text": input_str},
                {"type": "audio", "audio": file_path,"text":input_str}
            ])
        
        # Generate audio samples and return as ByteIO
        # 原始音频需要编码，不方便使用
        # samples = self.generate_audio_samples(input_str)
        file_path = self.save_audio_to_file(text=input_str)
        audio_source = path_to_preview_url(file_path)
        
        if kwargs.get('base64') is True:
            with open(file_path, 'rb') as audio_file:
                audio_bytes = audio_file.read()
            audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
            audio_source = f"data:audio/wav;base64,{audio_base64}"

        # 是否需要编码html
        if kwargs.get('html') is True:
            return AIMessage(content=[
                {"type": "audio", "audio": f'<audio src="{audio_source}" {audio_style} controls></audio>\n',"file_path":file_path,"text":input_str}
            ])
        
        return AIMessage(content=[
            {"type": "audio", "audio": audio_source,"file_path":file_path,"text":input_str}
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
            # file_name = f'audio/{date.today().strftime("%Y_%m_%d")}/{int(time.time())}.wav'
            file_path = f'{TTS_FILE_SAVE_PATH}/{int(time.time())}.wav'
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