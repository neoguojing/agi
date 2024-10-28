from faster_whisper import WhisperModel
import os
from agi.config import (
    WHISPER_MODEL_DIR as model_root,
    WHISPER_MODEL
)
from typing import Any, List, Mapping, Optional,Union
from agi.llms.base import CustomerLLM,MultiModalMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

class Speech2Text(CustomerLLM):
    whisper: Optional[Any] = Field(default=None)
    beam_size: int = Field(default=5)
    def __init__(self,device: str = "cuda", compute_type: str = "float16"):
        model_size = None
        if device == "cuda":
            model_size = os.path.join(model_root,"wisper-v3-turbo-c2")
        else:
            model_size = "base"
            
        whisper = WhisperModel(model_size, device=device, compute_type=compute_type)
        super().__init__(llm=whisper.model)
        self.whisper = whisper
        
    def invoke(
        self, input: MultiModalMessage, config: Optional[RunnableConfig] = None, **kwargs: Any
    ) -> MultiModalMessage:
        
        segments, info = self.whisper.transcribe(input.audio.samples, beam_size=self.beam_size)
        
        content = ""
        for segment in segments:
            content += segment.text
        
        output = MultiModalMessage(content=content,response_metadata=info._asdict())
        return output

    def print(self, segments, info):
        print("Detected language '%s' with probability %f" % (info.language, info.language_probability))
        for segment in segments:
            print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
