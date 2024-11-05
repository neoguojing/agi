from faster_whisper import WhisperModel
import os
from agi.config import (
    WHISPER_MODEL_DIR as model_root,
    WHISPER_MODEL
)
from typing import Any, List, Mapping, Optional,Union
from agi.llms.base import CustomerLLM,AudioType,Audio
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field
from langchain_core.messages import AIMessage, HumanMessage
import logging
class Speech2Text(CustomerLLM):
    whisper: Optional[Any] = Field(default=None)
    beam_size: int = Field(default=5)
    def __init__(self,device: str = "cuda", compute_type: str = "float16"):
        model_size = None
        if device == "cuda":
            model_size = os.path.join(model_root,"wisper-v3-turbo-c2")
            logging.info("use wisper-v3-turbo-c2")
        else:
            model_size = "base"
            logging.info("use base")
            compute_type = "default"
            
        whisper = WhisperModel(model_size, device=device, compute_type=compute_type)
        super().__init__(llm=whisper.model)
        self.whisper = whisper
        
    def invoke(
        self, input: HumanMessage, config: Optional[RunnableConfig] = None, **kwargs: Any
    ) -> AIMessage:
        segments = None
        info = None
        audio_input = None
        output = AIMessage(content="")
        
        if isinstance(input.content,list):
            for item in input.content:
                media_type = item.get("type")
                if isinstance(media_type,AudioType):
                    data = item.get(media_type)
                    if media_type == AudioType.URL:
                        input = Audio.from_url(data).samples
                    elif media_type == AudioType.FILE_PATH:
                        input = Audio.from_local(data).samples
                    elif media_type == AudioType.NUMPY:
                        # TODO
                        return output
                    elif media_type == AudioType.BYTE_IO:
                        audio_input = data
                        
            segments, info = self.whisper.transcribe(audio_input, beam_size=self.beam_size)
        
            content = ""
            for segment in segments:
                content += segment.text
            
            output = AIMessage(content=content,response_metadata=info._asdict())
        return output

    def print(self, segments, info):
        print("Detected language '%s' with probability %f" % (info.language, info.language_probability))
        for segment in segments:
            print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
