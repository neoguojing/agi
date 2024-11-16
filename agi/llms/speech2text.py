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
        """Process the input, transcribe audio, and return the output message."""
        audio_input = self._process_audio_input(input)
        
        if audio_input is None:
            return AIMessage(content="No valid audio input found.")
        
        # Transcribe the audio input
        content, response_metadata = self._transcribe_audio(audio_input)

        # Return the transcription result
        return AIMessage(content=content, response_metadata=response_metadata)

    def _process_audio_input(self, input: HumanMessage) -> Optional[Audio]:
        """Extract audio data from the input."""
        audio_input = None
        if isinstance(input.content, list):
            for item in input.content:
                media_type = item.get("type")
                if isinstance(media_type, AudioType):
                    data = item.get(media_type)
                    if media_type == AudioType.URL:
                        audio_input = Audio.from_url(data).samples
                    elif media_type == AudioType.FILE_PATH:
                        audio_input = Audio.from_local(data).samples
                    elif media_type == AudioType.NUMPY:
                        # Handle Numpy audio input (TODO)
                        return None
                    elif media_type == AudioType.BYTE_IO:
                        audio_input = data
        return audio_input

    def _transcribe_audio(self, audio_input) -> str:
        """Transcribe the audio input using Whisper."""
        segments, info = self.whisper.transcribe(audio_input, beam_size=self.beam_size)
        return self._format_transcription(segments, info)

    def _format_transcription(self, segments, info) -> str:
        """Format the transcription into a readable string."""
        content = "".join(segment.text for segment in segments)
        return content, info._asdict()
    
    def print(self, segments, info):
        print("Detected language '%s' with probability %f" % (info.language, info.language_probability))
        for segment in segments:
            print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
