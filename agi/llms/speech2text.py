from faster_whisper import WhisperModel
import os
from agi.config import (
    WHISPER_MODEL_DIR as model_root,
    WHISPER_GPU_ENABLE
)
from typing import Any, List, Mapping, Optional,Union
from agi.llms.base import CustomerLLM,parse_input_messages
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field
from langchain_core.messages import AIMessage, HumanMessage

from agi.config import log

# GPU 2600MB
class Speech2Text(CustomerLLM):
    whisper: Optional[Any] = Field(default=None)
    beam_size: int = Field(default=5)
    compute_type: str = Field(default="default")
    model_size: str = Field(default="base")
    local_files_only: bool = Field(default=True)
    def __init__(self,device: str = "cuda", compute_type: str = "float16",**kwargs):
        super().__init__(**kwargs)
        
        self.compute_type = compute_type
        if WHISPER_GPU_ENABLE:
            self.model_size = model_root
            self.device = "cuda"
            log.info(model_root)
            if not os.path.exists(self.model_size):
                self.model_size = "large-v3"
                self.local_files_only=False
        else:
            self.device = "cpu"
            self.model_size = model_root
            log.info(model_root)
            self.compute_type = "default"

        
    
    def _load_model(self):
        if self.model is None:
            log.info("loading Speech2Text model...")
            whisper = WhisperModel(self.model_size, device=self.device, compute_type=self.compute_type,local_files_only=self.local_files_only)
            self.whisper = whisper
            self.model = whisper.model

    def invoke(
        self, input: Union[HumanMessage,list[HumanMessage]], config: Optional[RunnableConfig] = None, **kwargs: Any
    ) -> AIMessage:
        """Process the input, transcribe audio, and return the output message."""
        self._load_model()

        audio_input,_ = parse_input_messages(input)
        
        if audio_input is None:
            return AIMessage(content="No valid audio input found.")
        
        # Transcribe the audio input
        content, response_metadata = self._transcribe_audio(audio_input.data)

        # Return the transcription result
        return AIMessage(content=content, response_metadata=response_metadata)


    def _transcribe_audio(self, audio_input) -> str:
        """Transcribe the audio input using Whisper."""
        segments, info = self.whisper.transcribe(audio_input, beam_size=self.beam_size)
        return self._format_transcription(segments, info)

    def _format_transcription(self, segments, info) -> str:
        """Format the transcription into a readable string."""
        content = "".join(segment.text for segment in segments)
        return content, info._asdict()
    
    def print(self, segments, info):
        log.debug("Detected language '%s' with probability %f" % (info.language, info.language_probability))
        for segment in segments:
            log.debug("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
