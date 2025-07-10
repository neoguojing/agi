from agi.config import (
    API_KEY,
    WHISPER_BASE_URL
)
from typing import Any, List, Mapping, Optional,Union
from agi.llms.base import CustomerLLM,parse_input_messages
from langchain_core.runnables import RunnableConfig
from pydantic import ConfigDict, Field
from langchain_core.messages import AIMessage, HumanMessage
from agi.config import log
from openai import OpenAI

# GPU 2600MB
class Speech2Text(CustomerLLM):
    client: OpenAI = Field(None, alias='client')
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.client = OpenAI(api_key=API_KEY,base_url=WHISPER_BASE_URL)

    @property
    def model_name(self) -> str:
        return "Speech2Text"

    def invoke(
        self, input: Union[HumanMessage,list[HumanMessage]], config: Optional[RunnableConfig] = None, **kwargs: Any
    ) -> AIMessage:
        """Process the input, transcribe audio, and return the output message."""

        audio_input,_ = parse_input_messages(input)
        
        if audio_input is None:
            return AIMessage(content="No valid audio input found.")
        
        # Transcribe the audio input
        transcription = self.client.audio.transcriptions.create(
                model="whisper-1",            # 模型名，必须是 "whisper-1"
                file=audio_input,
                response_format="json",       # 可选：json | text | srt | verbose_json | vtt
                language="zh",                # 可选：指定语言，如中文“zh”
                temperature=0.0               # 可选
        )

        log.info(f"speech to text:{transcription}")
        # Return the transcription result
        return AIMessage(content=transcription.text)
