
from agi.config import API_KEY,TTS_BASE_URL,TTS_FILE_SAVE_PATH
from agi.llms.base import CustomerLLM,parse_input_messages,path_to_preview_url
from langchain_core.runnables import RunnableConfig,run_in_executor
from typing import Any, Optional,Union,ClassVar
from langchain_core.messages import AIMessage, HumanMessage,AIMessageChunk
from agi.config import log
from openai import OpenAI
from pydantic import  Field,ConfigDict
import tempfile


class TextToSpeech(CustomerLLM):
    client: OpenAI = Field(None, alias='client')
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.client = OpenAI(api_key=API_KEY,base_url=TTS_BASE_URL)


    @property
    def model_name(self) -> str:
        return "tts"

    def invoke(self, input: Union[list[HumanMessage],HumanMessage,str], config: Optional[RunnableConfig] = None, **kwargs: Any) -> AIMessage:
        """Generate speech audio from input text."""

        user_id = "default"
        model_name = "cosyvoice"
        if config:
            user_id = config.get("configurable").get("user_id")
            model_name = config.get("configurable").get("model","cosyvoice")

        input_str = None
        if isinstance(input,str):
            input_str = input
        else:
            _,input_str = parse_input_messages(input)
            
        log.info(f"tts input: {input_str}")
        response = self.client.audio.speech.create(
            model=model_name,                     # 或 "tts-1-hd"
            voice="alloy",                    # 支持 alloy, echo, fable, onyx, nova, shimmer
            input=input_str,
            response_format="wav",           # 可选 "mp3", "opus", "aac", "flac"
            extra_body={"user": user_id},
        )

        # 保存为文件
        tmp_path = ""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(response.content)
            tmp_path = tmp.name

        return AIMessage(content=[
            {"type": "audio", "audio": tmp_path,"text":input_str}
        ],response_metadata={"finish_reason":"stop"})
        
    def stream(self, input: Union[list[HumanMessage],HumanMessage,str], config: Optional[RunnableConfig] = None, **kwargs: Any):
        
        user_id = "default"
        model_name = "cosyvoice"
        if config:
            user_id = config.get("configurable").get("user_id")
            model_name = config.get("configurable").get("model","cosyvoice")

        input_str = None
        if isinstance(input,str):
            input_str = input
        else:
            _,input_str = parse_input_messages(input)
            
        log.info(f"tts input: {input_str}")
        response = self.client.audio.speech.create(
            model=model_name,                     # 或 "tts-1-hd"
            voice="alloy",                    # 支持 alloy, echo, fable, onyx, nova, shimmer
            input=input_str,
            response_format="wav",           # 可选 "mp3", "opus", "aac", "flac"
            extra_body={"user": user_id,"stream":True},
        )

        # 保存为文件
        for chunk in response.iter_bytes():
            yield AIMessage(content=[
                {"type": "audio", "audio": chunk}
            ])

        yield AIMessage(content=[
                {"type": "audio", "audio": None}
            ],response_metadata={"finish_reason":"stop"})

