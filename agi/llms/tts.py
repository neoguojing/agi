
from agi.config import API_KEY,TTS_BASE_URL,TTS_FILE_SAVE_PATH
from agi.llms.base import CustomerLLM,parse_input_messages,path_to_preview_url
from langchain_core.runnables import RunnableConfig,run_in_executor
from typing import Any, Optional,Union,ClassVar
from langchain_core.messages import AIMessage, HumanMessage,AIMessageChunk
from agi.config import log
from openai import OpenAI
from pydantic import  Field

class TextToSpeech(CustomerLLM):
    client: OpenAI = Field(None, alias='client')

    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.client = OpenAI(api_key=API_KEY,base_url=TTS_BASE_URL)


    @property
    def model_name(self) -> str:
        return "tts"

    def invoke(self, input: Union[list[HumanMessage],HumanMessage,str], config: Optional[RunnableConfig] = None, **kwargs: Any) -> AIMessage:
        """Generate speech audio from input text."""

        user_id = config.get("configurable").get("user_id")
        input_str = None
        if isinstance(input,str):
            input_str = input
        else:
            _,input_str = parse_input_messages(input)
            
        log.info(f"tts input: {input_str}")
        response = self.client.audio.speech.create(
            model="tts-1",                     # 或 "tts-1-hd"
            voice="alloy",                    # 支持 alloy, echo, fable, onyx, nova, shimmer
            input=input_str,
            response_format="wav",           # 可选 "mp3", "opus", "aac", "flac"
            extra_body={"user": user_id},
        )

        # 保存为文件
        with open("output.mp3", "wb") as f:
            f.write(response.content)

        return AIMessage(content=[
            {"type": "audio", "audio": audio_source,"text":input_str}
        ],response_metadata={"finish_reason":"stop"})
        
    async def ainvoke(self, input: Union[list[HumanMessage],HumanMessage,str], config: Optional[RunnableConfig] = None, **kwargs: Any) -> AIMessage:
        log.debug("tts ainvoke ---------------")
        # return self.invoke(input, config=config, **kwargs)
        return await run_in_executor(config, self.invoke, input, config, **kwargs)


