from typing import Any, Union, Literal, List, Dict
from langchain_core.runnables import RunnableSerializable
from langchain_core.messages import AIMessage, HumanMessage
import os
from agi.utils.common import path_to_preview_url
from agi.config import log

# 从用户消息中抽取content的内容，转换为模型可处理的格式
def parse_input_messages(input: Union[HumanMessage,list[HumanMessage]]):
    """
    Parse the content of the HumanMessage to extract the image and prompt.
    """
    media = None
    prompt = None
    input_type = "text"
    if isinstance(input, list):
        input = input[-1]
        
    if isinstance(input.content, list):
        for content in input.content:
            media_type = content.get("type")
            if media_type == "image":
                # Create Image instance based on media type
                media_data = content.get("image")
                if media_data is not None and media_data != "":
                    input_type = "image"
                    media = media_data
            if media_type == "audio":
                # Create Image instance based on media type
                media_data = content.get("audio")
                if media_data is not None and media_data != "":
                    input_type = "audio"
                    media = media_data
            if media_type == "video":
                # Create Image instance based on media type
                media_data = content.get("video")
                if media_data is not None and media_data != "":
                    input_type = "video"
                    media = media_data
            elif media_type == "text":
                prompt = content.get("text")
    elif isinstance(input.content, str):
        prompt = input.content

    return media, prompt,input_type

# Custom LLM class for integration with runnable modules
class CustomerLLM(RunnableSerializable[HumanMessage, AIMessage]):
    model: Any = None
    tokenizer: Any = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = None
    

    def destroy(self):
        if self.model is not None:
            del self.model
        log.info(f"Model {self.model_name} destroyed successfully")

    def encode(self, input):
        if self.tokenizer is not None:
            return self.tokenizer.encode(input)
        return None
        
    def decode(self, ids):
        if self.tokenizer is not None:
            return self.tokenizer.decode(ids)
        return ""

    @property
    def model_name(self) -> str:
        return ""  # Model name placeholder (to be customized)
