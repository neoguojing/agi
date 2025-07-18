from typing import Any, List, Mapping, Optional,Union
from pydantic import  Field,ConfigDict
from agi.llms.base import CustomerLLM,parse_input_messages
from agi.config import API_KEY,IMAGE_GEN_BASE_URL,TEXT_TO_IMAGE_MODEL_NAME
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import AIMessage, HumanMessage
from openai import OpenAI
from agi.config import log


class Text2Image(CustomerLLM):
    client: OpenAI = Field(None, alias='client')
    model_config = ConfigDict(arbitrary_types_allowed=True)


    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.client = OpenAI(api_key=API_KEY,base_url=IMAGE_GEN_BASE_URL)

    @property
    def _llm_type(self) -> str:
        return "stabilityai/sdxl-turbo"
    
    @property
    def model_name(self) -> str:
        return "text2image"
    
    
    def invoke(self, input: Union[list[HumanMessage],HumanMessage,str], config: Optional[RunnableConfig] = None, **kwargs: Any) -> AIMessage:
        """Generate an image from the input text."""
        # Check if input is empty
        input_str = ""
        # log.debug("#########",input)
        if isinstance(input,str):
            input_str = input
        else:
            _, input_str,_ = parse_input_messages(input)
            
        if not input_str.strip():
            return AIMessage(content="No prompt provided.")
        
        model_name = TEXT_TO_IMAGE_MODEL_NAME
        if config:
            model_name = config.get("configurable").get("model",TEXT_TO_IMAGE_MODEL_NAME)
        
        response = self.client.images.generate(
            model=model_name,  # 可选 "" 或 "dall-e-3"
            prompt=input_str,
            size="1024x1024",  # 可选：1024x1024、512x512（dall-e-2 支持更多尺寸）
            quality="hd",      # 仅用于 DALL·E 3，可选 "standard" | "hd"
            n=1,               # 生成图数量
            response_format="url"  # 或 "b64_json"
        )

        print(response.data[0].url)
        
        # Handle and format the image for output
        return AIMessage(content=[{"type":"image","image":response.data[0].url}])
