from typing import Any, List, Mapping, Optional, Union
from pydantic import Field,ConfigDict
from agi.llms.base import CustomerLLM,parse_input_messages
from agi.config import API_KEY,IMAGE_GEN_BASE_URL
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import AIMessage, HumanMessage
from openai import OpenAI

from agi.config import log

# GPU : 942MB
class Image2Image(CustomerLLM):
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
        return "image2image"
    
    def invoke(self, input: Union[HumanMessage,list[HumanMessage]], config: Optional[RunnableConfig] = None, **kwargs: Any) -> AIMessage:
        output = AIMessage(content="")

        # Extract image and text prompt from input content
        input_image, prompt = parse_input_messages(input)
        
        if input_image is None:
            return output  # No valid image found in input
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        { "type": "text", "text": prompt},
                        { "type": "image_url", "image_url": { "url": input_image} }
                    ],
                }
            ],
        )
        
        log.info(response)
        output.content = response.choices[0].message.content
        return output
    
