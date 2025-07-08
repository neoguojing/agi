from agi.config import API_KEY,MULTI_MODEL_BASE_URL
from agi.llms.base import CustomerLLM
from langchain_core.runnables import RunnableConfig
from typing import Any, Optional,Union
from pydantic import BaseModel, Field

from langchain_core.messages import AIMessage, HumanMessage
import traceback
from agi.config import log
from openai import OpenAI

# GPU: 3B 13GB
class MultiModel(CustomerLLM):
    client: OpenAI = Field(None, alias='client')
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.client = OpenAI(api_key=API_KEY,base_url=MULTI_MODEL_BASE_URL)

    
    @property
    def model_name(self) -> str:
        return "MultiModel"
    
    def invoke(self, inputs: Union[list[HumanMessage],HumanMessage], config: Optional[RunnableConfig] = None, **kwargs: Any) -> AIMessage:
        """Generate speech audio from input text."""
        try:
            return_audio = False
            if config:
                return_audio = config.get("configurable",{}).get("need_speech",False)

            content = None
            if isinstance(inputs,HumanMessage):
                content = inputs.content
            elif isinstance(inputs,list):
                inputs = inputs[-1]
                content = inputs.content
            else:
                raise TypeError(f"Invalid input type: {type(inputs)}. Expected HumanMessage or List[HumanMessage].") 
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                extra_body={"need_speech": return_audio},
                messages=[
                    {
                        "role": "user",
                        "content": content
                    }
                ],
            )
            
            log.info(response)
            ret = AIMessage(content=response.choices[0].message.content)
            return ret
                        
        except Exception as e:
            log.error(e)
            print(traceback.format_exc())

