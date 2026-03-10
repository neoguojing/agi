from agi.config import API_KEY,MULTI_MODEL_BASE_URL,MULTI_MODEL_NAME
from agi.llms.base import CustomerLLM,parse_input_messages
from agi.utils.common import path_to_preview_url
from langchain_core.runnables import RunnableConfig
from typing import Any, Optional,Union
from pydantic import ConfigDict, Field

from langchain_core.messages import AIMessage, HumanMessage
import traceback
from agi.config import log
from openai import OpenAI
import os

# GPU: 3B 13GB
class MultiModel(CustomerLLM):
    client: OpenAI = Field(None, alias='client')
    model_config = ConfigDict(arbitrary_types_allowed=True)

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

            media,prompt,media_type = parse_input_messages(inputs)
            if media and os.path.exists(media):
                media = path_to_preview_url(media)
                
            response = self.client.chat.completions.create(
                model = os.environ.get("MULTI_MODEL_NAME", "gemma") ,
                extra_body={"need_speech": return_audio},
                messages=[
                    {
                        "role": "user",
                        "content": [{"type":"text","text":prompt},{"type":media_type,media_type:media}]
                    }
                ],
            )
            
            log.info(response)
            content = [{"type":"text","text":response.choices[0].message.content}]
            if response.choices[0].message.audio:
                content.insert(0, {"type":"audio","audio":response.choices[0].message.audio.data})
            ret = AIMessage(content=content)
            return ret
                        
        except Exception as e:
            log.error(e)
            print(traceback.format_exc())

