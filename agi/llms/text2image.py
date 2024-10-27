
import os
import base64
import io
import time
from datetime import date
from pathlib import Path
from diffusers import  AutoPipelineForText2Image
import torch
from langchain.llms.base import LLM
from typing import Any, List, Mapping, Optional,Union
from langchain.callbacks.manager import (
    CallbackManagerForLLMRun
)
from pydantic import  Field
from agi.llms.base import CustomerLLM,MultiModalMessage,Image
from agi.config import MODEL_PATH as model_root,CACHE_DIR
import hashlib
from langchain_core.runnables import RunnableConfig

style = 'style="width: 100%; max-height: 100vh;"'
class Text2Image(CustomerLLM):
    model_path: str = Field(None, alias='model_path')
    refiner: Any = None
    n_steps: int = 20
    high_noise_frac: float = 0.8
    file_path: str = CACHE_DIR
    save_image: bool = True

    def __init__(self, model_path: str=os.path.join(model_root,"sdxl-turbo"),**kwargs):
        if model_path is not None:
            super(Text2Image, self).__init__(
                llm=AutoPipelineForText2Image.from_pretrained(
                    os.path.join(model_root,"sdxl-turbo"), torch_dtype=torch.float16, variant="fp16"
            ))
            self.model_path = model_path
            # self.model.to(self.device)
            # 使用cpu和to('cuda')互斥，内存减小一半
            self.model.enable_model_cpu_offload()
        else:
            super(Text2Image, self).__init__(
                llm=AutoPipelineForText2Image.from_pretrained(
                    "stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16"
            ))
            self.model.enable_model_cpu_offload()

    @property
    def _llm_type(self) -> str:
        return "stabilityai/sdxl-turbo"
    
    @property
    def model_name(self) -> str:
        return "text2image"
    
    def invoke(
        self, input: MultiModalMessage, config: Optional[RunnableConfig] = None, **kwargs: Any
    ) -> MultiModalMessage:
        out = MultiModalMessage(content="")
        if input.content == "":
            return out
        
        image = self.model(prompt=input.content, num_inference_steps=1, guidance_scale=0.0).images[0]
        out = self.handle_output(image)

        return out
    
    def handle_output(self,image) -> MultiModalMessage:
        if self.save_image:
            file = f'{date.today().strftime("%Y_%m_%d")}/{int(time.time())}'  # noqa: E501
            output_file = Path(f"{self.file_path}/{file}.png")
            output_file.parent.mkdir(parents=True, exist_ok=True)

            image.save(output_file)
            image_source = f"file/{output_file}"
        else:
            # resize image to avoid huge logs
            image.thumbnail((512, 512 * image.height / image.width))
            buffered = io.BytesIO()
            image.save(buffered, format="JPEG")
            buffered.seek(0)
            image_bytes = buffered.getvalue()
            image_base64 = (
                "data:image/jpeg;base64," + base64.b64encode(image_bytes).decode()
            )
            image_source = image_base64

        formatted_result = f'<img src="{image_source}" {style}>\n'
        result = MultiModalMessage(content=formatted_result,image=Image(pil_image=image))
        return result
    
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"model_path": self.model_path}
    
