
import os
import base64
import io
import time
from datetime import date
from pathlib import Path
from diffusers import  AutoPipelineForText2Image
from diffusers import AutoPipelineForImage2Image
from diffusers.utils import load_image
import torch
from langchain.llms.base import LLM
from typing import Any, List, Mapping, Optional,Union
from pydantic import  Field
from agi.llms.base import CustomerLLM,MultiModalMessage,Image
from agi.config import MODEL_PATH as model_root

from langchain_core.runnables import RunnableConfig

style = 'style="width: 100%; max-height: 100vh;"'

class Image2Image(CustomerLLM):
    model_path: str = Field(None, alias='model_path')
    refiner: Any = None
    n_steps: int = 20
    high_noise_frac: float = 0.8
    file_path: str = "./pics/output"
    save_image = True

    def __init__(self, model_path: str=os.path.join(model_root,"sdxl-turbo"),**kwargs):
        if model_path is not None:
            super(Image2Image, self).__init__(
                llm=AutoPipelineForImage2Image.from_pretrained(
                    os.path.join(model_root,"sdxl-turbo"), torch_dtype=torch.float16, variant="fp16"
            ))
            # self.model.save_pretrained(os.path.join(model_root,"sdxl-turbo"))
            self.model_path = model_path
        else:
            super(StableDiff, self).__init__(
                llm=AutoPipelineForText2Image.from_pretrained(
                    "stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16"
            ))
        # self.model.to(self.device)
        # 使用cpu和to('cuda')互斥，内存减小一半
        self.model.enable_model_cpu_offload()

    @property
    def _llm_type(self) -> str:
        return "stabilityai/sdxl-turbo"
    
    @property
    def model_name(self) -> str:
        return "image2image"
    
    def invoke(
        self, input: MultiModalMessage, config: Optional[RunnableConfig] = None, **kwargs: Any
    ) -> MultiModalMessage:
        output = MultiModalMessage(content="")
        if input.content == "" or input.image is None:
            return output
        
        image = None
        prompt = input.content
        input_image = input.image.pil_image.resize((512, 512))
        image = self.model(prompt, image=input_image, num_inference_steps=2, strength=0.5, guidance_scale=0.0).images[0]

        if image is not None:
            output = self.handle_output(image,prompt)
        return output

    def get_inputs(self,prompt:str,batch_size=1):
        generator = [torch.Generator("cuda").manual_seed(i) for i in range(batch_size)]
        prompts = batch_size * [prompt]

        return {"prompt": prompts, "generator": generator, "num_inference_steps": self.n_steps}
    
    def handle_output(self,image,prompt) -> MultiModalMessage:
        img = Image()
        img.pil_image =image
        output = MultiModalMessage(image=image)
        if self.save_image:
            file = f'{date.today().strftime("%Y_%m_%d")}/{int(time.time())}'  # noqa: E501
            output_file = Path(f"{self.file_path}/{file}.png")
            output_file.parent.mkdir(parents=True, exist_ok=True)

            image.save(output_file)
            image_source = f"file/{output_file}"
            output.image = Image.new(image_source)
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
        formatted_result += f'<p> {prompt} </p>'
        output.content = formatted_result
        return output
    
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"model_path": self.model_path}




# if __name__ == '__main__':
#     sd = Image2Image()
#     output = sd.predict("a strong man",image_path="../../pics/2023_12_09/1702100538.png")
#     print(output)