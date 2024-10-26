
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
from agi.llms.base import CustomerLLM
from agi.config import MODEL_PATH as model_root
import hashlib



def calculate_md5(string):
    md5_hash = hashlib.md5()
    md5_hash.update(string.encode('utf-8'))
    md5_value = md5_hash.hexdigest()
    return md5_value

style = 'style="width: 100%; max-height: 100vh;"'
class Text2Image(CustomerLLM):
    model_path: str = Field(None, alias='model_path')
    refiner: Any = None
    n_steps: int = 20
    high_noise_frac: float = 0.8
    file_path: str = "./pics/output"
    save_image = True

    # def __init__(self, model_path: str=os.path.join(model_root,"stable-diffusion"),**kwargs):
    #     super(StableDiff, self).__init__(
    #         llm=DiffusionPipeline.from_pretrained(
    #             model_path, torch_dtype=torch.float16, variant="fp16", use_safetensors=True,
    #     ))
    #     self.model_path = model_path
    #     # self.model = DiffusionPipeline.from_pretrained(
    #     #     "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True,
    #     #     cache_dir=os.path.join(model_root,"stable-diffusion")
    #     # )
    #     # self.model.save_pretrained(os.path.join(model_root,"stable-diffusion"))
        
    #     # self.model = AutoPipelineForText2Image.from_pretrained(model_path, torch_dtype=torch.float16, variant="fp16")
    #     # self.model.scheduler = LCMScheduler.from_config(self.model.scheduler.config)
    #     # self.model.enable_attention_slicing()
    #     # 推理速度变慢
    #     # self.model.unet = torch.compile(self.model.unet, mode="reduce-overhead", fullgraph=True)
    #     # self.model.to(self.device)
    #     # 使用cpu和to('cuda')互斥，内存减小一半
    #     self.model.enable_model_cpu_offload()
    #     # 加速
    #     # adapter_id = "latent-consistency/lcm-lora-sdxl"
    #     # self.model.load_lora_weights(adapter_id)
    #     # self.model.fuse_lora()
    #     # self.model.save_lora_weights(os.path.join(model_root,"stable-diffusion"),unet_lora_layers)

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
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:

        # image = self.model(
        #     **self.get_inputs(prompt)
        # ).images[0]

        image = self.model(prompt=prompt, num_inference_steps=1, guidance_scale=0.0).images[0]

        # file_name = calculate_md5(prompt)+".png"
        out = self.handle_output(image)

        return out

    def get_inputs(self,prompt:str,batch_size=1):
        generator = [torch.Generator("cuda").manual_seed(i) for i in range(batch_size)]
        prompts = batch_size * [prompt]

        return {"prompt": prompts, "generator": generator, "num_inference_steps": self.n_steps}
    
    def handle_output(self,image):
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
        result = f"{formatted_result}"
        return result
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"model_path": self.model_path}
    



# if __name__ == '__main__':
#     sd = Image2Image()
#     output = sd.predict("a strong man",image_path="../../pics/2023_12_09/1702100538.png")
#     print(output)