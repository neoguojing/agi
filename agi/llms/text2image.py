
import os
import base64
import io
import time
from datetime import date
from pathlib import Path
from diffusers import  AutoPipelineForText2Image,StableDiffusion3Pipeline
import torch
from langchain.llms.base import LLM
from typing import Any, List, Mapping, Optional,Union
from langchain.callbacks.manager import (
    CallbackManagerForLLMRun
)
from pydantic import  Field
from agi.llms.base import CustomerLLM,parse_input_messages,path_to_preview_url
from agi.config import TEXT_TO_IMAGE_MODEL_PATH as model_root,IMAGE_FILE_SAVE_PATH
import hashlib
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import AIMessage, HumanMessage

from agi.config import log


style = 'style="width: 100%; max-height: 100vh;"'
class Text2Image(CustomerLLM):
    model_path: str = Field(None, alias='model_path')
    refiner: Any = None
    # n_steps: int = 28
    n_steps: int = 1
    high_noise_frac: float = 0.8
    # guidance_scale: float = 7.0
    guidance_scale: float = 0.0
    file_path: str = IMAGE_FILE_SAVE_PATH
    save_image: bool = True

    def __init__(self, model_path: str=model_root,**kwargs):
        super().__init__(**kwargs)
        self.model_path = model_path
        
           

    def _load_model(self):
        if self.model is None:
            log.info("loading Text2Image model...")
            if self.model_path is not None:
                self.model = AutoPipelineForText2Image.from_pretrained(
                        model_root, torch_dtype=torch.float16
                )
                # use 3.5 model
                # GPU 18000MB -> 900MB(off-load)
                # self.model = StableDiffusion3Pipeline.from_pretrained(model_root, torch_dtype=torch.bfloat16)
                # self.model = self.model.to("cuda")

            else:
                self.model = AutoPipelineForText2Image.from_pretrained(
                    "stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16"
                )

            self.model.enable_model_cpu_offload()

    @property
    def _llm_type(self) -> str:
        return "stabilityai/sdxl-turbo"
    
    @property
    def model_name(self) -> str:
        return "text2image"
    
    
    def invoke(self, input: Union[list[HumanMessage],HumanMessage,str], config: Optional[RunnableConfig] = None, **kwargs: Any) -> AIMessage:
        """Generate an image from the input text."""
        self._load_model()
        # Check if input is empty
        input_str = ""
        # log.debug("#########",input)
        if isinstance(input,str):
            input_str = input
        else:
            _, input_str = parse_input_messages(input)
            
        if not input_str.strip():
            return AIMessage(content="No prompt provided.")
        
        # Generate image from the provided prompt
        image = self._generate_image(input_str)
        
        # Handle and format the image for output
        return self.handle_output(image)

    def _generate_image(self, prompt: str) -> Any:
        """Generate an image based on the given prompt."""
        # Adjust the number of inference steps based on desired quality
        image = self.model(prompt=prompt, num_inference_steps=self.n_steps, guidance_scale=self.guidance_scale).images[0]
        return image

    def handle_output(self, image: Any,html:bool=False) -> AIMessage:
        """Handle the image output (save or return base64)."""
        image_source = self._save_or_resize_image(image)
        image_source = path_to_preview_url(image_source)
        # Format the result as HTML with embedded image and prompt
        if html:
            image_source = f'<img src="{image_source}" {style}>\n'
            
        result = AIMessage(content=[{"type": "image", "image": image_source}])
        # log.debug("#########1",result)
        return result

    def _save_or_resize_image(self, image: Any) -> str:
        """Save the image to disk or convert it to base64, depending on settings."""
        if self.save_image:
            return self._save_image(image)
        else:
            return self._convert_image_to_base64(image)

    def _save_image(self, image: Any) -> str:
        """Save the generated image to the file system."""
        # file_name = f'image/{date.today().strftime("%Y_%m_%d")}/{int(time.time())}.png'
        file_name = f'{int(time.time())}.png'
        output_file = Path(self.file_path) / file_name
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Save the image to the file system
        image.save(output_file)
        return f"{output_file}"

    def _convert_image_to_base64(self, image: Any) -> str:
        """Convert the image to a base64-encoded string."""
        image.thumbnail((512, 512 * image.height / image.width))
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        buffered.seek(0)
        image_bytes = buffered.getvalue()
        return f"data:image/jpeg;base64,{base64.b64encode(image_bytes).decode()}"

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Return identifying parameters for the model."""
        return {"model_path": self.model_path}