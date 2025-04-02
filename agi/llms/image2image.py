import os
import base64
import io
import time
from datetime import date
from pathlib import Path
from diffusers import AutoPipelineForImage2Image
import torch
from typing import Any, List, Mapping, Optional, Union
from pydantic import Field
from agi.llms.base import CustomerLLM,parse_input_messages,path_to_preview_url
from agi.config import IMAGE_TO_IMAGE_MODEL_PATH as model_root, CACHE_DIR,IMAGE_FILE_SAVE_PATH
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import AIMessage, HumanMessage
from PIL import Image as PILImage
# HTML style for rendering image
STYLE = 'style="width: 100%; max-height: 100vh;"'
import logging
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
class Image2Image(CustomerLLM):
    model_path: str = Field(default=model_root, alias='model_path')
    refiner: Optional[Any] = None
    n_steps: int = 20
    high_noise_frac: float = 0.8
    file_path: str = IMAGE_FILE_SAVE_PATH
    save_image: bool = True

    def __init__(self, model_path: str = model_root, **kwargs):
        super().__init__(**kwargs)
        
        self.model_path = model_path
    
    def _load_model(self):
        if self.model is None:
            self.model = AutoPipelineForImage2Image.from_pretrained(
                self.model_path, torch_dtype=torch.float16
            )
            # Enable CPU offloading for the model (optimize memory usage)
            self.model.enable_model_cpu_offload()

    @property
    def _llm_type(self) -> str:
        return "stabilityai/sdxl-turbo"
    
    @property
    def model_name(self) -> str:
        return "image2image"
    
    def invoke(self, input: Union[HumanMessage,list[HumanMessage]], config: Optional[RunnableConfig] = None, **kwargs: Any) -> AIMessage:
        self._load_model()

        output = AIMessage(content="")

        # Extract image and text prompt from input content
        input_image, prompt = parse_input_messages(input)
        
        if input_image is None:
            return output  # No valid image found in input

        # Resize image to fit model input requirements
        input_image = input_image.data.resize((512, 512))
        
        # Generate the image using the model
        generated_image = self.model(prompt, image=input_image, num_inference_steps=2, strength=0.5, guidance_scale=0.0).images[0]
        
        if generated_image is not None:
            output = self.handle_output(generated_image, prompt)

        return output
    
    
    def handle_output(self, image: PILImage.Image, prompt: str,html:bool = False) -> AIMessage:
        """
        Handle the output by either saving the image or returning a base64-encoded image.
        """
        if self.save_image:
            # Save image to the file system
            # file_name = f'image/{date.today().strftime("%Y_%m_%d")}/{int(time.time())}.png'
            file_name = f'{int(time.time())}.png'
            output_file = Path(self.file_path) / file_name
            output_file.parent.mkdir(parents=True, exist_ok=True)

            image.save(output_file)
            url = path_to_preview_url(output_file)
            image_source = f"{url}"
        else:
            # Resize image and convert to base64
            image.thumbnail((512, 512 * image.height / image.width))
            buffered = io.BytesIO()
            image.save(buffered, format="JPEG")
            buffered.seek(0)
            image_bytes = buffered.getvalue()
            image_base64 = "data:image/jpeg;base64," + base64.b64encode(image_bytes).decode()
            image_source = image_base64

        # Create HTML-embedded response
        if html:
            image_source = f'<img src="{image_source}" {STYLE}>\n'
            image_source += f'<p>{prompt}</p>'

        # Return AIMessage containing formatted image response and the image itself
        return AIMessage(content=[{"type": "image", "image": image_source}])
    
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"model_path": self.model_path}
