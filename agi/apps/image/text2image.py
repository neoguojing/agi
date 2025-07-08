import threading
import time
import os
import base64
import io
from pathlib import Path
from typing import Any
import soundfile as sf
import torch
from agi.config import TEXT_TO_IMAGE_VERSION,BASE_URL,CACHE_DIR
from agi.config import TEXT_TO_IMAGE_MODEL_PATH as model_root,IMAGE_FILE_SAVE_PATH
from agi.apps.common import path_to_preview_url

style = 'style="width: 100%; max-height: 100vh;"'

class Text2Image:
    n_steps: int = 1
    guidance_scale: float = 0.0
    def __init__(self, model_path: str=model_root, timeout: int = 300,save_image=True):
        """
        model_path: 模型本地路径或 huggingface 名称
        timeout: 超过多少秒未使用就自动卸载（默认10分钟）
        """
        self.model_path = model_path
        self.timeout = timeout
        self.model = None
        self.last_used = 0
        self.lock = threading.Lock()
        self.monitor_thread = threading.Thread(target=self._monitor, daemon=True)
        self.monitor_thread.start()
        self.save_image = save_image
        self.file_path = IMAGE_FILE_SAVE_PATH

    def get_model(self):
        """访问模型，如果未加载则自动加载"""
        with self.lock:
            self.last_used = time.time()
            if self.model is None:
                self._load()
            return self.model

    def _load(self):
        print(f"[Model] Loading model from {self.model_path}")
        if TEXT_TO_IMAGE_VERSION == "sdxl-turbo":
            from diffusers import AutoPipelineForText2Image
            self.model = AutoPipelineForText2Image.from_pretrained(
                    self.model_path, torch_dtype=torch.float16
            )
        elif TEXT_TO_IMAGE_VERSION == "3.5-medium":
            # use 3.5 model
            # GPU 18000MB -> 900MB(off-load)
            from diffusers import StableDiffusion3Pipeline
            self.n_steps = 20
            self.guidance_scale = 7.0
            self.model = StableDiffusion3Pipeline.from_pretrained(self.model_path, torch_dtype=torch.bfloat16)
            self.model = self.model.to("cuda")
            
        self.model.enable_model_cpu_offload()

    def invoke(self, input: str,resp_format="url") -> str:
        """Generate an image from the input text."""
        self.get_model()

        if not input.strip():
            return "No prompt provided."
        
        # Generate image from the provided prompt
        image = self.model(prompt=input, num_inference_steps=self.n_steps, guidance_scale=self.guidance_scale).images[0]
        
        if resp_format == "b64_json":
            self.save_image = False
        else:
            self.save_image = True
        # Handle and format the image for output
        return self.handle_output(image)
    
    def handle_output(self, image: Any,html:bool=False) -> str:
        """Handle the image output (save or return base64)."""
        image_source = self._save_or_resize_image(image)
        image_source = path_to_preview_url(image_source)
        # Format the result as HTML with embedded image and prompt
        if html:
            image_source = f'<img src="{image_source}" {style}>\n'

        return image_source

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

    def _unload(self):
        print(f"[Model] Unloading model from {self.model_path}")
        del self.model
        self.model = None
        torch.cuda.empty_cache()

    def _monitor(self):
        """后台线程定期检查是否应卸载模型"""
        while True:
            time.sleep(30)
            with self.lock:
                if self.model and (time.time() - self.last_used > self.timeout):
                    self._unload()
