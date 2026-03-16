import httpx
from typing import Type, Optional, Any
from pydantic import BaseModel, Field
from langchain.tools import BaseTool
from agi.config import IMAGE_GEN_BASE_URL,API_KEY
# --- 输入 Schema 定义 (保持不变) ---

class Text2ImageInput(BaseModel):
    prompt: str = Field(description="The descriptive text for the image to be generated.")
    model: str = Field(default="sdxl", description="The model to use: 'sdxl' or 'sd3.5'.")
    size: str = Field(default="1024x1024", description="Image resolution: '512x512' or '1024x1024'.")

class Image2ImageInput(BaseModel):
    prompt: str = Field(description="Instructions on how to modify or transform the existing image.")
    image_url: str = Field(description="The URL of the source image to be modified.")

# --- 工具类定义 ---

class RemoteImageGenTool(BaseTool):
    name: str = "remote_image_generation"
    description: str = "Generate a new image from a text description via a remote API."
    args_schema: Type[BaseModel] = Text2ImageInput
    
    api_base_url: str = Field(IMAGE_GEN_BASE_URL, exclude=True)
    api_key: str = Field(API_KEY, exclude=True)
    timeout: float = 60.0  # 图像生成通常较慢，设置较长的超时时间

    def _run(self, prompt: str, model: str = "sdxl", size: str = "1024x1024") -> str:
        url = f"{self.api_base_url}/v1/images/generations"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {
            "model": model,
            "prompt": prompt,
            "size": size,
            "response_format": "url"
        }
        
        # 使用 httpx.Client 进行同步调用
        with httpx.Client(timeout=self.timeout) as client:
            try:
                response = client.post(url, json=payload, headers=headers)
                response.raise_for_status()
                result = response.json()
                return result["data"][0]["url"]
            except httpx.HTTPStatusError as e:
                return f"API Status Error ({e.response.status_code}): {e.response.text}"
            except Exception as e:
                return f"API Connection Error: {str(e)}"

class RemoteImageEditTool(BaseTool):
    name: str = "remote_image_editing"
    description: str = "Modify or transform an existing image based on text instructions via a remote API."
    args_schema: Type[BaseModel] = Image2ImageInput
    
    api_base_url: str = Field(..., exclude=True)
    api_key: str = Field(..., exclude=True)
    timeout: float = 60.0

    def _run(self, prompt: str, image_url: str) -> str:
        url = f"{self.api_base_url}/v1/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_url}}
                ]
            }]
        }
        
        with httpx.Client(timeout=self.timeout) as client:
            try:
                response = client.post(url, json=payload, headers=headers)
                response.raise_for_status()
                result = response.json()
                return result["choices"][0]["message"]["content"]
            except httpx.HTTPStatusError as e:
                return f"API Status Error ({e.response.status_code}): {e.response.text}"
            except Exception as e:
                return f"API Connection Error: {str(e)}"