import httpx
from typing import Type, Optional, Any, List, Union
from pydantic import BaseModel, Field
from langchain.tools import BaseTool
from agi.config import MULTI_MODEL_BASE_URL,API_KEY

class MultiModalInput(BaseModel):
    text: str = Field(description="The text prompt or question for the assistant.")
    model: str = Field(default="gemma", description="Model name: 'gemma' or 'qwen' (best for audio).")
    image_url: Optional[str] = Field(default=None, description="URL of an image to analyze.")
    audio_url: Optional[str] = Field(default=None, description="URL of an audio file to perceive.")
    video_url: Optional[str] = Field(default=None, description="URL of a video file to analyze.")
    need_speech: bool = Field(default=False, description="Whether to request an audio speech response from the model.")


class RemoteMultiModalTool(BaseTool):
    name: str = "remote_multimodal_omni_assistant"
    description: str = (
        "A versatile assistant that can see, hear, and talk. "
        "Use this for questions about images, audio recordings, videos, "
        "or when the user expects an integrated voice-and-text response."
    )
    args_schema: Type[BaseModel] = MultiModalInput
    
    api_base_url: str = Field(MULTI_MODEL_BASE_URL, exclude=True)
    api_key: str = Field(API_KEY, exclude=True)
    timeout: float = 90.0

    def _run(
        self, 
        text: str, 
        model: str = "gemma",
        image_url: Optional[str] = None,
        audio_url: Optional[str] = None,
        video_url: Optional[str] = None,
        need_speech: bool = False
    ) -> str:
        """调用 /v1/chat/completions 接口执行多模态推理"""
        url = f"{self.api_base_url}/v1/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        
        # 1. 构建符合后端 ChatCompletionRequest 的 content 列表
        content_list = [{"type": "text", "text": text}]
        
        if image_url:
            content_list.append({"type": "image", "image": image_url})
        if audio_url:
            content_list.append({"type": "audio", "audio": audio_url})
        if video_url:
            content_list.append({"type": "video", "video": video_url})

        # 2. 构造 Payload
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": content_list}],
            "need_speech": need_speech
        }
        
        # 3. 发送请求
        with httpx.Client(timeout=self.timeout) as client:
            try:
                response = client.post(url, json=payload, headers=headers)
                response.raise_for_status()
                result = response.json()
                
                # 4. 解析响应
                choice = result["choices"][0]["message"]
                final_text = choice.get("content", "")
                
                # 如果有语音响应，将音频数据（HTML 或 URL）附加上去
                if choice.get("audio"):
                    audio_info = choice["audio"]
                    # 这里的 data 可能是后端生成的 HTML <audio> 标签或 URL
                    final_text += f"\n\n[Audio Response Output]: {audio_info.get('data')}"
                
                return final_text

            except httpx.HTTPStatusError as e:
                return f"Multimodal API Error ({e.response.status_code}): {e.response.text}"
            except Exception as e:
                return f"Multimodal Request Failed: {str(e)}"