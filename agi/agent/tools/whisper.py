import httpx
import os
from typing import Type, Optional, Any
from pydantic import BaseModel, Field
from langchain.tools import BaseTool
from agi.config import WHISPER_BASE_URL,API_KEY

class TranscriptionInput(BaseModel):
    audio_path: str = Field(description="The local file path of the audio to be transcribed.")
    model: str = Field(default="whisper-1", description="Model name, use 'base' for CPU inference or 'whisper-1' for default.")
    language: Optional[str] = Field(default=None, description="Optional: The language of the input audio (ISO-639-1 code).")


class RemoteTranscriptionTool(BaseTool):
    name: str = "remote_speech_to_text"
    description: str = (
        "Useful for converting audio files into text transcripts. "
        "Accepts a local audio file path and returns the transcribed text."
    )
    args_schema: Type[BaseModel] = TranscriptionInput
    
    api_base_url: str = Field(WHISPER_BASE_URL, exclude=True)
    api_key: str = Field(API_KEY, exclude=True)
    timeout: float = 180.0 # 转录长音频可能较慢

    def _run(self, audio_path: str, model: str = "whisper-1", language: Optional[str] = None) -> str:
        """同步执行：上传音频文件并获取转录文本"""
        
        # 1. 校验本地文件
        if not os.path.exists(audio_path):
            return f"Error: Audio file not found at {audio_path}"

        url = f"{self.api_base_url}/v1/audio/transcriptions"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        
        # 2. 准备 Form Data 字段
        data = {
            "model": model,
            "response_format": "json"
        }
        if language:
            data["language"] = language

        # 3. 使用 httpx 发送 multipart/form-data
        with httpx.Client(timeout=self.timeout) as client:
            try:
                with open(audio_path, "rb") as f:
                    files = {"file": (os.path.basename(audio_path), f, "audio/mpeg")}
                    
                    response = client.post(
                        url, 
                        data=data, 
                        files=files, 
                        headers=headers
                    )
                
                response.raise_for_status()
                result = response.json()
                
                # 返回转录出的文本
                return result.get("text", "Error: No text field in response.")
                
            except httpx.HTTPStatusError as e:
                return f"Transcription API Error ({e.response.status_code}): {e.response.text}"
            except Exception as e:
                return f"Transcription Failed: {str(e)}"