import httpx
from typing import Type, Optional, Any
from pydantic import BaseModel, Field
from langchain.tools import BaseTool
from agi.config import TTS_BASE_URL, API_KEY
class TTSToolInput(BaseModel):
    text: str = Field(description="The text content to be converted into speech.")
    model: str = Field(default="cosyvoice", description="TTS model choice: 'cosyvoice', 'xtts', or 'vibevoice'.")
    voice: str = Field(default="default", description="The voice/speaker ID to use.")
    response_format: str = Field(default="wav", description="Audio format: 'wav', 'mp3', etc.")


class RemoteTTSTool(BaseTool):
    name: str = "remote_text_to_speech"
    description: str = (
        "Useful for converting text into an audio file (WAV/MP3). "
        "Returns a URL to the generated audio file."
    )
    args_schema: Type[BaseModel] = TTSToolInput
    
    api_base_url: str = Field(TTS_BASE_URL, exclude=True)
    api_key: str = Field(API_KEY, exclude=True)
    timeout: float = 120.0

    def _run(self, text: str, model: str = "cosyvoice", voice: str = "default", response_format: str = "wav") -> str:
        """调用 /v1/audio/speech 接口获取音频文件"""
        url = f"{self.api_base_url}/v1/audio/speech"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        
        # 对应后端 SpeechRequest 结构
        payload = {
            "input": text,
            "model": model,
            "voice": voice,
            "response_format": response_format,
            "stream": False,
            "user": voice # 映射 user_id
        }
        
        with httpx.Client(timeout=self.timeout) as client:
            try:
                # 注意：后端逻辑返回的是 FileResponse (文件二进制) 或包含路径的 JSON
                # 根据你后端代码的返回，我们通常期望得到 URL。
                # 如果后端直接返回文件，这里需要处理 response.headers 中的文件名或预览地址
                response = client.post(url, json=payload, headers=headers)
                response.raise_for_status()
                
                # 如果后端返回的是 FileResponse，LangChain 工具通常需要的是一个可访问的地址
                # 在 AGI 架构中，通常后端会记录文件并返回 URL，或者由 API 网关处理
                if response.headers.get("content-type", "").startswith("audio/"):
                    # 假设后端在 header 或响应中提供了文件标识
                    return f"Audio generated successfully. Access URL: {response.url}"
                
                return f"TTS Success: {response.text}"
            except Exception as e:
                return f"TTS Error: {str(e)}"

class RemoteTTSStreamTool(BaseTool):
    name: str = "remote_tts_streaming_info"
    description: str = (
        "Initializes a streaming TTS task. Useful when the user wants real-time audio playback via WebSocket."
    )
    args_schema: Type[BaseModel] = TTSToolInput
    
    api_base_url: str = Field(..., exclude=True)
    api_key: str = Field(..., exclude=True)

    def _run(self, text: str, model: str = "cosyvoice", voice: str = "default", **kwargs) -> str:
        """调用 /v1/audio/speech/streaming 触发后端生成并返回 WebSocket 信息"""
        url = f"{self.api_base_url}/v1/audio/speech/streaming"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        
        payload = {
            "input": text,
            "model": model,
            "user": voice,
            "stream": True
        }
        
        # 转换 http 为 ws 协议地址
        ws_base = self.api_base_url.replace("http", "ws")
        ws_url = f"{ws_base}/v1/ws/audio_stream/{voice}"
        
        with httpx.Client(timeout=10.0) as client:
            try:
                # 触发流式任务
                response = client.post(url, json=payload, headers=headers)
                response.raise_for_status()
                
                return (
                    f"Streaming task initialized. The audio stream is being generated. "
                    f"Client should connect to WebSocket: {ws_url}"
                )
            except Exception as e:
                return f"Streaming Initialization Error: {str(e)}"