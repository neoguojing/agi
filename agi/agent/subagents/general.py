
from agi.agent.middlewares.debug_middleware import DebugLLMContextMiddleware
from agi.agent.tools import RemoteImageEditTool,RemoteImageGenTool,RemoteMultiModalTool,RemoteTranscriptionTool,RemoteTTSTool,search_web
from agi.agent.middlewares import BrowserMiddleware,FfmpegMiddleware
from agi.agent.models import ModelProvider
from agi.agent.sandbox.docker import DockerSandbox
from agi.agent.prompt import get_subagent_prompt
image_gen_tool = RemoteImageGenTool()
image_edit_tool = RemoteImageEditTool()
visual_subagent = {
    "name": "visual-artist",
    "description": "Used for creating new images or editing existing ones. Provide clear visual descriptions.",
    "system_prompt": get_subagent_prompt("visual-artist"),
    "tools": [image_gen_tool, image_edit_tool],
    "middleware": [
        DebugLLMContextMiddleware(name="visual_subagent")
    ]
}

omni_tool = RemoteMultiModalTool()
perception_subagent = {
    "name": "perception-expert",
    "description": "Used to transcribe audio, analyze images/videos, and generate speech responses.",
    "system_prompt": get_subagent_prompt("perception-expert"),
    "tools": [omni_tool],
    "middleware": [
        DebugLLMContextMiddleware(name="perception_subagent")
    ]
}


# Tools previously defined via HTTPX
stt_tool = RemoteTranscriptionTool()

stt_subagent = {
    "name": "stt-expert",
    "description": "Specialized in transcribing audio files (mp3, wav, m4a, etc.) into text. Use this when the user provides voice messages or recordings.",
    "system_prompt": get_subagent_prompt("stt-expert"),
    "tools": [stt_tool],
    "middleware": [
        DebugLLMContextMiddleware(name="stt_subagent")
    ]
}

# Tools previously defined via HTTPX
tts_tool = RemoteTTSTool()
tts_subagent = {
    "name": "tts-expert",
    "description": "Specialized in converting text into high-quality synthetic speech. Supports file generation and real-time audio streaming.",
    "system_prompt": get_subagent_prompt("tts-expert"),
    "tools": [tts_tool],
    "middleware": [
        DebugLLMContextMiddleware(name="tts_subagent")
    ]
}

browser_subagent = {
    "name": "browser-expert",
    "description": "Specialized in browsing websites, extracting content via OCR and DOM, and interacting with web pages.",
    "system_prompt": "",
    "middleware": [
        BrowserMiddleware(
            ocr_engine=ModelProvider.get_chat_model()
        ),
        DebugLLMContextMiddleware(name="browser_subagent")
    ]
}

web_search_subagent = {
    "name": "web-search-expert",
    "description": ("Specialized in web search and information retrieval. "),
    "system_prompt": get_subagent_prompt("web-search-expert"),
    "tools": [search_web],

    "middleware": [
        DebugLLMContextMiddleware(name="web_search_subagent")
    ]
}

ffmpeg_subagent = {
    "name": "video-expert",
    "description": "Specialized in video processing using FFmpeg.",
    "system_prompt": "",
    "middleware": [
        FfmpegMiddleware(
            backend=DockerSandbox()
        ),
        DebugLLMContextMiddleware(name="ffmpeg_subagent")
    ]
}

