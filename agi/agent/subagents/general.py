
from agi.agent.middlewares.debug_middleware import DebugLLMContextMiddleware
from agi.agent.tools import RemoteImageEditTool,RemoteImageGenTool,RemoteMultiModalTool,RemoteTranscriptionTool,RemoteTTSTool,search_web
from agi.agent.middlewares import BrowserMiddleware,FfmpegMiddleware
from agi.agent.models import ModelProvider
from agi.agent.sandbox.docker import DockerSandbox
image_gen_tool = RemoteImageGenTool()
image_edit_tool = RemoteImageEditTool()
visual_subagent = {
    "name": "visual-artist",
    "description": "Used for creating new images or editing existing ones. Provide clear visual descriptions.",
    "system_prompt": (
        "You are an expert digital artist. Your goal is to generate or edit images. "
        "When generating, expand user prompts into high-quality descriptive prompts. "
        "Always return the resulting Image URL to the main agent."
    ),
    "tools": [image_gen_tool, image_edit_tool],
    "middleware": [
        DebugLLMContextMiddleware(name="visual_subagent")
    ]
}

omni_tool = RemoteMultiModalTool()
perception_subagent = {
    "name": "perception-expert",
    "description": "Used to transcribe audio, analyze images/videos, and generate speech responses.",
    "system_prompt": (
        "You are an expert in multi-modal understanding. You can hear audio via Whisper, "
        "see images/videos via Omni models, and speak via TTS. "
        "If a user provides a file path, use the appropriate tool to process it."
    ),
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
    "system_prompt": (
        "You are an expert Transcriptionist and Linguist. "
        "Your primary goal is to convert audio input into highly accurate text using the provided tools. "
        "Guidelines:\n"
        "1. Identify the primary language of the audio automatically unless specified.\n"
        "2. Clean up the transcript: remove filler words (um, ah, etc.) and fix obvious stutters to ensure the final text is professional and coherent.\n"
        "3. Maintain the original meaning and tone of the speaker.\n"
        "4. If the audio is low quality or the transcription fails, explain the specific reason to the main agent."
    ),
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
    "system_prompt": (
        "You are a professional Voice Director and Audio Engineer. "
        "Your goal is to synthesize natural-sounding speech from text. "
        "Guidelines:\n"
        "1. Select the most appropriate TTS model based on context: 'cosyvoice' for high-fidelity/emotional content, 'xtts' for professional/multilingual content.\n"
        "2. If real-time playback is required, use the streaming tool and provide the WebSocket connection details.\n"
        "3. Ensure the text is sanitized (e.g., expand abbreviations like 'Dr.' to 'Doctor') before synthesis to improve pronunciation.\n"
        "4. Return the final Audio URL or streaming instructions clearly to the main agent."
    ),
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
    "description": (
        "Specialized in web search and information retrieval. "
        "Can search the internet, summarize results, "
        "and provide structured answers from online sources. "
        "Operates on public web data and returns concise, relevant information."
    ),
    "system_prompt": "",
    "tools": [search_web],

    "middleware": [
        DebugLLMContextMiddleware(name="web_search_subagent")
    ]
}

ffmpeg_subagent = {
    "name": "video-expert",
    "description": (
        "Specialized in video processing and editing tasks. "
        "Can cut, merge, snapshot, transcode, remove watermarks, and apply filters on videos. "
        "Operates on local or sandboxed video files and reports success messages for each operation."
    ),
    "system_prompt": "",
    "middleware": [
        FfmpegMiddleware(
            backend=DockerSandbox()
        ),
        DebugLLMContextMiddleware(name="ffmpeg_subagent")
    ]
}

