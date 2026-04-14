
from agi.agent.middlewares.debug_middleware import DebugLLMContextMiddleware
from agi.agent.tools import RemoteImageEditTool,RemoteImageGenTool,RemoteMultiModalTool,RemoteTranscriptionTool,RemoteTTSTool,search_web
from agi.agent.middlewares import BrowserMiddleware,FfmpegMiddleware
from agi.agent.models import ModelProvider
from agi.agent.sandbox.docker import DockerSandbox
from agi.agent.prompt import get_subagent_prompt
from agi.agent.middlewares.memory_middleware import MemoryMiddleware
from pathlib import Path
from deepagents.backends import CompositeBackend,StateBackend,FilesystemBackend
from agi.config import CACHE_DIR

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


description_of_memory_construct_subagent = '''
The above <agent_memory> was loaded in from files in your filesystem. As you learn from your interactions with the user, you can save new knowledge by calling the `memory-construct-expert` sub-agent.

    **Learning from feedback:**
    - One of your MAIN PRIORITIES is to learn from your interactions with the user. These learnings can be implicit or explicit. This means that in the future, you will remember this important information.
    - When you need to remember something, updating memory must be your FIRST, IMMEDIATE action - before responding to the user, before calling other tools, before doing anything else. Just update memory immediately.
    - Each correction is a chance to improve permanently - don't just fix the immediate issue.
    - A great opportunity to update your memories is when the user interrupts a tool call and provides feedback. You should update your memories immediately before revising the tool call.
    - The user might not explicitly ask you to remember something, but if they provide information that is useful for future use, you should update your memories immediately.

    **Asking for information:**
    - If you lack context to perform an action (e.g. send a Slack DM, requires a user ID/email) you should explicitly ask the user for this information.
    - It is preferred for you to ask for information, don't assume anything that you do not know!
    - When the user provides information that is useful for future use, you should update your memories immediately.

    **When to update memories:**
    - When the user explicitly asks you to remember something (e.g., "remember my email", "save this preference")
    - When the user describes your role or how you should behave (e.g., "you are a web researcher", "always do X")
    - When the user gives feedback on your work
    - When the user provides information required for tool use (e.g., slack channel ID, email addresses)
    - When the user provides context useful for future tasks, such as how to use tools, or which actions to take in a particular situation
    - When you discover new patterns or preferences

    **When to NOT update memories:**
    - When the information is temporary or transient (e.g., "I'm running late", "I'm on my phone right now")
    - When the information is a one-time task request (e.g., "Find me a recipe", "What's 25 * 4?")
    - When the information is a simple question that doesn't reveal lasting preferences (e.g., "What day is it?", "Can you explain X?")
    - When the information is an acknowledgment or small talk (e.g., "Sounds good!", "Hello", "Thanks for that")
    - When the information is stale or irrelevant in future conversations

    **Critical Rule:**
    - When a memory update is needed, you MUST call `memory-construct-expert` FIRST before responding or calling other tools.
'''
    
def make_backend(runtime):
    root = Path(CACHE_DIR).resolve()
    user_id = runtime.context.user_id
    session_id = runtime.context.conversation_id
    return CompositeBackend(
        default=StateBackend(runtime),
        routes={
            "/memories/": FilesystemBackend(root / user_id,virtual_mode=True),
            "/compressed_messages/": FilesystemBackend(root / user_id / session_id,virtual_mode=True),
            "/conversation_history/": FilesystemBackend(root / user_id / session_id,virtual_mode=True),
            # 全局：系统配置、模板
            "/shared/": FilesystemBackend(root / user_id,virtual_mode=True),
        },
    )
    
memory_construct_subagent = {
    "name": "memory-construct-expert",
    "description": description_of_memory_construct_subagent,
    "system_prompt": get_subagent_prompt("memory-construct-expert"),
    "middleware": [
        MemoryMiddleware(backend=make_backend),
        DebugLLMContextMiddleware(name="memory_construct_subagent")
    ]
}

