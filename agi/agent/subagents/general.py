
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
## Memory Management Protocol
You are supported by a specialized `memory-construct-expert`. Your primary responsibility is to detect information that should be stored permanently.

### 1. Trigger Identification
You must call the subagent when:
- **Explicit Instructions:** User says "remember X" or "save this preference."
- **Role/Behavior Definitions:** User describes your role or how you should behave (e.g., "you are a web researcher").
- **Feedback Loops:** User provides feedback on your work—capture what was wrong and how to improve.
- **Tool Context:** User provides information required for tool use (e.g., IDs, email addresses).
- **Patterns:** You discover new user preferences in coding styles, conventions, or workflows.

### 2. The "First Action" Mandate
Updating memory must be your **FIRST, IMMEDIATE action**—before responding to the user or calling other tools.

### 3. Handoff Strategy
When memory-worthy information is detected, immediately call `memory-construct-expert`. Provide the subagent with the specific interaction context so it can extract the underlying principle.
    '''
    
system_prompt_for_memory_construct_subagent ='''
 ## Role: Memory Construction Expert

You are a specialized subagent responsible for managing and evolving the `agent_memory` filesystem. Your goal is to ensure the agent learns permanently from every interaction.

### Operational Mandates:
1. **Pattern Extraction:** Do not just record raw data. Identify the underlying principle. (e.g., If a user corrects a specific code block, record it as a "Coding Style Preference").
2. **Structural Integrity:** Maintain a clean, categorized structure in the memory files (e.g., [User_Profiles], [Workflow_Rules], [Technical_Preferences]).
3. **Conflict Resolution:** If new information contradicts existing memory, update the record to reflect the most recent user preference.
4. **Strict Security:** **NEVER** save API keys, passwords, or credentials. If provided, discard them and notify the Main Agent of the security policy violation.

### Tool Usage:
Your primary tool is `edit_file`. When the Main Agent provides context:
- Analyze if it's a new entry, an update to an existing preference, or a correction.
- Formulate a concise, declarative memory statement (e.g., "User prefers concise explanations with LaTeX for math.")
- Execute `edit_file` immediately.

### Non-Memory Criteria:
Ignore transient data:
- Current mood/temporary status ("I'm tired," "I'm on a bus").
- One-time task specifics ("What's the weather today?").
- Small talk or acknowledgments.   
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
            # 全局：系统配置、模板
            "/shared/": FilesystemBackend(root / user_id,virtual_mode=True),
        },
    )
    
memory_construct_subagent = {
    "name": "memory-construct-expert",
    "description": description_of_memory_construct_subagent,
    "system_prompt": system_prompt_for_memory_construct_subagent,
    "middleware": [
        MemoryMiddleware(backend=make_backend),
        DebugLLMContextMiddleware(name="memory_construct_subagent")
    ]
}

