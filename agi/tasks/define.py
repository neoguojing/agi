from langchain_core.documents import Document
from typing import (
    Annotated,
    Callable,
    Optional,
    Sequence,
    Type,
    TypedDict,
    TypeVar,
    Union,
)
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    SystemMessage,
)
from langgraph.graph.message import add_messages
from langgraph.managed import IsLastStep

class Feature:
    AGENT = "agent"
    RAG = "rag"
    WEB = "web"
    LLM = "llm"
    HUMAN = "human"
    TTS = "tts"

    SPEECH = "speech"
    VOICECHAT = "voice_chat"

    IMAGE2TEXT = "image2text"
    IMAGE2IMAGE = "image2image"

    VIDEOPARSE = "videoparse"

    # cv模型，适用于图片和视频
    DETECT = "detect"
    FEATUREEXTRACT = "feature"
    CLASS = "class"
    SEGMENT = ""



class InputType:
    AUDIO = "audio"
    TEXT = "text"
    IMAGE = "image"
    VIDEO = "video"

class AgentState(TypedDict):
    """The state of the agent."""

    messages: Annotated[Sequence[BaseMessage], add_messages]

    is_last_step: IsLastStep

class State(AgentState):
    # messages: Annotated[Sequence[BaseMessage], add_messages]
    input_type: str
    need_speech: bool
    user_id: str
    conversation_id: str
    feature: str  # 支持的特性，1.agent，2.web 3.rag，4.tts，5.speech，6.image_recog 默认为agent
    
    context: str
    docs: list[Document]
    citations: list[any]
    collection_names: list[str]

    auto_decide_reuslt: str
    status: str
    step: list[str] #用于保存执行的步骤,按顺序排列
    
    user_feedback: str