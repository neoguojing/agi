from langchain_core.documents import Document
from typing import (
    Annotated,
    Callable,
    Optional,
    Sequence,
    Type,
    # TypedDict,  #for 3.12 and above
    TypeVar,
    Union,
)

from typing_extensions import TypedDict
from pydantic import BaseModel,Field
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    SystemMessage,
)
from langgraph.graph.message import add_messages
from langgraph.managed import IsLastStep, RemainingSteps

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

    remaining_steps: RemainingSteps

class State(AgentState):
    # messages: Annotated[Sequence[BaseMessage], add_messages]
    input_type: str
    need_speech: bool
    user_id: str
    conversation_id: str
    feature: str  # 支持的特性，1.agent，2.web 3.rag，4.tts，5.speech，6.image_recog 默认为agent
    # for rag search
    context: str
    urls: list[str]
    docs: list[Document]
    citations: list[any]
    collection_names: list[str]
    # for doc store
    file_path: str
    db_documents: list[Document]
    filted_texts: list[str]
    embds: list[list[float]]
    clusters: list[Document]
    collection_name: str



    auto_decide_result: str
    status: str
    step: list[str] #用于保存执行的步骤,按顺序排列
    
    user_feedback: str


# 一个tool的定义,和模型绑定,让模型决定是否调用该工具
# 不会实际实现该函数
class AskHuman(BaseModel):
    """
    This model is used when an automated system or agent determines that 
    human intervention is required. It represents a question that the agent 
    wants to ask the human to proceed with a task, resolve ambiguity, or 
    make a decision that requires human judgment.

    Typical use cases include:
    - Uncertain model predictions or low-confidence outcomes
    - Missing information that requires user input
    - Safety-critical or policy-sensitive decisions
    """

    question: str = Field(
        description="A question the system wants to ask the human for clarification, confirmation, or additional input"
    )