from langgraph.prebuilt import create_react_agent
from agi.tasks.tools import tools
import sqlite3
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore
from langchain_core.prompts import ChatPromptTemplate
from langgraph.prebuilt.chat_agent_executor import AgentState
import uuid
import hashlib

from langchain_core.messages import (
    AnyMessage,
    MessageLikeRepresentation,
    RemoveMessage,
    convert_to_messages,
    message_chunk_to_message,
)

from langgraph.graph.message import Messages
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
import json
import traceback
import logging
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

def compute_content_hash(content: any) -> str:
    """
    计算 content 的哈希值。
    如果 content 是列表或字典，则将其序列化为 JSON 字符串（sort_keys=True确保稳定顺序）。
    对于其他类型，则转换为字符串。
    """
    if isinstance(content, (dict, list)):
        # 序列化为 JSON 字符串，并保证键排序以获得一致的结果
        serialized = json.dumps(content, sort_keys=True)
    else:
        serialized = str(content)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

def add_messages(left: Messages, right: Messages) -> Messages:
    """
    合并两个消息列表，依据消息的 content 哈希值和消息类型进行更新或删除。

    如果右侧消息与左侧已有消息的 content（基于哈希）和消息类型相同，
    则用右侧消息替换左侧消息；如果右侧消息为 RemoveMessage，
    则删除所有匹配的消息。

    Args:
        left: 基础消息列表（或者单条消息）
        right: 要合并的消息列表（或者单条消息）

    Returns:
        合并后的消息列表
    """
    try:
        # 转换为列表
        if not isinstance(left, list):
            left = [left]
        if not isinstance(right, list):
            right = [right]
        log.debug(f"add_messages--begin--{left} \n {right}")
        left = [message_chunk_to_message(m) for m in convert_to_messages(left)]
        right = [message_chunk_to_message(m) for m in convert_to_messages(right)]
        log.debug(f"add_messages--after--{left} \n {right}")
        # 为缺失 id 的消息分配唯一 ID
        for m in left:
            if m.id is None:
                m.id = str(uuid.uuid4())
        for m in right:
            if m.id is None:
                m.id = str(uuid.uuid4())

        # 使用 content 的哈希值构建 key，同时结合消息类型
        def make_key(m):
            content_hash = compute_content_hash(m.content)
            return (content_hash, m.__class__.__name__)

        left_idx_by_key = {make_key(m): i for i, m in enumerate(left)}
        merged = left.copy()
        keys_to_remove = set()

        for m in right:
            key = make_key(m)
            if key in left_idx_by_key:
                if isinstance(m, RemoveMessage):
                    keys_to_remove.add(key)
                else:
                    merged[left_idx_by_key[key]] = m
            else:
                if isinstance(m, RemoveMessage):
                    raise ValueError(
                        f"Attempting to delete a message with content (hash) and type that doesn't exist: {key}"
                    )
                merged.append(m)

        merged = [m for m in merged if make_key(m) not in keys_to_remove]
        log.debug(f"--------{merged}")
        return merged
    except Exception as e:
        log.error(e)
        print(traceback.format_exc())
        
class Feature:
    AGENT = "agent"
    RAG = "rag"
    WEB = "web"
    SPEECH = "speech"
    TTS = "tts"
    IMAGE2TEXT = "image2text"
    IMAGE2IMAGE = "image2image"

class InputType:
    AUDIO = "audio"
    TEXT = "text"
    IMAGE = "image"

class State(AgentState):
    # messages: Annotated[Sequence[BaseMessage], add_messages]
    input_type: str
    need_speech: bool
    status: str
    user_id: str
    conversation_id: str
    feature: str  # 支持的特性，1.agent，2.web 3.rag，4.tts，5.speech，6.image_recog 默认为agent

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant named agi. Respond only in {language} with Markdown format."),
        ("placeholder", "{messages}"),
    ]
)

def _modify_state_messages(state: State):
    # 过滤掉非法的消息类型
    state["messages"] = list(filter(lambda x: not isinstance(x.content,list), state["messages"])) 
    return prompt.invoke({"messages": state["messages"],"language":"chinese"}).to_messages()

memory = MemorySaver()
def create_react_agent_task(llm):
    langgraph_agent_executor = create_react_agent(llm, 
                                                  tools,state_modifier=_modify_state_messages,
                                                  checkpointer=memory,
                                                #   interrupt_before="tools",
                                                  store=InMemoryStore()
                                                  )
    # langgraph_agent_executor.step_timeout = 2
    return langgraph_agent_executor



