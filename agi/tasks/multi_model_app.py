from agi.llms.model_factory import ModelFactory
from agi.tasks.prompt import default_modify_state_messages_runnable,multimodal_input_template,traslate_modify_state_messages_runnable
from langchain_core.output_parsers import StrOutputParser,ListOutputParser
from langchain_core.runnables import RunnablePassthrough,RunnableLambda,RunnableBranch
from pydantic import BaseModel, Field
from typing import Literal, Union,List
from langchain_core.messages import HumanMessage,BaseMessage,AIMessage,ToolMessage
from langchain_core.prompt_values import StringPromptValue,PromptValue,ChatPromptValue
from langgraph.prebuilt.chat_agent_executor import AgentState
from agi.tasks.utils import graph_response_format_runnable,refine_last_message_text
from agi.config import (
    OLLAMA_API_BASE_URL,
    OPENAI_API_KEY,
    RAG_EMBEDDING_MODEL,
    OLLAMA_DEFAULT_MODE,
)
from langchain_openai import ChatOpenAI
from langchain_ollama import OllamaEmbeddings,ChatOllama
from urllib.parse import urljoin
from agi.tasks.utils import split_think_content
from agi.config import log


# Input: AgentState
# Output: AgentState
# 翻译后的内容替换最后一条消息
def create_translate_chain(llm):
    # 输入：AgentState
    # 输出：字符串
    chain = traslate_modify_state_messages_runnable | llm | StrOutputParser()
    # 修改AgentState的最后一条消息的内容为翻译后的内容
    def translate_node(state: AgentState):
        result = chain.invoke(state)
        # think 标签过滤
        _, result = split_think_content(result)
        if result:
            last_message = state["messages"][-1]
            if last_message:
                if isinstance(last_message.content,str):
                    last_message.content = result
                elif isinstance(last_message.content,list):
                    for content in last_message.content:
                        media_type = content.get("type")
                        if media_type == "text":
                            content["text"] = result
        return state["messages"]
    translate_runnable = RunnableLambda(translate_node)
    
    return RunnablePassthrough.assign(messages=translate_runnable).with_config(run_name="translate_chain")

# 仅提取消息列表,将AgentState -> List[BaseMessages]
def multimodel_state_modifier(state: AgentState):
    return state["messages"]

multimodel_state_modifier_runnable = RunnableLambda(multimodel_state_modifier)
# Input: AgentState
# Output: AgentState
def create_text2image_chain(llm):
    translate = create_translate_chain(llm)
    text2image = ModelFactory.get_model("text2image")
    
    return translate| multimodel_state_modifier_runnable | text2image | graph_response_format_runnable

# 或者基于agent执行图像生成？
def user_understand(llm):
    class TextItem(BaseModel):
        type: Literal["text"] = Field(description="The type of the item, must be 'text'")
        text: str = Field(
            description="In English. Based on the user's conversation history and current question, "
                        "generate a new user request that logically follows from the dialogue. "
                        "The request should reflect the user's needs, preferences, and intent, "
                        "while keeping the flow of the conversation natural and coherent."
        )


    class ImageItem(BaseModel):
        type: Literal["image"] = Field(description="The type of the item, must be 'image'")
        image: str = Field(
            description="If the user has previously provided an image and wants to modify or interact with it, "
                        "retrieve the most relevant image from the user's history. Include the image only if it "
                        "directly relates to the current request for modification or customization."
        )
    class Schema(BaseModel):
        """A list of schema items, each of which can be either a text or image type."""
        __root__: List[Union[TextItem, ImageItem]]

    model_with_structure = llm.with_structured_output(Schema)
    chain = default_modify_state_messages_runnable | model_with_structure | graph_response_format_runnable
    return chain

# Input: AgentState
# Output: AgentState
# TODO 依据历史消息，分析用户的意图
def create_image_gen_chain(llm):
    translate = create_translate_chain(llm)
    image2image = ModelFactory.get_model("image2image")
    text2image = ModelFactory.get_model("text2image")
    
    def is_image2image(x: list[BaseMessage]):
        message = x[-1]
        if isinstance(message,HumanMessage) and isinstance(message.content,list):
            for content in message.content:
                image = content.get("image")
                if image is not None and image != "":
                    return True
        return False
    
    branch = RunnableBranch(
            (
                (lambda x: not is_image2image(x),text2image)
            ),
            image2image
    )
    
    chain = (
        translate
        | multimodel_state_modifier_runnable
        | branch
    )
        
    return chain | graph_response_format_runnable

# Input：AgentState
# Output：AgentState
def create_text2speech_chain():
    text2speech = ModelFactory.get_model("text2speech")
    chain = multimodel_state_modifier_runnable | text2speech
          
    return chain | graph_response_format_runnable

# Input: AgentState
# Output: AgentState
def create_speech2text_chain():
    speech2text = ModelFactory.get_model("speech2text")
    chain = multimodel_state_modifier_runnable | speech2text | graph_response_format_runnable
    
    # 修改content,将audio直接转换为输入
    def state_modifier(x:AgentState):
        ai = chain.invoke(x)
        if isinstance(x["messages"][-1].content,list):
            x["messages"][-1].content = ai["messages"][-1].content
        return x["messages"]
        
    # 仅做语音到文本转换时，返回AIMessage，否则需要修改最后一条HumanMessage，增加text值
    branch = RunnableBranch(
        (
            (lambda x: x.get("feature") == "speech",chain)
        ),
        RunnablePassthrough.assign(messages=state_modifier).with_config(run_name="state_modifier")
    )
        
    return branch 


def create_llm_task(**kwargs):
    llm = None
    model_name = kwargs.get("model_name") or OLLAMA_DEFAULT_MODE
    if kwargs.get("ollama"):
        llm = ChatOllama(
            model=model_name,
            base_url=OLLAMA_API_BASE_URL,
        )
    else:
        llm = ChatOpenAI(
            model=model_name,
            openai_api_key=OPENAI_API_KEY,
            base_url=urljoin(OLLAMA_API_BASE_URL, "/v1/")
        )
        
    return llm
    
# Helper functions for each task type creation
def create_embedding_task(**kwargs):
    model_name = kwargs.get("model_name") or RAG_EMBEDDING_MODEL
    return OllamaEmbeddings(
        model=model_name,
        base_url=OLLAMA_API_BASE_URL,
    )

def create_multimodel_chain():
    multimodel = ModelFactory.get_model("multimodel")
    chain = multimodel_state_modifier_runnable | multimodel | graph_response_format_runnable
    
    return chain 