from agi.llms.model_factory import ModelFactory
from agi.tasks.prompt import english_traslate_template,multimodal_input_template
from langchain_core.output_parsers import StrOutputParser,ListOutputParser
from agi.llms.text2image import Text2Image
from agi.llms.image2image import Image2Image
from agi.llms.tts import TextToSpeech
from agi.llms.speech2text import Speech2Text
from langchain_core.runnables import RunnablePassthrough,RunnableLambda,RunnableBranch
import json
from langchain_core.messages import HumanMessage,BaseMessage,AIMessage
from langchain_core.prompt_values import StringPromptValue,PromptValue,ChatPromptValue
from langgraph.prebuilt.chat_agent_executor import AgentState
from agi.llms.base import parse_input_messages
from agi.config import (
    OLLAMA_API_BASE_URL,
    OPENAI_API_KEY,
    RAG_EMBEDDING_MODEL,
    OLLAMA_DEFAULT_MODE,
)
from langchain_openai import ChatOpenAI
from langchain_ollama import OllamaEmbeddings,ChatOllama
from urllib.parse import urljoin
import logging
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

def graph_input_format(state: AgentState):
    return state["messages"]
    
def graph_parser(x):
    if isinstance(x,BaseMessage):
        return AgentState(messages=[x])
    elif isinstance(x,list[BaseMessage]):
        return AgentState(messages=x)
    
def build_messages(input :dict):
   
    media = None
    type = ""
    if input.get('type'):  # 首先获取type
        type = input['type']
        
    if input.get('data'):  # 获取媒体数据
        media =  input['data']
    
    if media is None:
        return HumanMessage(content=input.get("text"))

    return HumanMessage(content=[
        {"type": "text", "text": input.get("text")},
        {"type": type, type: media},
    ])
    

def parse_input(input: PromptValue) -> list[BaseMessage]:
    try:
        # 使用json模板输入
        if isinstance(input,StringPromptValue):
            log.debug(input.to_json())
            data = json.loads(input.to_string())
            return [build_messages(data)]
        # 使用message模板输入
        elif isinstance(input,ChatPromptValue):
            return input.to_messages()
    except json.JSONDecodeError as e:
        log.error(e,input.to_string())
        return {}


        
def create_translate_chain(llm,graph):
    chain = english_traslate_template | llm | StrOutputParser()
    if graph:
        def translate_node(state: AgentState):
            messages = state["messages"]
            if messages:
                if isinstance(messages, list):
                    messages = messages[-1]
                _,text = parse_input_messages(messages)
                result = chain.invoke({"text": text})
                if isinstance(messages.content,str):
                    return [HumanMessage(content=result)]
                elif isinstance(messages.content,list):
                    for content in messages.content:
                        media_type = content.get("type")
                        if media_type == "text":
                            content["text"] = result
                            return [messages]
        return RunnableLambda(translate_node)
        
    return chain


def create_text2image_chain(llm,graph=False):
    translate = create_translate_chain(llm,graph)
    # text2image = Text2Image()
    text2image = ModelFactory.get_model("text2image")
    
    return translate | text2image 

def create_image_gen_chain(llm,graph=False):
    translate = create_translate_chain(llm,graph)
    # image2image = Image2Image()
    image2image = ModelFactory.get_model("image2image")
    # text2image = Text2Image()
    text2image = ModelFactory.get_model("text2image")
    
    def is_image2image(x: list[BaseMessage]):
        for message in x:
            if isinstance(message.content,list):
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
        RunnablePassthrough.assign(text=translate.with_config(run_name="translate"))
        | multimodal_input_template
        | RunnableLambda(parse_input)
        | branch
    )
    
    if graph:
        chain = translate | branch | graph_parser
        
    return chain
# graph 模式
# Input：AgentState
# Output：AgentState
def create_text2speech_chain(graph=False):
    # text2speech = TextToSpeech()
    text2speech = ModelFactory.get_model("text2speech")
    chain = (multimodal_input_template
        | RunnableLambda(parse_input)
        | text2speech
    )
    
    if graph:
        chain = graph_input_format | text2speech | graph_parser
        
    return chain

# Input: AgentState
# Output: AgentState
def create_speech2text_chain(graph=False):
    # speech2text = Speech2Text()
    speech2text = ModelFactory.get_model("speech2text")
    chain = (multimodal_input_template
        | RunnableLambda(parse_input)
        | speech2text
    )
    
    if graph:
        def _convert_ai_huiman(x:AIMessage):
            if isinstance(x,list):
                x = x[-1]
            if isinstance(x,AIMessage):
                return HumanMessage(content=x.content)
            
        chain = graph_input_format | speech2text | _convert_ai_huiman | graph_parser
        
    return chain

def create_llm_task(**kwargs):
    model_name = kwargs.get("model_name") or OLLAMA_DEFAULT_MODE
    if kwargs.get("ollama"):
        return ChatOllama(
            model=model_name,
            base_url=OLLAMA_API_BASE_URL,
        )
    return ChatOpenAI(
        model=model_name,
        openai_api_key=OPENAI_API_KEY,
        base_url=urljoin(OLLAMA_API_BASE_URL, "/v1/")
    )
    
# Helper functions for each task type creation
def create_embedding_task(**kwargs):
    model_name = kwargs.get("model_name") or RAG_EMBEDDING_MODEL
    return OllamaEmbeddings(
        model=model_name,
        base_url=OLLAMA_API_BASE_URL,
    )
