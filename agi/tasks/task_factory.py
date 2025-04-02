import time
from typing import Any,Union
import threading
from agi.llms.model_factory import ModelFactory
from agi.tasks.agent import create_react_agent_task
from langchain_core.embeddings import Embeddings
from langchain_core.runnables import Runnable
from urllib.parse import urljoin
from agi.config import (
    OLLAMA_API_BASE_URL,
    OPENAI_API_KEY,
    RAG_EMBEDDING_MODEL,
    OLLAMA_DEFAULT_MODE,
    CACHE_DIR
)
from langchain_openai import ChatOpenAI
from langchain_ollama import OllamaEmbeddings
from agi.tasks.llm_app import (
    create_chat,
    create_llm_with_history,
)
from agi.tasks.multi_model_app import (
    create_image_gen_chain,
    create_text2image_chain,
    create_translate_chain,
    create_text2speech_chain,
    create_speech2text_chain,
    create_embedding_task,
    create_llm_task,
)
from agi.tasks.retriever import FilterType,SimAlgoType
from langchain.globals import set_debug
from agi.tasks.retriever import KnowledgeManager

# set_debug(True)

TASK_LLM = "llm"
TASK_LLM_WITH_HISTORY = "llm_with_history"
TASK_AGENT = "agent"
TASK_TRANSLATE = "translate"
TASK_IMAGE_GEN = "image_gen"
TASK_TTS = "tts"
TASK_SPEECH_TEXT = "speech2text"
TASK_RETRIEVER = "rag"
TASK_RAG = "custom_rag"
TASK_WEB_SEARCH = "web_search"
TASK_DOC_CHAT = "doc_chat"

def create_llm_chat_task(**kwargs):
    return create_chat(TaskFactory._llm)

def create_llm_with_history_task(**kwargs):
    return create_llm_with_history(TaskFactory._llm)

def create_rag_task(**kwargs):
    from agi.tasks.llm_app import create_rag
    return create_rag(TaskFactory._knowledge_manager)

def create_web_search_task(**kwargs):
    from agi.tasks.llm_app import create_websearch
    return create_websearch(TaskFactory._knowledge_manager)

def create_docchain_task(**kwargs):
    from agi.tasks.llm_app import create_stuff_documents_chain
    return create_stuff_documents_chain(TaskFactory._llm)

def create_translate_task(**kwargs):
    return create_translate_chain(TaskFactory._llm)

def create_image_gen_task(**kwargs):
    return create_image_gen_chain(TaskFactory._llm)

def create_tts_task(**kwargs):
    return create_text2speech_chain()

def create_speech_text_task(**kwargs):
    return create_speech2text_chain()


# agent with history
# TODO agent support history
def create_agent_task(**kwargs):
    # llm_with_history = create_llm_with_history(TaskFactory._llm)
    return create_react_agent_task(TaskFactory._llm)

class TaskFactory:
    _instances = {}
    _lock = threading.Lock()  # 异步锁
    _llm = ChatOpenAI(
            model=OLLAMA_DEFAULT_MODE,
            openai_api_key=OPENAI_API_KEY,
            base_url=urljoin(OLLAMA_API_BASE_URL, "/v1/")
        )
    _embedding = OllamaEmbeddings(
            model=RAG_EMBEDDING_MODEL,
            base_url=OLLAMA_API_BASE_URL,
        )
    
    task_creators = {
        TASK_LLM: create_llm_chat_task,
        TASK_LLM_WITH_HISTORY: create_llm_with_history_task,
        
        TASK_AGENT: create_agent_task,
        
        TASK_RAG: create_rag_task,
        TASK_WEB_SEARCH: create_web_search_task,
        TASK_DOC_CHAT: create_docchain_task,
        
        TASK_TRANSLATE: create_translate_task,
        TASK_IMAGE_GEN: create_image_gen_task,
        TASK_TTS: create_tts_task,
        TASK_SPEECH_TEXT: create_speech_text_task,
        
    }
    
    _knowledge_manager = KnowledgeManager(CACHE_DIR,_llm,_embedding)
    @staticmethod
    def get_knowledge_manager():
        return TaskFactory._knowledge_manager
    
    @staticmethod
    def get_embedding():
        return TaskFactory._embedding
    
    @staticmethod
    def get_llm():
        return TaskFactory._llm
    
    @staticmethod
    def create_task(task_type,**kwargs) -> Union[Runnable,Embeddings,KnowledgeManager]:
        if task_type not in TaskFactory._instances:
            with TaskFactory._lock:
                if task_type not in TaskFactory._instances:
                    try:
                        instance = TaskFactory.task_creators.get(task_type, lambda *args, **kwargs: None)(**kwargs)

                        if instance is None:
                            raise ValueError(f"Task type {task_type} not supported or invalid.")
                        
                        # Store the instance in the main instances dictionary
                        TaskFactory._instances[task_type] = instance

                    except Exception as e:
                        raise RuntimeError(f"Error creating task of type {task_type}: {e}")

        # Return the main instance
        return TaskFactory._instances.get(task_type)
