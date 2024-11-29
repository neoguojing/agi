import time
from typing import Any
import threading
from agi.llms.model_factory import ModelFactory
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
    create_chat_with_history,
    create_chat_with_rag
)
from agi.tasks.common import (
    create_image_gen_chain,
    create_text2image_chain,
    create_translate_chain,
    create_text2speech_chain,
    create_speech2text_chain
)
from agi.tasks.retriever import FilterType,SimAlgoType
from langchain.globals import set_debug
set_debug(True)

TASK_LLM = "llm"
TASK_LLM_WITH_HISTORY = "llm_with_history"
TASK_LLM_WITH_RAG = "llm_with_rag"

TASK_EMBEDDING = "embedding"
TASK_AGENT = "agent"
TASK_TRANSLATE = "translate"
TASK_IMAGE_GEN = "image_gen"
TASK_TTS = "tts"
TASK_SPEECH_TEXT = "speech2text"
TASK_RETRIEVER = "rag"
TASK_RETRIEVER = "embedding"
    
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
    @staticmethod
    def create_task(task_type,**kwargs) -> Runnable:
        if task_type not in TaskFactory._instances:
            with TaskFactory._lock:
                if task_type not in TaskFactory._instances:
                    try:
                        if task_type == TASK_LLM:
                            model_name = kwargs.get("model_name") or OLLAMA_DEFAULT_MODE
                            instance = ChatOpenAI(
                                model=model_name,
                                openai_api_key=OPENAI_API_KEY,
                                base_url=urljoin(OLLAMA_API_BASE_URL, "/v1/")
                            )
                            
                        elif task_type == TASK_EMBEDDING:
                            model_name = kwargs.get("model_name") or RAG_EMBEDDING_MODEL
                            instance = OllamaEmbeddings(
                                model=model_name,
                                base_url=OLLAMA_API_BASE_URL,
                            )
                        elif task_type == TASK_LLM_WITH_HISTORY:
                            return create_chat_with_history(TaskFactory._llm)
                        elif task_type == TASK_LLM_WITH_RAG:
                            return create_chat_with_rag(TaskFactory._llm,TaskFactory._embedding,kwargs)
                        elif task_type == TASK_TRANSLATE:
                            instance = create_translate_chain(TaskFactory._llm)
                        elif task_type == TASK_IMAGE_GEN:
                            instance = create_image_gen_chain(TaskFactory._llm)
                        elif task_type == TASK_TTS:
                            instance = create_text2speech_chain()
                        elif task_type == TASK_SPEECH_TEXT:
                            instance = create_speech2text_chain()
                        elif task_type == TASK_RETRIEVER:
                            from agi.tasks.retriever import create_retriever
                            instance = create_retriever(TaskFactory._llm,TaskFactory._embedding,kwargs=kwargs)
                
                        elif task_type == TASK_AGENT:
                            # instance = Agent()
                            pass

                        TaskFactory._instances[task_type] = instance
                    except Exception as e:
                        print(e)

        return TaskFactory._instances[task_type]



        