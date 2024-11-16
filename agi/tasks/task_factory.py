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

TASK_AGENT = "agent"
TASK_TRANSLATE = "translate"
TASK_IMAGE_GEN = "image_gen"
TASK_TTS = "tts"
TASK_SPEECH_TEXT = "speech2text"
TASK_GENERAL = "llm"
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
    _embeddin = OllamaEmbeddings(
            model=RAG_EMBEDDING_MODEL,
            base_url=OLLAMA_API_BASE_URL,
        )
    @staticmethod
    def create_task(task_type,**kwargs) -> Runnable:
        if task_type not in TaskFactory._instances:
            with TaskFactory._lock:
                if task_type not in TaskFactory._instances:
                    try:
                        if task_type == TASK_AGENT:
                            # instance = Agent()
                            pass
                        elif task_type == TASK_TRANSLATE:
                            from agi.tasks.common import create_translate_chain
                            instance = create_translate_chain(TaskFactory._llm)
                        elif task_type == TASK_IMAGE_GEN:
                            collection_names
                        elif task_type == TASK_GENERAL:
                            model_name = kwargs.get("model_name") or OLLAMA_DEFAULT_MODE
                            instance = ChatOpenAI(
                                model=model_name,
                                openai_api_key=OPENAI_API_KEY,
                                base_url=urljoin(OLLAMA_API_BASE_URL, "/v1/")
                            )
                        elif task_type == "embedding":
                            model_name = kwargs.get("model_name") or RAG_EMBEDDING_MODEL
                            instance = OllamaEmbeddings(
                                model=model_name,
                                base_url=OLLAMA_API_BASE_URL,
                            )
                        elif task_type == TASK_RETRIEVER:
                            from agi.tasks.retriever import KnowledgeManager
                            collection_names = kwargs.get("collection_names","all")
                            km = KnowledgeManager(CACHE_DIR,TaskFactory._llm,TaskFactory._embeddin)
                            instance = km.get_retriever(collection_names)

                        TaskFactory._instances[task_type] = instance
                    except Exception as e:
                        print(e)

        return TaskFactory._instances[task_type]



        