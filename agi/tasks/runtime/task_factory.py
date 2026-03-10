from typing import Union
import threading
from agi.tasks.agent import create_react_agent_task
from langchain_core.embeddings import Embeddings
from langchain_core.runnables import Runnable
from agi.config import (
    OLLAMA_API_BASE_URL,
    RAG_EMBEDDING_MODEL,
    CACHE_DIR,
    EMBEDDING_BASE_URL,
    OLLAMA_THINKING_MODE
)
from langchain_ollama import OllamaEmbeddings,ChatOllama

from agi.tasks.rag.knowledge import KnowledgeManager
from agi.tasks.utils import refine_last_message_runnable

TASK_LLM = "llm"
TASK_AGENT = "agent"


def create_deepagent_task(**kwargs):
    return create_react_agent_task(TaskFactory._llm)

class TaskFactory:
    _instances = {}
    _lock = threading.Lock()  # 异步锁
    _llm = ChatOllama(
            model=OLLAMA_THINKING_MODE,
            base_url=OLLAMA_API_BASE_URL,
            num_ctx=4096,
            # num_gpu=1,
            temperature=0.1,
            top_k=10,
            top_p=0.5,
            repeat_penalty=1.5,
            repeat_last_n=-1,
            seed=42
        )
    ollama_embedding = OllamaEmbeddings(
            model="bge-m3:latest",
            base_url=OLLAMA_API_BASE_URL,
        )
    qwen_embedding = OllamaEmbeddings(
            model="qwen",
            base_url=EMBEDDING_BASE_URL
        )

    
    _knowledge_manager = KnowledgeManager(CACHE_DIR,_llm,ollama_embedding)
    @staticmethod
    def get_knowledge_manager():
        return TaskFactory._knowledge_manager
    
    @staticmethod
    def get_embedding(model=RAG_EMBEDDING_MODEL):
        if model == "qwen":
            return TaskFactory.qwen_embedding
        return TaskFactory.ollama_embedding
    
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
