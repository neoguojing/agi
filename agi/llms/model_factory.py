import os
import threading
import gc
from urllib.parse import urljoin
from agi.llms.image2image import Image2Image
from agi.llms.text2image import Text2Image
from agi.llms.tts import TextToSpeech
from agi.llms.speech2text import Speech2Text
from agi.config import MODEL_PATH as model_root
from agi.llms.base import CustomerLLM
from agi.config import (
    OLLAMA_API_BASE_URL,
    OPENAI_API_KEY,
    RAG_EMBEDDING_MODEL,
    OLLAMA_DEFAULT_MODE
)
from langchain_openai import ChatOpenAI
from langchain_ollama import OllamaEmbeddings


class ModelFactory:
    _instances = {}
    _lock = threading.Lock()

    @staticmethod
    def get_model(model_type: str, model_name: str = "") -> CustomerLLM:
        """Retrieve and cache the model instance."""
        with ModelFactory._lock:
            if model_type not in ModelFactory._instances or ModelFactory._instances[model_type] is None:
                ModelFactory._instances[model_type] = ModelFactory._load_model(model_type, model_name)
            return ModelFactory._instances[model_type]

    @staticmethod
    def _load_model(model_type: str, model_name: str = "") -> CustomerLLM:
        """Load the model based on the model type."""
        print(f"Loading the model: {model_type}...")
        model = None
        
        if model_type == "ollama":
            model_name = model_name or OLLAMA_DEFAULT_MODE
            model = ChatOpenAI(
                model=model_name,
                openai_api_key=OPENAI_API_KEY,
                base_url=urljoin(OLLAMA_API_BASE_URL, "/v1/")
            )
        
        elif model_type == "text2image":
            model = Text2Image()
        
        elif model_type == "image2image":
            model = Image2Image()
        
        elif model_type == "speech2text":
            model = Speech2Text()
        
        elif model_type == "text2speech":
            model = TextToSpeech()
        
        elif model_type == "embedding":
            model_name = model_name or RAG_EMBEDDING_MODEL
            model = OllamaEmbeddings(
                model=model_name,
                base_url=OLLAMA_API_BASE_URL,
            )
        
        if not model:
            raise ValueError(f"Invalid model type: {model_type}")

        return model

    @staticmethod
    def destroy(model_name: str) -> None:
        """Destroy a specific model instance."""
        with ModelFactory._lock:
            instance = ModelFactory._instances.pop(model_name, None)
            if instance:
                ModelFactory._safe_destroy(instance)

    @staticmethod
    def _safe_destroy(instance: CustomerLLM) -> None:
        """Safely destroy the model instance if it is no longer referenced."""
        refcount = len(gc.get_referrers(instance))
        if refcount <= 2:  # Safe to delete when there are no external references
            if isinstance(instance, CustomerLLM):
                instance.destroy()
            del instance
            gc.collect()

    @staticmethod
    def release() -> None:
        """Release and destroy all cached model instances."""
        with ModelFactory._lock:
            for model_name in list(ModelFactory._instances.keys()):
                ModelFactory.destroy(model_name)
