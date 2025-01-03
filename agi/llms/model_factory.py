from typing import Union
import threading
import gc
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
from collections import OrderedDict
from langchain_core.runnables import Runnable

class ModelFactory:
    _instances =  OrderedDict()
    _lock = threading.Lock()
    max_models = 2
    
    @staticmethod
    def get_model(model_type: str, model_name: str = "") -> CustomerLLM:
        """Retrieve and cache the model instance."""
        with ModelFactory._lock:
            if model_type not in ModelFactory._instances or ModelFactory._instances[model_type] is None:
                ModelFactory._instances[model_type] = ModelFactory._load_model(model_type, model_name)
                if len(ModelFactory._instances) > ModelFactory.max_models:
                    # 如果超出了最大运行模型数，移除最久未使用的模型
                    removed_model = ModelFactory._instances.popitem(last=False)
                    if isinstance(removed_model[1],CustomerLLM):
                        removed_model[1].destroy()
            else:
                ModelFactory._instances.move_to_end(model_type)
                
            return ModelFactory._instances[model_type]

    @staticmethod
    def _load_model(model_type: str, model_name: str = "") -> Union[CustomerLLM,Runnable]:
        """Load the model based on the model type."""
        print(f"Loading the model: {model_type}...")
        model = None
        
        if model_type == "text2image":
            model = Text2Image()
        
        elif model_type == "image2image":
            model = Image2Image()
        
        elif model_type == "speech2text":
            model = Speech2Text()
        
        elif model_type == "text2speech":
            model = TextToSpeech()
                
        if not model:
            raise ValueError(f"Invalid model type: {model_type}")

        return model

    @staticmethod
    def _safe_destroy(instance: CustomerLLM) -> None:
        """Safely destroy the model instance if it is no longer referenced."""
        refcount = len(gc.get_referrers(instance))
        if refcount <= 2:  # Safe to delete when there are no external references
            if isinstance(instance, CustomerLLM):
                instance.destroy()
            del instance
            gc.collect()
