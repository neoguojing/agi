from typing import Union
import threading
import gc
from agi.llms.image2image import Image2Image
from agi.llms.text2image import Text2Image
from agi.llms.tts import TextToSpeech
from agi.llms.speech2text import Speech2Text
from agi.llms.multimodel import MultiModel
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

from agi.config import log

class ModelFactory:
    _instances =  OrderedDict()
    _lock = threading.Lock()
    max_models = 1
    
    @staticmethod
    def get_model(model_type: str) -> CustomerLLM:
        """Retrieve and cache the model instance."""
        with ModelFactory._lock:
            if model_type not in ModelFactory._instances or ModelFactory._instances[model_type] is None:
                ModelFactory._instances[model_type] = ModelFactory._load_model(model_type)
            #     # 超出容量：弹出最久未使用
            #     while len(ModelFactory._instances) > ModelFactory.max_models:
            #         old_key, old_inst = ModelFactory._instances.popitem(last=False)
            #         log.info(f"Unloading model {old_key}: {old_inst}")
            #         try:
            #             old_inst.destroy()
            #         except Exception as e:
            #             log.warning(f"Error destroying model {old_key}: {e}")
            # else:
            #     ModelFactory._instances.move_to_end(model_type)
                
            return ModelFactory._instances[model_type]

    @staticmethod
    def _load_model(model_type: str) -> Union[CustomerLLM,Runnable]:
        """Load the model based on the model type."""
        log.info(f"createt the: {model_type}...")
        model = None
        
        if model_type == "text2image":
            model = Text2Image()
        
        elif model_type == "image2image":
            model = Image2Image()
        
        elif model_type == "speech2text":
            model = Speech2Text()
        
        elif model_type == "text2speech":
            model = TextToSpeech()
        
        elif model_type == "multimodel":
            model = MultiModel()
                
        if not model:
            raise ValueError(f"Invalid model type: {model_type}")

        return model

