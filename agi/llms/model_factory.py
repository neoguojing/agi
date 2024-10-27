import os
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

import threading
import gc
from langchain_community.chat_models import QianfanChatEndpoint
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# openai
os.environ['OPENAI_API_KEY'] = ''
# qianfan
os.environ["QIANFAN_AK"] = "your_ak"
os.environ["QIANFAN_SK"] = "your_sk"
# tongyi
os.environ["DASHSCOPE_API_KEY"] = ""

from urllib.parse import urljoin
class ModelFactory:
    _instances = {}
    _lock = threading.Lock()

    @staticmethod
    def get_model(model_type, model_name=""):
        """获取模型实例，并缓存"""
        if model_type not in ModelFactory._instances or ModelFactory._instances[model_type] is None:
            with ModelFactory._lock:
                if model_type not in ModelFactory._instances or ModelFactory._instances[model_type] is None:
                    ModelFactory._instances[model_type] = ModelFactory._load_model(model_type, model_name)
        return ModelFactory._instances[model_type]

    @staticmethod
    def _load_model(model_type, model_name=""):
        """实际负责模型加载的私有方法"""
        print(f"Loading the model {model_type}, wait a minute...")
        if model_type == "ollama":
            from langchain_openai import ChatOpenAI
            if model_name == "":
                model_name = OLLAMA_DEFAULT_MODE
            return ChatOpenAI(model=model_name,openai_api_key=OPENAI_API_KEY,base_url=urljoin(OLLAMA_API_BASE_URL,"/v1/"))
        elif model_type == "text2image": 
            return Text2Image()
        elif model_type == "image2image": 
            return Image2Image()
        elif model_type == "speech2text": 
            return Speech2Text()
        elif model_type == "text2speech": 
            return  TextToSpeech()
        elif model_type == "embedding": 
            from langchain_ollama import OllamaEmbeddings
            if model_name == "":
                model_name = RAG_EMBEDDING_MODEL
            return OllamaEmbeddings(
                model=model_name,
                base_url=OLLAMA_API_BASE_URL,
                # num_gpu=100
            )
        else:
            raise ValueError(f"Invalid model name: {model_type}")

    @staticmethod
    def destroy(model_name):
        """销毁模型实例"""
        if model_name in ModelFactory._instances and ModelFactory._instances[model_name] is not None:
            with ModelFactory._lock:
                instance = ModelFactory._instances.pop(model_name, None)
                if instance:
                    ModelFactory._safe_destroy(instance)

    @staticmethod
    def _safe_destroy(instance):
        """安全销毁模型实例"""
        refcount = len(gc.get_referrers(instance))
        if refcount <= 2:
            if isinstance(instance, CustomerLLM):
                instance.destroy()
            del instance
            gc.collect()

    @staticmethod
    def release():
        """释放所有模型实例"""
        for model_name in list(ModelFactory._instances.keys()):
            ModelFactory.destroy(model_name)