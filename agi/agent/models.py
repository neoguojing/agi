from typing import Union
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.embeddings import Embeddings

class ModelProvider:
    @staticmethod
    def get_chat_model(
        provider: str, 
        model_name: str, 
        temperature: float = 0.7,
        base_url: str = None
    ) -> BaseChatModel:
        """
        获取文本模型实例
        provider: "openai" 或 "ollama"
        """
        if provider.lower() == "openai":
            return ChatOpenAI(
                model=model_name,
                temperature=temperature,
                openai_api_base=base_url # 可选，用于代理或兼容接口
            )
        elif provider.lower() == "ollama":
            return ChatOllama(
                model=model_name,
                temperature=temperature,
                base_url=base_url or "http://localhost:11434"
            )
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    @staticmethod
    def get_embeddings(
        provider: str, 
        model_name: str,
        base_url: str = None
    ) -> Embeddings:
        """
        获取 Embedding 模型实例
        """
        if provider.lower() == "openai":
            return OpenAIEmbeddings(model=model_name, openai_api_base=base_url)
        elif provider.lower() == "ollama":
            return OllamaEmbeddings(model=model_name, base_url=base_url or "http://localhost:11434")
        else:
            raise ValueError(f"Unsupported provider: {provider}")