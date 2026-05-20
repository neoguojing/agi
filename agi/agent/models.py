import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

# LangChain 核心抽象
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.embeddings import Embeddings

# 厂商具体实现（按需安装：pip install langchain-openai langchain-ollama langchain-google-genai）
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

from agi.config import OLLAMA_API_BASE_URL,GOOGLE_API_KEY,GOOGLE_CLOUD_PROJECT,OPENROUTER_API_KEY

# 初始化日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("ModelRouter")


@dataclass
class ModelNode:
    """模型节点配置：支持手动指定优先级"""
    provider: str          # 供应商标识，如: 'openai', 'ollama', 'google', 'openrouter'
    model_name: str        # 模型技术名称，如: 'gpt-4o', 'llama3', 'gemini-3.1-pro-preview'
    priority: int          # 手动优先级：数值越小越优先（例如：0为主线，1为备用，2为终极兜底）
    api_key: str
    extra_params: Dict[str, Any] = field(default_factory=dict) # 针对不同厂商的个性化参数


class FallbackEmbeddings(Embeddings):
    """
    自定义高可用 Embedding 包装器。
    由于 LangChain 原生 Embeddings 未提供 .with_fallbacks 接口，
    此处采用装饰器模式实现：当高优先级向量模型报错时，自动无缝切换至下一顺位。
    """
    def __init__(self, embeddings_list: List[Embeddings]):
        self.embeddings_list = embeddings_list

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        for idx, emb in enumerate(self.embeddings_list):
            try:
                return emb.embed_documents(texts)
            except Exception as e:
                logger.warning(f"Embedding 节点 [{idx}] 调用失败: {e}. 正在尝试下一优先级的模型...")
        raise RuntimeError("所有配置的 Embedding 节点均已失效/额度耗尽。")

    def embed_query(self, text: str) -> List[float]:
        for idx, emb in enumerate(self.embeddings_list):
            try:
                return emb.embed_query(text)
            except Exception as e:
                logger.warning(f"Embedding 节点 [{idx}] 调用失败: {e}. 正在尝试下一优先级的模型...")
        raise RuntimeError("所有配置的 Embedding 节点均已失效/额度耗尽。")


class DynamicModelRouter:
    """动态模型路由器：支持手动优先级排序与自动降级切换"""
    
    def __init__(self, nodes: List[ModelNode]):
        if not nodes:
            raise ValueError("模型配置节点池（nodes）不能为空")
        
        # 【核心逻辑 1】根据用户手动指定的 priority 进行升序排序（数值越小越优先调用）
        self.nodes = sorted(nodes, key=lambda x: x.priority)
        self._print_routing_chain()

    def _print_routing_chain(self):
        """打印当前的路由调用链，方便调试与审计"""
        chain_str = " -> ".join([f"[{n.priority}]{n.provider}:{n.model_name}" for n in self.nodes])
        logger.info(f"🚀 路由链初始化成功。当前调用顺序: {chain_str}")

    def _create_chat_instance(self, node: ModelNode) -> BaseChatModel:
        """
        工厂方法：将配置节点转化为 LangChain ChatModel 实例。
        【可扩展性】后续若加入新厂商（如：DeepSeek, DashScope），只需在此处增加 elif 分支。
        """
        p = node.provider.lower()
        params = node.extra_params.copy()
        
        if p == "openai":
            return ChatOpenAI(model=node.model_name, **params)
            
        elif p == "ollama":
            return ChatOllama(model=node.model_name, **params)
            
        elif p == "google":
            # 兼容 Gemini 3.0+ 推荐的默认常规参数
            if "temperature" not in params:
                params["temperature"] = 1.0
            return ChatGoogleGenerativeAI(model=node.model_name,api_key=node.api_key,**params)
            
        elif p == "openrouter":
            # OpenRouter 使用 OpenAI 协议标准，通过特殊注入 base_url 兼容
            return ChatOpenAI(model=node.model_name,api_key=node.api_key, **params)
            
        else:
            raise ValueError(f"暂不支持的模型 Provider: {p}。请在 _create_chat_instance 中进行扩展。")

    def get_chat_model(self) -> BaseChatModel:
        """
        获取具备自动降级/额度切换能力的统一文本模型。
        利用 LangChain 内置的 with_fallbacks，在上游报错（如429限流、402欠费）时自动寻找下一顺位。
        """
        chat_models = [self._create_chat_instance(node) for node in self.nodes]
        primary_model = chat_models[0]
        
        # 【核心逻辑 2】如果配置了多个模型，自动挂载后备链
        if len(chat_models) > 1:
            logger.info(f"已成功为您注入 {len(chat_models) - 1} 个容错/额度兜底节点。")
            # return primary_model.with_fallbacks(chat_models[1:])
        
        return primary_model
    
    def get_falback_model(self) -> BaseChatModel:
        """
        获取具备自动降级/额度切换能力的统一文本模型。
        利用 LangChain 内置的 with_fallbacks，在上游报错（如429限流、402欠费）时自动寻找下一顺位。
        """
        chat_models = [self._create_chat_instance(node) for node in self.nodes]
        model = chat_models[-1]
        
        return model
    
    def get_chat_models(self) -> list[BaseChatModel]:
        """
        获取具备自动降级/额度切换能力的统一文本模型。
        利用 LangChain 内置的 with_fallbacks，在上游报错（如429限流、402欠费）时自动寻找下一顺位。
        """
        chat_models = [self._create_chat_instance(node) for node in self.nodes]
        return chat_models

    def get_embeddings(
        self,
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

# 1. 业务人员手动编排模型节点池与绝对优先级
my_model_pool = [
    ModelNode(
        provider="google",
        model_name="gemini-3.1-flash-lite",
        priority=1,
        api_key=GOOGLE_API_KEY,
        # extra_params={"project": GOOGLE_CLOUD_PROJECT}
    ),

    ModelNode(
        provider="google",
        model_name="gemma-4-31b-it",
        priority=2,
        api_key=GOOGLE_API_KEY,
        # extra_params={"project": GOOGLE_CLOUD_PROJECT}
    ),

    ModelNode(
        provider="openrouter",
        model_name="google/gemma-4-31b-it:free",
        priority=4,
        api_key=OPENROUTER_API_KEY,
        extra_params={"temperature": 0.2,"base_url": "https://openrouter.ai/api/v1"}
    ),
    
    ModelNode(
        provider="ollama",
        model_name="gemma4:31b-cloud",
        priority=3,
        api_key="",

        extra_params={"base_url": OLLAMA_API_BASE_URL, "temperature": 0.2}
    ),

    ModelNode(
        provider="ollama",
        model_name="qwen3.5:9b",
        priority=6,
        api_key="",
        extra_params={"base_url": OLLAMA_API_BASE_URL, "temperature": 0.2}
    ),
    
    
]

# 2. 将池子注入路由器（内部会自动将其按照 1 -> 2 -> 3 顺序重排）
ModelProvider = DynamicModelRouter(nodes=my_model_pool)


if __name__ == "__main__":
    # 3. 产生具有智能容错能力的终极 model 对象
    model = ModelProvider.get_chat_model()

    print(GOOGLE_API_KEY,GOOGLE_CLOUD_PROJECT)
    # 4. 业务层正常使用
    print("\n================ 业务请求开始 ================")
    try:
        # 如果此时 本地 Ollama 挂了或报额度不足，系统会自动向 Gemini 发起请求；
        # 如果 Gemini 也因为密钥过期报错，系统会自动向 OpenRouter 的 Claude 发起请求。
        response = model.invoke("请问什么是大模型的责任链路由模式？")
        print("💡 最终胜出模型的回答是:", response.content)
        
    except Exception as total_failure_exception:
        # 只有当 1、2、3 号模型全部都彻底崩溃、扣款失败时，才会走到这一步
        print("🚨 灾难性警报：所有后备渠道已全部沦陷！具体错误:", total_failure_exception)