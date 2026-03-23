import os
from dotenv import load_dotenv
import logging

# -----------------------------
# 工具函数
# -----------------------------
def get_env(name: str, default: str = None) -> str:
    """从环境变量读取值，未设置则使用默认值"""
    return os.getenv(name, default)

def get_env_bool(name: str, default: bool = False) -> bool:
    """从环境变量读取布尔值"""
    val = os.getenv(name, str(default))
    return val.lower() in ("1", "true", "yes", "on")

# -----------------------------
# 加载 .env 文件
# -----------------------------
load_dotenv(override=True)

# -----------------------------
# 服务相关
# -----------------------------
BASE_URL = get_env("BASE_URL", "http://localhost:8000")
API_KEY = get_env("API_KEY", "123")  # 请替换为实际的 API 密钥

# -----------------------------
# 存储相关
# -----------------------------
CACHE_DIR = os.path.abspath(get_env("CACHE_DIR", "/data/agi"))
os.makedirs(CACHE_DIR, exist_ok=True)

langchain_db_path = os.path.join(CACHE_DIR, "langchain.db")
LANGCHAIN_DB_PATH = get_env("LANGCHAIN_DB_PATH", f"sqlite:///{langchain_db_path}")

FILE_STORAGE_PATH = os.path.join(CACHE_DIR, "files")
os.makedirs(FILE_STORAGE_PATH, exist_ok=True)
FILE_STORAGE_URL = get_env("FILE_STORAGE_URL", f"file://{FILE_STORAGE_PATH}")

BROWSER_STORAGE_PATH = os.path.join(CACHE_DIR, "browser")
os.makedirs(BROWSER_STORAGE_PATH, exist_ok=True)

# -----------------------------
# 模型相关
# -----------------------------
MODEL_PATH = get_env("MODEL_PATH", "/data/model")

# LLM
OLLAMA_API_BASE_URL = get_env("OLLAMA_API_BASE_URL", "http://localhost:11434")
OLLAMA_DEFAULT_MODE = get_env("OLLAMA_DEFAULT_MODE", "qwen3.5:9b")
OLLAMA_THINKING_MODE = get_env("OLLAMA_SMALL_MODE", "qwen3:4b-thinking")
OPENAI_API_KEY = get_env("OPENAI_API_KEY", "xxx")
LLM_WITH_NO_THINKING = get_env("LLM_WITH_NO_THINKING", "/no_think")

# Embedding
RAG_EMBEDDING_MODEL = get_env("RAG_EMBEDDING_MODEL", "bge")
RAG_EMBEDDING_MODEL_PATH = get_env("RAG_EMBEDDING_MODEL_PATH", os.path.join(MODEL_PATH, "Qwen3-Embedding-0.6B"))
EMBEDDING_BASE_URL = get_env("EMBEDDING_BASE_URL", "http://localhost:8006")
CLUSTER_ALGO = get_env("CLUSTER_ALGO", "dpmeans")
COMPUTE_TYPE = get_env("COMPUTE_TYPE", "float16")

# Speech-to-Text
WHISPER_GPU_ENABLE = get_env_bool("WHISPER_GPU_ENABLE", True)
WHISPER_MODEL_DIR = os.path.join(MODEL_PATH, "wisper-v3-turbo-c2") if WHISPER_GPU_ENABLE \
    else os.path.join(MODEL_PATH, "models--Systran--faster-whisper-base")
WHISPER_MODLE_NAME = get_env("WHISPER_MODLE_NAME", "large")
WHISPER_BASE_URL = get_env("WHISPER_BASE_URL", "http://localhost:8003/v1/")

# TTS
TTS_SPEAKER_WAV = get_env("TTS_SPEAKER_WAV", "asset/zero_shot_prompt.wav")
TTS_GPU_ENABLE = get_env_bool("TTS_GPU_ENABLE", True)
TTS_MODEL_DIR = os.path.join(MODEL_PATH, "cosyvoice/CosyVoice2-0.5B") if TTS_GPU_ENABLE \
    else os.path.join(MODEL_PATH, "tts_models--zh-CN--baker--tacotron2-DDC-GST")
TTS_MODLE_NAME = get_env("TTS_MODLE_NAME", "cosyvoice")
TTS_BASE_URL = get_env("TTS_BASE_URL", "http://localhost:8002/v1/")

# Image
IMAGE_TO_IMAGE_MODEL_PATH = get_env("IMAGE_TO_IMAGE_MODEL_PATH", os.path.join(MODEL_PATH, "sdxl-turbo"))
TEXT_TO_IMAGE_MODEL_PATH = get_env("TEXT_TO_IMAGE_MODEL_PATH", os.path.join(MODEL_PATH, "sdxl-turbo"))
IMAGE_GEN_BASE_URL = get_env("IMAGE_GEN_BASE_URL", "http://localhost:8001/v1/")
TEXT_TO_IMAGE_MODEL_NAME = get_env("TEXT_TO_IMAGE_MODEL_NAME", "sdxl")

# Multi-model
MULTI_MODEL_PATH = get_env("MULTI_MODEL_PATH", os.path.join(MODEL_PATH, "Qwen2.5-Omni-3B"))
MULTI_MODEL_BASE_URL = get_env("MULTI_MODEL_BASE_URL", "http://localhost:8005/v1/")
MULTI_MODEL_NAME = get_env("MULTI_MODEL_NAME", "gemma")

# Web
EXA_API_KEY = get_env("EXA_API_KEY", "")
TAVILY_API_KEY = get_env("TAVILY_API_KEY", "")
SEARXNG_BASE_URL = get_env("SEARXNG_BASE_URL", "http://localhost:8091")

# Stock
ALPHAVANTAGE_API_KEY = get_env("ALPHAVANTAGE_API_KEY", "")

# 系统参数
STOP_WORDS_PATH = get_env("STOP_WORDS_PATH", "./asset/stopwords.txt")
AGI_DEBUG = get_env_bool("AGI_DEBUG")
AGI_LONG_TERM_MEMORY_ENABLED = get_env_bool("AGI_LONG_TERM_MEMORY_ENABLED", True)
AGI_MEMORY_PATH_PREFIX = get_env("AGI_MEMORY_PATH_PREFIX", "/memories/")
AGI_ASSISTANT_ID = get_env("AGI_ASSISTANT_ID", "deepagent_main")
AGI_TENANT_ID = get_env("AGI_TENANT_ID", "default_tenant")

# -----------------------------
# LangChain 调试
# -----------------------------
def init_langchain_debug():
    if not get_env_bool("LANGCHAIN_DEBUG"):
        return

    from langchain.globals import set_debug, set_verbose

    set_debug(True)
    set_verbose(True)

    os.environ["LANGSMITH_TRACING"] = "true" if get_env_bool("LANGSMITH_TRACING") else "false"
    os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
    os.environ["LANGSMITH_API_KEY"] = get_env("LANGSMITH_API_KEY", "")
    os.environ["LANGSMITH_PROJECT"] = get_env("LANGSMITH_PROJECT", "agi")

# -----------------------------
# Logger 初始化
# -----------------------------
def init_logger() -> logging.Logger:
    log = logging.getLogger()
    if not log.handlers:
        handler = logging.StreamHandler()
        level = logging.DEBUG if AGI_DEBUG else logging.INFO
        log.setLevel(level)
        handler.setLevel(level)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
        )
        handler.setFormatter(formatter)
        log.addHandler(handler)

        # 降低冗余库日志
        for noisy_logger in ["chromadb", "httpcore", "httpx", "cosyvoice_tts"]:
            logging.getLogger(noisy_logger).setLevel(logging.WARNING)

    return log

# -----------------------------
# 初始化
# -----------------------------
log = init_logger()
init_langchain_debug()