
import os
from dotenv import load_dotenv
import logging


load_dotenv(override=True)  # 加载 .env 文件中的环境变量
#服务相关
BASE_URL = os.environ.get("BASE_URL","http://localhost:8000")
API_KEY =os.environ.get("API_KEY", "123")  # 请替换为实际的 API 密钥
# 存储相关
CACHE_DIR = os.path.abspath(os.environ.get("CACHE_DIR","./cache"))

os.makedirs(CACHE_DIR, exist_ok=True)
langchain_db_path = os.path.join(CACHE_DIR,"langchain.db")
LANGCHAIN_DB_PATH = os.environ.get("LANGCHAIN_DB_PATH",f"sqlite:///{langchain_db_path}")

FILE_STORAGE_PATH = os.path.join(CACHE_DIR,'files')
FILE_STORAGE_URL = os.getenv("FILE_STORAGE_URL",f"file://{FILE_STORAGE_PATH}")
# 本地文件系统
# FILE_STORAGE_URL = "file:///data/files/"
# # AWS S3
# FILE_STORAGE_URL = "s3://your-bucket-name/path/"
# # MinIO（需额外配置环境变量或 boto3 session）
# FILE_STORAGE_URL = "s3://minio-bucket/path/"

# 模型相关
MODEL_PATH = os.environ.get(
    "MODEL_PATH", "/data/model"
)

## LLM
OLLAMA_API_BASE_URL = os.environ.get(
    "OLLAMA_API_BASE_URL", "http://localhost:11434"
)

OLLAMA_DEFAULT_MODE = os.environ.get("OLLAMA_DEFAULT_MODE", "qwen3:4b-instruct")
OLLAMA_THINKING_MODE = os.environ.get("OLLAMA_SMALL_MODE", "qwen3:4b-thinking")


OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "xxx")

LLM_WITH_NO_THINKING = os.environ.get("LLM_WITH_NO_THINKING", "/no_think")
## embedding
# RAG_EMBEDDING_MODEL = os.environ.get("RAG_EMBEDDING_MODEL", "bge-m3:latest")
RAG_EMBEDDING_MODEL = os.environ.get("RAG_EMBEDDING_MODEL", "bge") #qwen
RAG_EMBEDDING_MODEL_PATH = os.getenv("RAG_EMBEDDING_MODEL_PATH", os.path.join(MODEL_PATH,"Qwen3-Embedding-0.6B"))
EMBEDDING_BASE_URL = os.getenv("EMBEDDING_BASE_URL", "http://localhost:8006")
CLUSTER_ALGO = os.getenv("CLUSTER_ALGO", "dpmeans") #hdbscan

COMPUTE_TYPE = os.getenv("COMPUTE_TYPE", "float16")
## speech to text 
WHISPER_GPU_ENABLE =os.getenv("WHISPER_GPU_ENABLE", "true").lower() in ("1", "true", "yes")
WHISPER_MODEL_DIR = os.getenv("WHISPER_MODEL_DIR", os.path.join(MODEL_PATH,"wisper-v3-turbo-c2"))
if not WHISPER_GPU_ENABLE:
    WHISPER_MODEL_DIR = os.getenv("WHISPER_MODEL_DIR", os.path.join(MODEL_PATH,"models--Systran--faster-whisper-base"))
WHISPER_MODLE_NAME = os.getenv("WHISPER_MODLE_NAME", "large") #base
WHISPER_BASE_URL = os.getenv("WHISPER_BASE_URL", "http://localhost:8003/v1/")
## tts 
# TTS_SPEAKER_WAV = os.getenv("TTS_SPEAKER_WAV", os.path.join(MODEL_PATH,"XTTS-v2","samples/zh-cn-sample.wav"))
TTS_SPEAKER_WAV = os.getenv("TTS_SPEAKER_WAV", "asset/zero_shot_prompt.wav")
# TTS_MODEL_DIR = os.getenv("TTS_MODEL_DIR", os.path.join(MODEL_PATH,"tts_models--multilingual--multi-dataset--xtts_v2"))
TTS_MODEL_DIR = os.getenv("TTS_MODEL_DIR", os.path.join(MODEL_PATH,"cosyvoice/CosyVoice2-0.5B"))
TTS_GPU_ENABLE = os.getenv("TTS_GPU_ENABLE", "True").lower() == "true"
if not TTS_GPU_ENABLE:
    TTS_MODEL_DIR = os.getenv("TTS_MODEL_DIR", os.path.join(MODEL_PATH,"tts_models--zh-CN--baker--tacotron2-DDC-GST"))
TTS_MODLE_NAME = os.getenv("TTS_MODLE_NAME","cosyvoice") #xtts  dag vibevoice
TTS_BASE_URL = os.getenv("TTS_BASE_URL", "http://localhost:8002/v1/")

## image 
IMAGE_TO_IMAGE_MODEL_PATH = os.getenv("IMAGE_TO_IMAGE_MODEL_PATH",os.path.join(MODEL_PATH, "sdxl-turbo"))
# TEXT_TO_IMAGE_MODEL_PATH = os.getenv("TEXT_TO_IMAGE_MODEL_PATH",os.path.join(MODEL_PATH, "stable-diffusion-3.5-medium"))
TEXT_TO_IMAGE_MODEL_PATH = os.getenv("TEXT_TO_IMAGE_MODEL_PATH",os.path.join(MODEL_PATH, "sdxl-turbo"))
IMAGE_GEN_BASE_URL = os.getenv("IMAGE_GEN_BASE_URL","http://localhost:8001/v1/")
TEXT_TO_IMAGE_MODEL_NAME = os.getenv("TEXT_TO_IMAGE_MODEL_NAME","sdxl") # sd3.5

## multi model
MULTI_MODEL_PATH = os.getenv("MULTI_MODEL_PATH",os.path.join(MODEL_PATH, "Qwen2.5-Omni-3B"))
MULTI_MODEL_BASE_URL = os.getenv("MULTI_MODEL_BASE_URL","http://localhost:8005/v1/")
MULTI_MODEL_NAME = os.getenv("MULTI_MODEL_NAME","gemma")  #qwen
## web
EXA_API_KEY = os.getenv("EXA_API_KEY","")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY","")

## stock
ALPHAVANTAGE_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY","")

## 系统参数
STOP_WORDS_PATH = os.getenv("STOP_WORDS_PATH","./asset/stopwords.txt")
def get_env_bool(name: str, default=False) -> bool:
    val = os.getenv(name, str(default))
    return val.lower() in ("1", "true", "yes", "on")


def init_langchain_debug():
    if not get_env_bool("LANGCHAIN_DEBUG"):
        return

    from langchain.globals import set_debug, set_verbose

    set_debug(True)
    set_verbose(True)

    # 可根据需要导出环境变量或做其他处理
    # 设置相关环境变量
    os.environ["LANGSMITH_TRACING"] = "true" if get_env_bool("LANGSMITH_TRACING") else "false"
    os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
    # 从 .env 中读取，若未设置则设默认
    os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY", "")
    os.environ["LANGSMITH_PROJECT"] = os.getenv("LANGSMITH_PROJECT", "agi")

logging.getLogger("cosyvoice_tts").setLevel(logging.ERROR)
AGI_DEBUG = get_env_bool("AGI_DEBUG")

# session / memory persistence
AGI_LONG_TERM_MEMORY_ENABLED = get_env_bool("AGI_LONG_TERM_MEMORY_ENABLED", True)
AGI_MEMORY_PATH_PREFIX = os.getenv("AGI_MEMORY_PATH_PREFIX", "/memories/")
AGI_ASSISTANT_ID = os.getenv("AGI_ASSISTANT_ID", "deepagent_main")
AGI_TENANT_ID = os.getenv("AGI_TENANT_ID", "default_tenant")

def init_logger() -> logging.Logger:
    log = logging.getLogger()

    if not log.handlers:  # 避免重复添加 handler
        handler = logging.StreamHandler()
        debug_mode = get_env_bool("AGI_DEBUG")
        level = logging.DEBUG if debug_mode else logging.INFO
        log.setLevel(level)
        handler.setLevel(level)

        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
        )
        handler.setFormatter(formatter)
        log.addHandler(handler)

        # 降低冗余库的日志等级
        for noisy_logger in ["chromadb", "httpcore", "httpx"]:
            logging.getLogger(noisy_logger).setLevel(logging.WARNING)

    return log


# 初始化
log = init_logger()
