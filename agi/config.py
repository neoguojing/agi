
import os
from dotenv import load_dotenv


load_dotenv(override=True)  # 加载 .env 文件中的环境变量

#服务相关
BASE_URL = os.environ.get("BASE_URL","http://localhost:8000")

# 存储相关
CACHE_DIR = os.path.abspath(os.environ.get("CACHE_DIR","./cache"))

os.makedirs(CACHE_DIR, exist_ok=True)
langchain_db_path = os.path.join(CACHE_DIR,"langchain.db")
LANGCHAIN_DB_PATH = os.environ.get("LANGCHAIN_DB_PATH",f"sqlite:///{langchain_db_path}")

# 模型相关
MODEL_PATH = os.environ.get(
    "MODEL_PATH", "/data/model"
)

## LLM
OLLAMA_API_BASE_URL = os.environ.get(
    "OLLAMA_API_BASE_URL", "http://localhost:11434"
)

OLLAMA_DEFAULT_MODE = os.environ.get("OLLAMA_DEFAULT_MODE", "qwen2.5:14b")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "xxx")

LLM_WITH_NO_THINKING = os.environ.get("LLM_WITH_NO_THINKING", "/no_think")
## embedding
RAG_EMBEDDING_MODEL = os.environ.get("RAG_EMBEDDING_MODEL", "bge-m3:latest")

## speech to text 
WHISPER_GPU_ENABLE = os.getenv("WHISPER_GPU_ENABLE", True)
WHISPER_MODEL_DIR = os.getenv("WHISPER_MODEL_DIR", os.path.join(MODEL_PATH,"wisper-v3-turbo-c2"))
if not WHISPER_GPU_ENABLE:
    WHISPER_MODEL_DIR = "base"

## tts 
# TTS_SPEAKER_WAV = os.getenv("TTS_SPEAKER_WAV", os.path.join(MODEL_PATH,"XTTS-v2","samples/zh-cn-sample.wav"))
TTS_SPEAKER_WAV = os.getenv("TTS_SPEAKER_WAV", "asset/zero_shot_prompt.wav")
TTS_GPU_ENABLE = os.getenv("TTS_GPU_ENABLE", True)
# TTS_MODEL_DIR = os.getenv("TTS_MODEL_DIR", os.path.join(MODEL_PATH,"tts_models--multilingual--multi-dataset--xtts_v2"))
TTS_MODEL_DIR = os.getenv("TTS_MODEL_DIR", os.path.join(MODEL_PATH,"cosyvoice/CosyVoice2-0.5B"))
if not TTS_GPU_ENABLE:
    TTS_MODEL_DIR = "tts_models/zh-CN/baker/tacotron2-DDC-GST"

TTS_FILE_SAVE_PATH = os.getenv("TTS_FILE_SAVE_PATH",os.path.join(CACHE_DIR, "audio"))

## image 
IMAGE_TO_IMAGE_MODEL_PATH = os.getenv("IMAGE_TO_IMAGE_MODEL_PATH",os.path.join(MODEL_PATH, "sdxl-turbo"))
# TEXT_TO_IMAGE_MODEL_PATH = os.getenv("TEXT_TO_IMAGE_MODEL_PATH",os.path.join(MODEL_PATH, "stable-diffusion-3.5-medium"))
TEXT_TO_IMAGE_MODEL_PATH = os.getenv("TEXT_TO_IMAGE_MODEL_PATH",os.path.join(MODEL_PATH, "sdxl-turbo"))
IMAGE_FILE_SAVE_PATH = os.getenv("IMAGE_FILE_SAVE_PATH",os.path.join(CACHE_DIR, "image"))


## rag
FILE_UPLOAD_PATH = os.getenv("FILE_UPLOAD_PATH",os.path.join(CACHE_DIR,"upload"))

## multi model
MULTI_MODEL_PATH = os.getenv("MULTI_MODEL_PATH",os.path.join(MODEL_PATH, "Qwen2.5-Omni-3B"))

## web
EXA_API_KEY = os.getenv("EXA_API_KEY","")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY","")
## stock
ALPHAVANTAGE_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY","")

## 系统参数
AGI_DEBUG = os.getenv("AGI_DEBUG","")
LANGCHAIN_DEBUG = os.getenv("LANGCHAIN_DEBUG","")

## LANGCHAIN 设置
if LANGCHAIN_DEBUG:
    from langchain.globals import set_debug
    from langchain.globals import set_verbose
    set_debug(True)
    set_verbose(True)

LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING",False)
LANGSMITH_ENDPOINT="https://api.smith.langchain.com"
LANGSMITH_API_KEY= os.getenv("LANGSMITH_API_KEY","")
LANGSMITH_PROJECT= os.getenv("LANGSMITH_PROJECT","agi")

## 日志设置
import logging
log = logging.getLogger(__name__)
# 创建控制台 handler
ch = logging.StreamHandler()
if AGI_DEBUG:
    log.setLevel(logging.DEBUG)
    ch.setLevel(logging.DEBUG)
else:
    log.setLevel(logging.INFO)
    ch.setLevel(logging.INFO)

logging.getLogger("chromadb").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
# 设置带文件名和行号的格式
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
)

ch.setFormatter(formatter)
log.addHandler(ch)
