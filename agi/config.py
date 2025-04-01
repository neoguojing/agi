
import os
from dotenv import load_dotenv

load_dotenv()  # 加载 .env 文件中的环境变量

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
## embedding
RAG_EMBEDDING_MODEL = os.environ.get("RAG_EMBEDDING_MODEL", "bge-m3:latest")

## speech to text 
WHISPER_GPU_ENABLE = os.getenv("WHISPER_GPU_ENABLE", True)
WHISPER_MODEL_DIR = os.getenv("WHISPER_MODEL_DIR", os.path.join(MODEL_PATH,"wisper-v3-turbo-c2"))
if not WHISPER_GPU_ENABLE:
    WHISPER_MODEL_DIR = "base"

## tts 
TTS_SPEAKER_WAV = os.getenv("TTS_SPEAKER_WAV", os.path.join(MODEL_PATH,"XTTS-v2","samples/zh-cn-sample.wav"))
TTS_GPU_ENABLE = os.getenv("TTS_DEVICE", True)
TTS_MODEL_DIR = os.getenv("TTS_MODEL_DIR", os.path.join(MODEL_PATH,"tts_models--multilingual--multi-dataset--xtts_v2"))
if not TTS_GPU_ENABLE:
    TTS_MODEL_DIR = "tts_models/zh-CN/baker/tacotron2-DDC-GST"

TTS_FILE_SAVE_PATH = os.getenv("TTS_FILE_SAVE_PATH",os.path.join(CACHE_DIR, "audio"))

## image 
IMAGE_TO_IMAGE_MODEL_PATH = os.getenv("IMAGE_TO_IMAGE_MODEL_PATH",os.path.join(MODEL_PATH, "sdxl-turbo"))
TEXT_TO_IMAGE_MODEL_PATH = IMAGE_TO_IMAGE_MODEL_PATH
IMAGE_FILE_SAVE_PATH = os.getenv("IMAGE_FILE_SAVE_PATH",os.path.join(CACHE_DIR, "image"))


## rag
FILE_UPLOAD_PATH = os.getenv("FILE_UPLOAD_PATH",os.path.join(CACHE_DIR,"upload"))

## multi model
MULTI_MODEL_PATH = os.getenv("IMAGE_TO_IMAGE_MODEL_PATH",os.path.join(MODEL_PATH, "Qwen2.5-Omni-7B"))

## web
EXA_API_KEY = os.getenv("FILE_UPLOAD_PATH","")


## 系统参数
AGI_DEBUG = os.getenv("AGI_DEBUG",True)
