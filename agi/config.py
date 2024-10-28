
import os
from dotenv import load_dotenv

load_dotenv()  # 加载 .env 文件中的环境变量


MODEL_PATH = os.environ.get(
    "MODEL_PATH", "/data/model"
)

CACHE_DIR = os.environ.get("CACHE_DIR","./cache")

OLLAMA_API_BASE_URL = os.environ.get(
    "OLLAMA_API_BASE_URL", "http://localhost:11434"
)

OLLAMA_DEFAULT_MODE = "qwen2.5:14b"

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "xxx")

RAG_EMBEDDING_MODEL = os.environ.get("RAG_EMBEDDING_MODEL", "bge-m3:latest")

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
WHISPER_MODEL_DIR = os.getenv("WHISPER_MODEL_DIR", f"{MODEL_PATH}/")

TTS_SPEAKER_WAV = os.getenv("TTS_SPEAKER_WAV", os.path.join(MODEL_PATH,"XTTS-v2","samples/zh-cn-sample.wav"))

# 其他配置项
