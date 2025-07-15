import os
from TTS.api import TTS

os.environ["TTS_CACHE_DIR"] = "/data/model"

# 初始化并自动下载模型到指定目录
tts = TTS(model_name="tts_models/zh-CN/baker/tacotron2-DDC-GST")
tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2")
