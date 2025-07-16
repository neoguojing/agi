import os
from TTS.api import TTS
from torch.serialization import add_safe_globals
from TTS.utils.radam import RAdam 
from TTS.tts.configs.xtts_config import XttsConfig 
from TTS.tts.models.xtts import XttsAudioConfig,XttsArgs
from TTS.config.shared_configs import BaseDatasetConfig
from collections import defaultdict

os.environ["TTS_CACHE_DIR"] = "/data/model"
add_safe_globals([RAdam,defaultdict,dict,XttsConfig,XttsAudioConfig,BaseDatasetConfig,XttsArgs])

# 初始化并自动下载模型到指定目录
tts = TTS(model_name="tts_models/zh-CN/baker/tacotron2-DDC-GST")
tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2")
