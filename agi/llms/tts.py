import os
import time
from datetime import date
from pathlib import Path
import torch
import numpy as np
import torchaudio
from TTS.api import TTS
from TTS.utils.radam import RAdam 
from TTS.tts.configs.xtts_config import XttsConfig 
from TTS.tts.models.xtts import XttsAudioConfig,XttsArgs
from TTS.config.shared_configs import BaseDatasetConfig
from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav
from collections import defaultdict
from agi.config import TTS_MODEL_DIR as model_root, CACHE_DIR, TTS_SPEAKER_WAV,TTS_GPU_ENABLE,TTS_FILE_SAVE_PATH
from agi.llms.base import CustomerLLM,parse_input_messages,path_to_preview_url
from langchain_core.runnables import RunnableConfig,run_in_executor
from typing import Any, Optional,Union,ClassVar
from pydantic import BaseModel, Field
from collections.abc import (
    AsyncIterator,
    Iterator,
)
from langchain_core.messages import AIMessage, HumanMessage,AIMessageChunk
import base64
import numpy as np
from pydub import AudioSegment
import io
from torch.serialization import add_safe_globals
from agi.config import log

from queue import Queue,Full
from threading import Lock

audio_style = "width: 300px; height: 50px;"  # 添加样式
END_TAG = b'\x00' * 8192
# for torch 2.6
add_safe_globals([RAdam,defaultdict,dict,XttsConfig,XttsAudioConfig,BaseDatasetConfig,XttsArgs])
class TextToSpeech(CustomerLLM):
    tts: Optional[Any] = Field(default=None)
    speaker_wav: Any = Field(default=TTS_SPEAKER_WAV)
    is_gpu: bool = Field(default=TTS_GPU_ENABLE)
    language: str = Field(default="zh-cn")
    save_file: bool = Field(default=True)

     # ✅ 明确声明为类变量，避免 Pydantic 处理
    _queues: ClassVar[dict[str, Queue]] = {}
    _lock: ClassVar[Lock] = Lock()
    
    def __init__(self,save_file: bool = False,**kwargs):
        super().__init__(**kwargs)

        self.save_file = save_file
        self.model = None

    @classmethod
    def get_queue(cls, tenant_id: str="default") -> Queue:
        if tenant_id not in cls._queues:
            with cls._lock:  # 确保线程安全
                if tenant_id not in cls._queues:
                    cls._queues[tenant_id] = Queue(maxsize=1000)
        return cls._queues[tenant_id]
       
    def _load_model(self):
        """Initialize the TTS model based on the available hardware."""
        if self.tts is None:
            if self.is_gpu:
                # GPU：2739MB
                log.info("loading TextToSpeech model(GPU)...")
                # model_path = os.path.join(model_root, "tts_models--multilingual--multi-dataset--xtts_v2")
                model_path = model_root
                if "cosyvoice" in model_path:
                    self.speaker_wav = load_wav(TTS_SPEAKER_WAV, 16000)
                    self.tts = CosyVoice2(model_path, load_jit=False, load_trt=False, fp16=False,use_flow_cache=False)
                    self.model = self.tts.model
                else:
                    config_path = os.path.join(model_path, "config.json")
                    log.info("use ts_models--multilingual--multi-dataset--xtts_v2")
                    self.tts = TTS(model_path=model_path, config_path=config_path).to(torch.device("cuda"))
                    self.model = self.tts.synthesizer
            else:
                log.info("loading TextToSpeech model(CPU)...")
                # self.tts = TTS(model_name="tts_models/zh-CN/baker/tacotron2-DDC-GST").to(torch.device("cpu"))
                self.tts = TTS(model_name=model_root).to(torch.device("cpu"))
                # model_dir = os.path.join(model_root, "tts_models--zh-CN--baker--tacotron2-DDC-GST")
                # model_path = os.path.join(model_dir, "model_file.pth")
                # config_path = os.path.join(model_dir, "config.json")
                # return TTS(model_path=model_path, config_path=config_path)
                self.model = self.tts.synthesizer

    def list_available_models(self):
        """Return a list of available TTS models."""
        self._load_model()
        return self.tts.list_models()

    def invoke(self, input: Union[list[HumanMessage],HumanMessage,str], config: Optional[RunnableConfig] = None, **kwargs: Any) -> AIMessage:
        """Generate speech audio from input text."""
       
        self._load_model()

        user_id = config.get("configurable").get("user_id")
        input_str = None
        if isinstance(input,str):
            input_str = input
        else:
            _,input_str = parse_input_messages(input)
            
        log.info(f"tts input: {input_str}")
        final_np_pcm = np.array([], dtype=np.int16)
        file_path = None

        if self.save_file:
            file_path = self.save_audio_to_file(text=input_str)
            audio_source = path_to_preview_url(file_path)
            return AIMessage(content=[
                {"type": "audio", "audio": audio_source,"file_path":file_path,"text":input_str}
            ])
        
        # Generate audio samples and return as ByteIO
        # 原始音频需要编码，不方便使用
        for sample in self.generate_audio_samples(input_str):
            list_pcm = self.uniform_model_output(sample)
            np_pcm = self.list_pcm_normalization_int16(list_pcm)
            self.send_pcm(user_id,np_pcm)
            np.append(final_np_pcm,np_pcm)

        # 发送结束标记
        self.send_pcm(user_id,None,end_tag=END_TAG)
        # 默认返回路径
        audio_source = self.np_pcm_to_wave(final_np_pcm)
        # 是否需要编码html
        if kwargs.get('html') is True:
            return AIMessage(content=[
                {"type": "audio", "audio": f'<audio src="{audio_source}" {audio_style} controls></audio>\n',"file_path":file_path,"text":input_str}
            ])
        
        log.info(f"tts output: {audio_source}")
        return AIMessage(content=[
            {"type": "audio", "audio": audio_source,"text":input_str}
        ])
        
    async def ainvoke(self, input: Union[list[HumanMessage],HumanMessage,str], config: Optional[RunnableConfig] = None, **kwargs: Any) -> AIMessage:
        log.debug("tts ainvoke ---------------")
        # return self.invoke(input, config=config, **kwargs)
        return await run_in_executor(config, self.invoke, input, config, **kwargs)
        
    def generate_audio_samples(self, text: str):
        """Generate audio samples from the input text."""
        try:
            if self.is_gpu:
                if "cosyvoice" in model_root:
                    # 流式合成，超长文本报错
                    for sentence in self.sentence_segmenter(text):
                        for c_idx, data in enumerate(self.tts.inference_cross_lingual(sentence, self.speaker_wav, stream=False)):
                            tensor_data = data['tts_speech']
                            print("************",self.tts.sample_rate,tensor_data)
                            yield tensor_data
                else:
                    for sentence in self.sentence_segmenter(text):
                        yield self.tts.tts(text=sentence, speaker_wav=self.speaker_wav, language=self.language)
            else:
                for sentence in self.sentence_segmenter(text):
                    yield self.tts.tts(text=sentence, speaker_wav=self.speaker_wav)
        except Exception as e:
            log.error(f"Error generating audio samples: {e}")
            raise RuntimeError("Failed to generate audio samples.")

    def save_audio_to_file(self, text: str, file_path: str = "") -> str:
        """Save the generated audio to a file and return the file path."""
        if not file_path:
            # file_name = f'audio/{date.today().strftime("%Y_%m_%d")}/{int(time.time())}.wav'
            file_path = f'{TTS_FILE_SAVE_PATH}/{int(time.time())}.wav'
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
        try:
            if "cosyvoice" in model_root:
                for c_idx, data in enumerate(self.tts.inference_cross_lingual(text, self.speaker_wav, stream=False)):
                    torchaudio.save(file_path, data['tts_speech'], self.tts.sample_rate)
            else:
                self.tts.tts_to_file(
                    text=text,
                    speaker_wav=self.speaker_wav,
                    language=self.language if self.is_gpu else None,
                    file_path=file_path
                )
        except Exception as e:
            log.error(f"Error saving audio to file: {e}")
            raise RuntimeError("Failed to save audio to file.")
        
        return file_path

    # 将模型输出，统一转换为list
    def uniform_model_output(self,obj):
        if isinstance(obj, torch.Tensor) or isinstance(obj, np.ndarray):
            ndim = obj.ndim
            if ndim == 2:
                obj = obj[0].tolist()
            else:
                obj = obj.tolist()

        else:
            print(f"是其他类型: {type(obj)}")
        return obj

    # 将np格式的pcm转换为wav格式，然后内存编码为base64或者文件
    def np_pcm_to_wave(self,audio_array: np.ndarray, sample_rate: int=24000,to_base64=False) -> str:
        # 创建音频段（单声道，int16格式）
        audio_segment = AudioSegment(
            audio_array.tobytes(),
            frame_rate=sample_rate,
            sample_width=2,  # int16 = 2 bytes
            channels=1
        )

        audio_source = None
        if to_base64:
            # 写入内存中的 BytesIO
            mp3_io = io.BytesIO()
            audio_segment.export(mp3_io, format="wav")
            mp3_io.seek(0)

            # Base64 编码
            mp3_base64 = base64.b64encode(mp3_io.read()).decode("utf-8")
            audio_source = f"data:audio/wav;base64,{mp3_base64}"
        else:
            audio_source = f'{TTS_FILE_SAVE_PATH}/{int(time.time())}.wav'
            audio_segment.export(audio_source, format="wav")

        audio_source = path_to_preview_url(audio_source)

        return audio_source
    
    # 列表格式的pcm数据归一化，转换为int16
    # 输出： np.array
    def list_pcm_normalization_int16(self,audio_data):
        """
        将音频数据转换为 int16 格式，尽量减少音质损失，通过归一化和缩放。
        
        :param audio_data: 输入的音频数据，可以是 list、ndarray 或其他类型
        :return: 转换后的 int16 格式的音频数据
        """
        # 将输入数据转换为 numpy 数组
        audio_array = np.array(audio_data)
        
        # 如果音频数据已经是 int16 类型，直接返回
        if audio_array.dtype == np.int16:
            print("音频数据已经是 int16 格式，无需转换。")
            return audio_array
        
        # 如果是浮动类型（如 float32），先缩放到 [-32768, 32767] 范围
        if np.issubdtype(audio_array.dtype, np.floating):
            # print("检测到浮动类型音频数据，正在进行归一化并转换为 int16。")
            # 将浮动数值缩放到 int16 范围
            audio_array = np.clip(audio_array, -1.0, 1.0)  # 防止溢出
            audio_array = np.int16(audio_array * 32767)  # 缩放到 int16 范围
            return audio_array
        
        # 如果是其他整数类型（如 int32），首先缩放到 int16 范围
        if np.issubdtype(audio_array.dtype, np.integer):
            print(f"检测到整数类型音频数据（{audio_array.dtype}），正在进行缩放并转换为 int16。")
            # 如果数据是 int32，可以通过其最大值和最小值来进行缩放
            max_val = np.max(audio_array)
            min_val = np.min(audio_array)
            
            # 根据 int32 的范围缩放到 int16 的范围
            if max_val != min_val:  # 防止除以零
                scaling_factor = 32767.0 / max(abs(max_val), abs(min_val))  # 计算缩放因子
                audio_array = np.int16(audio_array * scaling_factor)
            else:
                audio_array = np.int16(audio_array)  # 如果数据范围很小，直接转换
            
            return audio_array
    # 句子分割
    def sentence_segmenter(self,text, min_length=20, max_length=50):
        if len(text) < max_length:
            log.info(f"sentence_segmenter:{text}")
            # yield text
            return [text]

        import re
        # 使用正则表达式根据中英文标点进行分割
        sentence_endings = r'(?<=[。！？.!?])\s*'
        sentences = re.split(sentence_endings, text)
        
        # 移除空的句子
        sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
        
        # 合并小于 min_length 的句子
        i = 0
        while i < len(sentences) - 1:
            if len(sentences[i]) + len(sentences[i + 1]) < min_length:
                sentences[i] = sentences[i] + " " + sentences[i + 1]
                sentences.pop(i + 1)
            else:
                i += 1
        
        # 拆分大于 max_length 的句子
        result = []
        for sentence in sentences:
            if len(sentence) <= max_length:
                result.append(sentence)  # 如果句子长度小于或等于 max_length，直接添加
                # log.info(f"sentence_segmenter:{sentence}")
                # yield sentence
            
            else:
                while len(sentence) > max_length:
                    # 优先在较大的标点符号后拆分，如：句号（。）、问号（？）等
                    split_point = max(sentence.rfind(p, 0, max_length) for p in ['。', '！', '？', '.', '!', '?'])
                    
                    if split_point == -1:  # 如果找不到大的标点符号，则尝试更小的标点符号（如逗号）
                        # 在中英文逗号、分号、冒号后拆分
                        split_point = max(sentence.rfind(p, 0, max_length) for p in [',', '，', ';', '；', ':', '：'])
                        
                        if split_point == -1:  # 如果没有找到标点符号，则直接按 max_length 截取
                            split_point = max_length
                    
                    # 分割句子
                    result.append(sentence[:split_point + 1].strip())
                    # log.info(f"sentence_segmenter:{sentence[:split_point + 1].strip()}")
                    # yield sentence[:split_point + 1].strip()
                    sentence = sentence[split_point + 1:].strip()
                
                # 添加剩余的部分
                if sentence:
                    result.append(sentence)
                    # log.info(f"sentence_segmenter:{sentence}")
                    # yield sentence
        log.debug(f"sentence_segmenter out:{result}")
        return result
    
    def send_pcm(self, tenant_id: str, pcm_np: np.ndarray, chunk_size: int = 1024 ,end_tag=None):
        queue = self.get_queue(tenant_id)
        # 查看是否结束
        if end_tag:
             queue.put(END_TAG)
             return
         
        buffer = np.array([], dtype=np.int16)  # int16 数据类型匹配
        offset = 0
        total_len = len(pcm_np)

        while offset < total_len:
            end = min(offset + chunk_size, total_len)
            chunk = pcm_np[offset:end]
            buffer = np.concatenate([buffer, chunk])
            offset += chunk_size

            while len(buffer) >= chunk_size:
                to_send = buffer[:chunk_size]
                try:
                    queue.put(to_send.tobytes(), block=False)
                    log.debug(f"send_pcm: {len(to_send)} samples, {len(to_send.tobytes())} bytes")
                except Full:
                    log.warning("Queue full, dropping PCM chunk.")
                buffer = buffer[chunk_size:]

        # 最后剩余不足chunk_size的部分，进行补零填充
        if len(buffer) > 0:
            padding = np.zeros(chunk_size - len(buffer), dtype=np.int16)
            final_chunk = np.concatenate([buffer, padding])
            try:
                queue.put(final_chunk.tobytes(), block=False)
                log.debug(f"send_pcm (final): {len(final_chunk)} samples, {len(final_chunk.tobytes())} bytes")
            except Full:
                log.warning("Queue full, dropping final PCM chunk.")


