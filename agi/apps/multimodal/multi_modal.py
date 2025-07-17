import threading
import time
import soundfile as sf
import base64
from pathlib import Path
from typing import Any
import torch
from agi.config import MULTI_MODEL_PATH as model_root,FILE_STORAGE_PATH,log
from agi.utils.common import path_to_preview_url
from qwen_omni_utils import process_mm_info
import traceback

audio_style = "width: 300px; height: 50px;"  # 添加样式

class MultiModel:
    processor: Any = None
    speaker_wav: str = "Chelsie" # also Ethan
    is_gpu: bool = True
    language: str = "zh-cn"
    save_file: bool = True
    def __init__(self, model_path: str=model_root, timeout: int = 300,save_file=True):
        """
        model_path: 模型本地路径或 huggingface 名称
        timeout: 超过多少秒未使用就自动卸载（默认10分钟）
        """
        self.model_path = model_path
        self.timeout = timeout
        self.model = None
        self.last_used = 0
        self.lock = threading.Lock()
        self.monitor_thread = threading.Thread(target=self._monitor, daemon=True)
        self.monitor_thread.start()
        self.save_file = save_file

    def get_model(self):
        """访问模型，如果未加载则自动加载"""
        with self.lock:
            self.last_used = time.time()
            if self.model is None:
                self._load()
            return self.model

    def _load(self):
        print(f"[Model] Loading model from {self.model_path}")
        from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
        self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            model_root,
            torch_dtype=torch.float16,
            device_map="auto",
            enable_audio_output=True,
            attn_implementation="flash_attention_2"
        )
        self.processor = Qwen2_5OmniProcessor.from_pretrained(model_root)

    def invoke(self, text: str="",audio: any=None,image: any = None,video: any = None,return_audio=False,return_fmt=""):
        try:
            self.get_model()

            conversation = [{
                    "role": "system",
                    "content": [{"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}],
                }
            ]

            inputs = {
                "role":"user",
                "content":[]
            }

            if text != "":
                inputs["content"].append({"type":"text","text":text})
            if audio is not None:
                inputs["content"].append({"type":"audio","audio":audio})
            if image is not None:
                inputs["content"].append({"type":"image","image":image})
            if video is not None:
                inputs["content"].append({"type":"video","video":video})

            conversation.append(inputs)
            '''
            ['<|im_start|>system\nYou are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.<|im_end|>\n<|im_start|>user\n<|audio_bos|><|AUDIO|><|audio_eos|><|im_end|>\n<|im_start|>assistant\n']

            '''
            text = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
            print(conversation)
            print(text)
            audios, images, videos = process_mm_info(conversation, use_audio_in_video=True)
            inputs = self.processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=True)
            inputs = inputs.to(self.model.device).to(self.model.dtype)
            

            text_ids = None
            audio = None
            if return_audio:
                text_ids, audio = self.model.generate(**inputs, return_audio=return_audio,speaker=self.speaker_wav)
            else:
                text_ids = self.model.generate(**inputs, return_audio=return_audio)

            text = self.processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

            text_output = ""
            file_path = ""
            audio_source = ""
            for t in text:
                text_output = t.split("assistant\n")[-1]
                if return_audio:
                    file_path = f'{FILE_STORAGE_PATH}/{int(time.time())}.wav'
                    Path(file_path).parent.mkdir(parents=True, exist_ok=True)

                    sf.write(
                        file_path,
                        audio.reshape(-1).detach().cpu().numpy(),
                        samplerate=24000,
                    )

                    audio_source = path_to_preview_url(file_path)
                    if return_fmt == "base64":
                        with open(file_path, 'rb') as audio_file:
                            audio_bytes = audio_file.read()
                        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
                        audio_source = f"data:audio/wav;base64,{audio_base64}"

                    # 是否需要编码html
                    if return_fmt == "html":
                        audio_source = f'<audio src="{audio_source}" {audio_style} controls></audio>\n'

            return text_output,file_path,audio_source
                        
        except Exception as e:
            log.error(e)
            print(traceback.format_exc())

    def _unload(self):
        print(f"[Model] Unloading model from {self.model_path}")
        del self.model
        self.model = None
        self.processor = None
        torch.cuda.empty_cache()

    def _monitor(self):
        """后台线程定期检查是否应卸载模型"""
        while True:
            time.sleep(30)
            with self.lock:
                if self.model and (time.time() - self.last_used > self.timeout):
                    self._unload()
