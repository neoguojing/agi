import os
import time
from pathlib import Path
import torch
import soundfile as sf
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info

from agi.config import MULTI_MODEL_PATH as model_root,TTS_FILE_SAVE_PATH
from agi.llms.base import CustomerLLM,path_to_preview_url
from langchain_core.runnables import RunnableConfig
from typing import Any, Optional,Union
from pydantic import BaseModel, Field

from langchain_core.messages import AIMessage, HumanMessage
import base64

import traceback
import threading
from agi.config import log

audio_style = "width: 300px; height: 50px;"  # 添加样式

_load_lock = threading.Lock()
# GPU: 3B 13GB
class MultiModel(CustomerLLM):
    processor: Optional[Any] = Field(default=None)
    speaker_wav: str = Field(default="Chelsie") # also Ethan
    is_gpu: bool = Field(default=True)
    language: str = Field(default="zh-cn")
    save_file: bool = Field(default=True)
    
    
    def __init__(self, save_file: bool = False,**kwargs):
        super().__init__(**kwargs)

        self.save_file = save_file
        self.model = None
        self.processor = None
       
    def _load_model(self):
        """Initialize the TTS model based on the available hardware."""
        if self.model is not None and self.processor is not None:
            return  # 已初始化，不需要重复加载
        with _load_lock:
            if self.model is None or self.processor is None:
                log.info("loading MultiModel model...")
                self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
                    model_root,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    enable_audio_output=True,
                    attn_implementation="flash_attention_2"
                )
                self.processor = Qwen2_5OmniProcessor.from_pretrained(model_root)

    
    def invoke(self, inputs: Union[list[HumanMessage],HumanMessage], config: Optional[RunnableConfig] = None, **kwargs: Any) -> AIMessage:
        """Generate speech audio from input text."""
        try:
            self._load_model()
            return_audio = False
            if config:
                return_audio = config.get("configurable",{}).get("need_speech",False)
            
            content = [{
                "role": "system",
                "content": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.",
            }]

            if isinstance(inputs,HumanMessage):
                content.append({"role":"user","content":inputs.content})
            elif isinstance(inputs,list):
                inputs = inputs[-1]
                content.append({"role":"user","content":inputs.content})
            else:
                raise TypeError(f"Invalid input type: {type(inputs)}. Expected HumanMessage or List[HumanMessage].") 
            
            '''
            ['<|im_start|>system\nYou are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.<|im_end|>\n<|im_start|>user\n<|audio_bos|><|AUDIO|><|audio_eos|><|im_end|>\n<|im_start|>assistant\n']

            '''
            text = self.processor.apply_chat_template(content, add_generation_prompt=True, tokenize=False)
            audios, images, videos = process_mm_info(content, use_audio_in_video=True)
            inputs = self.processor(text=text, audios=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=True)
            inputs = inputs.to(self.model.device).to(self.model.dtype)

            text_ids = None
            audio = None
            if return_audio:
                text_ids, audio = self.model.generate(**inputs, return_audio=return_audio,spk=self.speaker_wav)
            else:
                text_ids = self.model.generate(**inputs, return_audio=return_audio,spk=self.speaker_wav)

            text = self.processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

            ret = AIMessage(content=[])
            for t in text:
                t = t.split("assistant\n")[-1]
                if return_audio:
                    file_path = f'{TTS_FILE_SAVE_PATH}/{int(time.time())}.wav'
                    Path(file_path).parent.mkdir(parents=True, exist_ok=True)

                    sf.write(
                        file_path,
                        audio.reshape(-1).detach().cpu().numpy(),
                        samplerate=24000,
                    )

                    audio_source = path_to_preview_url(file_path)
                    if kwargs.get('base64') is True:
                        with open(file_path, 'rb') as audio_file:
                            audio_bytes = audio_file.read()
                        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
                        audio_source = f"data:audio/wav;base64,{audio_base64}"

                    # 是否需要编码html
                    if kwargs.get('html') is True:
                        return ret.content.append(
                            {"type": "audio", "audio": f'<audio src="{audio_source}" {audio_style} controls></audio>\n',"file_path":file_path,"text":t}
                        )
                    
                    ret.content.append(
                        {"type": "audio", "audio": audio_source,"file_path":file_path,"text":t}
                    )
                else:
                    ret.content.append({"type": "text","text":t})
            return ret
                        
        except Exception as e:
            log.error(e)
            print(traceback.format_exc())

