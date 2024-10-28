import os

from TTS.api import TTS
from agi.config import MODEL_PATH as model_root,CACHE_DIR
from agi.llms.base import CustomerLLM,MultiModalMessage,Audio
from langchain_core.runnables import RunnableConfig
from typing import Any, List, Mapping, Optional,Union

class TextToSpeech(CustomerLLM):
    def __init__(self, model_path: str = os.path.join(model_root,"tts_models--multilingual--multi-dataset--xtts_v2"), 
                 speaker_wav: str = os.path.join(model_root,"XTTS-v2","samples/zh-cn-sample.wav"), 
                 language: str = "zh-cn",save_file: bool = True):
        config_path = os.path.join(model_path,"config.json")
        tts = TTS(model_path=model_path,config_path=config_path)
        # self.tts = TTS(model_name="tts_models--zh-CN--baker--tacotron2-DDC-GST").to(self.device)
        
        super(TextToSpeech, self).__init__(llm=tts.synthesizer)
        self.tts = tts.to(self.device)
        self.speaker_wav = speaker_wav
        self.language = language
        self.save_file = save_file

    def list_available_models(self):
        return self.tts.list_models()

    def invoke(
        self, input: MultiModalMessage, config: Optional[RunnableConfig] = None, **kwargs: Any
    ) -> MultiModalMessage:
        output = None
        if self.save_file:
            file_path = self.save_audio_to_file(text=input.content)
            output = MultiModalMessage(content=input.content,audio=Audio(file_path=file_path))
        else:
            samples = self.tts.tts(text=input.content, speaker_wav=self.speaker_wav, language=self.language)
            output = MultiModalMessage(content=input.content,audio=Audio(samples=samples))
        return output

    def save_audio_to_file(self, text: str, file_path: str=CACHE_DIR):
        # self.tts.tts_to_file(text=text, speaker_wav=self.speaker_wav, language=self.language, file_path=file_path)
        return self.tts.tts_to_file(text=text, speaker_wav=self.speaker_wav, file_path=file_path)

# 使用示例
if __name__ == "__main__":
    # tts_instance = TextToSpeech("tts_models/multilingual/multi-dataset/xtts_v2")
    # tts_instance = TextToSpeech()
    tts_instance = TextToSpeech(language="zh-CN")
    # 列出可用模型
    print(tts_instance.list_available_models())
    
    input = '''
        以下是每个缩写的简要解释：

hag: Hanga — 指的是一种语言，主要在巴布亚新几内亚的Hanga地区使用。

hnn: Hanunoo — 指的是菲律宾的一种语言，主要由Hanunoo人使用，属于马来-波利尼西亚语系。

bgc: Haryanvi — 指的是印度哈里亚纳邦的一种方言，属于印地语的一种变体。

had: Hatam — 指的是巴布亚新几内亚的一种语言，主要在Hatam地区使用。

hau: Hausa — 指的是西非的一种语言，广泛用于尼日利亚和尼日尔，是主要的交易语言之一。

hwc: Hawaii Pidgin — 指的是夏威夷的一种克里奥尔语，受英语和夏威夷土著语言影响，常用于当地的日常交流。

hvn: Hawu — 指的是印度尼西亚的一种语言，主要在西努沙登加拉省的Hawu地区使用。

hay: Haya — 指的是坦桑尼亚的一种语言，由Haya人使用，属于尼日尔-刚果语系。
    '''
    # 生成音频并保存到文件
    tts_instance.save_audio_to_file(input, "1tacoutput.wav")
