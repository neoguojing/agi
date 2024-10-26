from faster_whisper import WhisperModel
import os
from agi.config import (
    WHISPER_MODEL_DIR as model_root,
    WHISPER_MODEL
)
from agi.llms.base import CustomerLLM

class Speech2Text(CustomerLLM):
    def __init__(self, model_size: str = os.path.join(model_root,"wisper-v3-turbo-c2"), 
                 device: str = "cuda", compute_type: str = "float16"):
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)

    def transcribe(self, audio_file: str, beam_size: int = 5):
        segments, info = self.model.transcribe(audio_file, beam_size=beam_size)
        return segments, info

    def print(self, segments, info):
        print("Detected language '%s' with probability %f" % (info.language, info.language_probability))
        for segment in segments:
            print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))

# 使用示例
if __name__ == "__main__":
    # transcriber = AudioTranscriber(model_size="large-v3", device="cuda", compute_type="float16")
    transcriber = Speech2Text(device="cpu", compute_type="int8")
    
    # 转录音频
    import time
    start_time = time.time() 
    segments, info = transcriber.transcribe("output.wav")
    transcriber.print(segments, info)
    end_time = time.time()     # 记录结束时间
    execution_time = end_time - start_time  # 计算执行时间
    print(f"Execution time: {execution_time:.5f} seconds")
