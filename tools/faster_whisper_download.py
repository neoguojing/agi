from faster_whisper import WhisperModel
# CPU: 优化 8-bit
model_cpu = WhisperModel("base", device="cpu", compute_type="int8",download_root="./")

# GPU: 8-bit 编码 + 混合精度解码
model = WhisperModel("large-v3", device="cuda", compute_type="int8",download_root="./")

