from huggingface_hub import snapshot_download
snapshot_download('FunAudioLLM/Fun-CosyVoice3-0.5B-2512', 
                  local_dir='/data/model/Fun-CosyVoice3-0.5B',
                  resume_download=True)
# -----------------------------
# 下载 LLM 模型（文本 → audio token）
# -----------------------------
snapshot_download(
    'yuekai/Fun-CosyVoice3-0.5B-2512-LLM-HF',
    local_dir='/data/model/hf_cosyvoice3_llm',
    resume_download=True
)

# -----------------------------
# 下载 FP16 ONNX 模型（token → wav）
# -----------------------------
snapshot_download(
    'yuekai/Fun-CosyVoice3-0.5B-2512-FP16-ONNX',
    local_dir='/data/model/Fun-CosyVoice3-0.5B-ONNX',
    resume_download=True
)
