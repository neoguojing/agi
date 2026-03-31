# =========================================================
# CosyVoice3 Builder (Model External Mount Version)
# =========================================================

FROM nvcr.io/nvidia/tritonserver:25.06-py3

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# -------------------------
# 系统依赖
# -------------------------
RUN apt-get update && apt-get install -y \
    git \
    git-lfs \
    curl \
    wget \
    ffmpeg \
    sox \
    libgl1 \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# -------------------------
# Python 依赖
# -------------------------
RUN pip3 install --upgrade pip

RUN pip3 install \
    huggingface_hub \
    transformers \
    sentencepiece \
    accelerate \
    safetensors \
    datasets \
    x_transformers \
    s3tokenizer \
    librosa \
    soundfile \
    scipy \
    numpy \
    diffusers \
    mpi4py

# -------------------------
# Git LFS
# -------------------------
RUN git lfs install

# -------------------------
# 工作目录
# -------------------------
WORKDIR /workspace

# -------------------------
# 仅 clone 代码（不下载模型）
# -------------------------
RUN git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git

WORKDIR /workspace/CosyVoice
RUN git submodule update --init --recursive

# -------------------------
# 环境变量
# -------------------------
ENV COSYVOICE_PATH=/workspace/CosyVoice
ENV PYTHONPATH=${COSYVOICE_PATH}:$PYTHONPATH
ENV PYTHONPATH=${COSYVOICE_PATH}/third_party/Matcha-TTS:$PYTHONPATH

# -------------------------
# 预留挂载目录（关键）
# -------------------------
RUN mkdir -p /models/hf_llm \
             /models/cosyvoice \
             /models/trt_weights \
             /models/trt_engines \
             /models/model_repo

# -------------------------
# 默认工作路径
# -------------------------
WORKDIR /workspace/CosyVoice/runtime/triton_trtllm

CMD ["/bin/bash"]