# =========================================================
# CosyVoice3 Builder (Model External Mount Version)
# =========================================================

FROM nvcr.io/nvidia/tritonserver:25.06-py3

# -------------------------
# 基础环境变量
# -------------------------
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
ENV PYTHONUNBUFFERED=1

# -------------------------
# 修复系统依赖 (重点修复处)
# -------------------------
RUN export DEBIAN_FRONTEND=noninteractive && \
    # 彻底清除 Triton 镜像自带的干扰钩子
    rm -rf /etc/apt/apt.conf.d/* && \
    echo 'Binary::apt::APT::Keep-Downloaded-Packages "true";' > /etc/apt/apt.conf.d/01build-fix && \
    # 更新并安装基础工具
    apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    wget \
    ffmpeg \
    sox \
    libgl1 \
    build-essential \
    cmake \
    ca-certificates \
    tzdata && \
    # 清理缓存
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# -------------------------
# Python 依赖优化
# -------------------------
ENV PIP_PROGRESS_BAR=off

# 分层安装以利用缓存，CosyVoice 核心依赖
RUN pip3 install --no-cache-dir \
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
# 工作目录与代码 Clone
# -------------------------
WORKDIR /workspace

COPY CosyVoice /workspace/CosyVoice

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