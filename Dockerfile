# 使用官方 Python 3.11 slim 作为基础镜像
# FROM python:3.11-slim
FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-devel

# 设置工作目录
WORKDIR /agi

# 将 requirements.txt 拷贝到容器中，并安装 Python 依赖
COPY requirements.txt .
COPY requirements/ ./requirements/
COPY depend/ ./depend/

# 更新 apt-get 并安装必要的系统依赖（可根据需要调整）
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git \
    libglib2.0-0 \
    libnss3 \
    libnspr4 \
    libdbus-1-3 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libx11-6 \
    libxcomposite1 \
    libxdamage1 \
    libxext6 \
    libxfixes3 \
    libxrandr2 \
    libgbm1 \
    libxcb1 \
    libxkbcommon0 \
    libpango-1.0-0 \
    libcairo2 \
    libasound2 \
    libatspi2.0-0 \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --upgrade pip && pip install packaging && pip install -r requirements.txt && rm -rf /root/.cache
RUN python -m playwright install chromium && rm -rf /root/.cache

# 将应用代码拷贝到容器中
COPY . .

# 暴露应用运行的端口（默认 FastAPI 使用 8000）
EXPOSE 8000

# 启动命令（假设你的 FastAPI 应用在 main.py 中，并且实例名称为 app）
CMD ["uvicorn", "agi.fastapi_agi:app", "--host", "0.0.0.0", "--port", "8000"]
