# 使用官方 Python 3.11 slim 作为基础镜像
FROM guojingneo/agi-fastapi-app:base

# 设置工作目录
WORKDIR /agi

# 将 requirements.txt 拷贝到容器中，并安装 Python 依赖
COPY requirements/ ./requirements/

RUN pip install --no-cache-dir -r ./requirements/langchain.txt && rm -rf /root/.cache /tmp/* /var/tmp/*
RUN pip install --no-cache-dir -r ./requirements/common.txt && rm -rf /root/.cache /tmp/* /var/tmp/*
RUN pip install --no-cache-dir -r ./requirements/extra.txt && rm -rf /root/.cache /tmp/* /var/tmp/*
RUN pip install --no-cache-dir -r ./requirements/rag.txt && rm -rf /root/.cache /tmp/* /var/tmp/*

RUN python -m nltk.downloader stopwords punkt
RUN python -m spacy download en_core_web_sm
# 将应用代码拷贝到容器中
COPY agi/ /agi/agi/
RUN mkdir -p /agi/asset
COPY asset/stopwords.txt /agi/asset/stopwords.txt

# 暴露应用运行的端口（默认 FastAPI 使用 8000）
EXPOSE 8000

# 启动命令（假设你的 FastAPI 应用在 main.py 中，并且实例名称为 app）
CMD ["uvicorn", "agi.fastapi_agi:app", "--host", "0.0.0.0", "--port", "8000"]
