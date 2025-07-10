# 使用官方 Python 3.11 slim 作为基础镜像
FROM agi-fastapi-app:base

# 设置工作目录
WORKDIR /agi

# 将 requirements.txt 拷贝到容器中，并安装 Python 依赖
COPY requirements/ ./requirements/

RUN pip install --no-cache-dir -r ./requirements/langchain.txt 
RUN pip install --no-cache-dir -r ./requirements/common.txt 
RUN pip install --no-cache-dir -r ./requirements/extra.txt 
RUN rm -rf /root/.cache /tmp/* /var/tmp/*

# 将应用代码拷贝到容器中
COPY . .

# 暴露应用运行的端口（默认 FastAPI 使用 8000）
EXPOSE 8000

# 启动命令（假设你的 FastAPI 应用在 main.py 中，并且实例名称为 app）
CMD ["uvicorn", "agi.fastapi_agi:app", "--host", "0.0.0.0", "--port", "8000"]
