# 使用官方 Python 3.11 slim 作为基础镜像
FROM guojingneo/agi-fastapi-app:b46e693-main

# 设置工作目录
WORKDIR /agi

# 将应用代码拷贝到容器中
COPY . .

# 暴露应用运行的端口（默认 FastAPI 使用 8000）
EXPOSE 8000

# 启动命令（假设你的 FastAPI 应用在 main.py 中，并且实例名称为 app）
CMD ["uvicorn", "agi.fastapi_agi:app", "--host", "0.0.0.0", "--port", "8000"]
