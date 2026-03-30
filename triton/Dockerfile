FROM nvcr.io/nvidia/tritonserver:25.06-py3

# 不安装额外依赖，保持最小化
WORKDIR /workspace

# 只复制启动脚本
COPY start.sh /workspace/start.sh
RUN chmod +x /workspace/start.sh

ENV CUDA_VISIBLE_DEVICES=0

CMD ["/workspace/start.sh"]