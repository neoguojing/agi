# =========================================
# CosyVoice3 TRT Build Makefile
# =========================================

IMAGE_NAME=triton:25.06
CONTAINER_NAME=cosyvoice-build

# 挂载目录（宿主机）
MODEL_DIR=/data/model

# 容器内路径（固定）
CONTAINER_MODEL_DIR=/models
WORKDIR=/workspace/CosyVoice/runtime/triton_trtllm

# 默认参数
TRT_DTYPE?=float16
MAX_BATCH_SIZE?=8
MAX_NUM_TOKENS?=4096

# -----------------------------------------
# 构建镜像
# -----------------------------------------
build-image:
	docker build -t $(IMAGE_NAME) -f Dockerfile .

# -----------------------------------------
# 进入容器（调试用）
# -----------------------------------------
shell:
	docker run --gpus all -it --rm \
		-v $(MODEL_DIR):$(CONTAINER_MODEL_DIR) \
		$(IMAGE_NAME) bash

# -----------------------------------------
# Stage 1: 转换 checkpoint → TRT weights
# -----------------------------------------
convert:
	docker run --gpus all --rm \
		-v $(MODEL_DIR):$(CONTAINER_MODEL_DIR) \
		$(IMAGE_NAME) bash -c "\
		cd $(WORKDIR) && \
		python3 scripts/convert_checkpoint.py \
			--model_dir $(CONTAINER_MODEL_DIR)/hf_llm \
			--output_dir $(CONTAINER_MODEL_DIR)/trt_weights \
			--dtype $(TRT_DTYPE)"

# -----------------------------------------
# Stage 1: 构建 TRT engine
# -----------------------------------------
build-engine:
	docker run --gpus all --rm \
		-v $(MODEL_DIR):$(CONTAINER_MODEL_DIR) \
		$(IMAGE_NAME) bash -c "\
		cd $(WORKDIR) && \
		trtllm-build \
			--checkpoint_dir $(CONTAINER_MODEL_DIR)/trt_weights \
			--output_dir $(CONTAINER_MODEL_DIR)/trt_engines \
			--max_batch_size $(MAX_BATCH_SIZE) \
			--max_num_tokens $(MAX_NUM_TOKENS) \
			--gemm_plugin $(TRT_DTYPE)"

# -----------------------------------------
# 测试 engine
# -----------------------------------------
test-engine:
	docker run --gpus all --rm \
		-v $(MODEL_DIR):$(CONTAINER_MODEL_DIR) \
		$(IMAGE_NAME) bash -c "\
		cd $(WORKDIR) && \
		python3 scripts/test_llm.py \
			--input_text '你好，请问你叫什么？' \
			--tokenizer_dir $(CONTAINER_MODEL_DIR)/hf_llm \
			--engine_dir $(CONTAINER_MODEL_DIR)/trt_engines"

# -----------------------------------------
# Stage 2: 构建 Triton model repo
# -----------------------------------------
build-repo:
	docker run --gpus all --rm \
		-v $(MODEL_DIR):$(CONTAINER_MODEL_DIR) \
		$(IMAGE_NAME) bash -c "\
		cd $(WORKDIR) && \
		rm -rf $(CONTAINER_MODEL_DIR)/model_repo && \
		mkdir -p $(CONTAINER_MODEL_DIR)/model_repo && \
		cp -r model_repo_cosyvoice3/* $(CONTAINER_MODEL_DIR)/model_repo/ && \
		python3 scripts/fill_template.py \
			-i $(CONTAINER_MODEL_DIR)/model_repo/cosyvoice3/config.pbtxt \
			model_dir:$(CONTAINER_MODEL_DIR)/cosyvoice,bls_instance_num:4,llm_tokenizer_dir:$(CONTAINER_MODEL_DIR)/hf_llm,triton_max_batch_size:1,decoupled_mode:True,max_queue_delay_microseconds:0"

# -----------------------------------------
# 一键执行全部
# -----------------------------------------
all: convert build-engine test-engine build-repo

# -----------------------------------------
# 清理
# -----------------------------------------
clean:
	rm -rf models/trt_weights/*
	rm -rf models/trt_engines/*
	rm -rf models/model_repo/*