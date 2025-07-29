# Makefile

# 定义变量
PYTHON = python3
PIP = pip3
TEST_DIR = tests
TEST_LLMS = tests/llms
TEST_TASKS = tests/tasks
APP_DIR = agi
TARGET = agi.fastapi_agi:app
IMAGE_GEN_TAGET = agi.apps.image.fast_api_image:app
TTS_TARGET = agi.apps.tts.fast_api_audio:app
WHISPER_TARGET = agi.apps.whisper.fast_api_whisper:app
HUGFACE_TARGET = agi.apps.multimodal.fast_api_multimodal:app
EMBD_TARGET = agi.apps.embding.fast_api_embding:app


REGISTRY = "docker.io"
# 可选值: tts 或 cosyvoice
TTS_TYPE ?= cosyvoice
# 可选值sdxl
IMAGE_GEN_VERION ?= "sd3.5"
# 默认目标
.PHONY: all
all: install test

# 安装依赖
.PHONY: install
install:
	$(PIP) install -r requirements.txt
# 打包
.PHONY: package
package:
	$(PYTHON) setup.py sdist bdist_wheel

# 运行单元测试
.PHONY: test
test:
	@$(PYTHON) -m pytest -s $(TEST_DIR)

.PHONY: test_llms
test_llms:
	@$(PYTHON) -m pytest -s $(TEST_LLMS)

.PHONY: test_tasks
test_tasks:
	@$(PYTHON) -m pytest -s $(TEST_TASKS)

.PHONY: test_api
test_api:
	@$(PYTHON) -m pytest -s tests/fastapi_agi_test.py

# 启动 FastAPI 服务
.PHONY: run run_image
run_image:
	$(PYTHON) -m uvicorn $(IMAGE_GEN_TAGET) --host 0.0.0.0 --port 8001 --reload

run_tts:
	$(PYTHON) -m uvicorn $(TTS_TARGET) --host 0.0.0.0 --port 8002 --reload

run_whisper:
	$(PYTHON) -m uvicorn $(WHISPER_TARGET) --host 0.0.0.0 --port 8003 --reload

run_hugface:
	$(PYTHON) -m uvicorn $(HUGFACE_TARGET) --host 0.0.0.0 --port 8005 --reload

run_embd:
	$(PYTHON) -m uvicorn $(EMBD_TARGET) --host 0.0.0.0 --port 8006 --reload
run:
	python -m playwright install chromium
	$(PYTHON) -m uvicorn $(TARGET) --host 0.0.0.0 --port 8000 --reload

# 清理目标
.PHONY: clean
clean:
	rm -rf dist/ build/ *.egg-info

.PHONY: image_base
image_base:
	docker build \
	-f ./Dockerfile.base \
	-t $(REGISTRY)/guojingneo/agi-fastapi-app:base .

.PHONY: image
image:
	docker build \
	-f ./Dockerfile \
	--build-arg COMMIT_HASH=$$(git rev-parse HEAD) \
	--build-arg BRANCH_NAME=$$(git rev-parse --abbrev-ref HEAD) \
	-t $(REGISTRY)/guojingneo/agi-fastapi-app:$$(git rev-parse --short HEAD)-$$(git rev-parse --abbrev-ref HEAD) .

.PHONY: runi
runi:
	docker run -d -p 8000:8000 guojingneo/agi-fastapi-app:$$(git rev-parse --short HEAD)-$$(git rev-parse --abbrev-ref HEAD)

.PHONY: models
MODEL_DIR := ./modelfiles  # 指定存放 Modelfile 的目录
models:
	@for file in $(shell find $(MODEL_DIR) -type f); do \
	    model_name=$$(basename $$file | sed 's/Modelfile-//'); \
	    echo "Creating model: $$model_name from $$file"; \
	    ollama create $$model_name -f $$file; \
	done


.PHONY: image_image_gen
image_image_gen:
	docker build \
	-f ./Dockerfile.${IMAGE_GEN_VERION}.image \
	--build-arg COMMIT_HASH=$$(git rev-parse HEAD) \
	--build-arg BRANCH_NAME=$$(git rev-parse --abbrev-ref HEAD) \
	-t $(REGISTRY)/guojingneo/agi-fastapi-image:$$(git rev-parse --short HEAD)-$$(git rev-parse --abbrev-ref HEAD) .

.PHONY: image_tts
image_tts:
	docker build \
	-f ./Dockerfile.tts \
	--build-arg COMMIT_HASH=$$(git rev-parse HEAD) \
	--build-arg BRANCH_NAME=$$(git rev-parse --abbrev-ref HEAD) \
	-t $(REGISTRY)/guojingneo/agi-fastapi-tts:$$(git rev-parse --short HEAD)-$$(git rev-parse --abbrev-ref HEAD) .

.PHONY: image_tts_base
image_tts_base:
	docker build \
		-f ./Dockerfile.${TTS_TYPE}.base \
		-t $(REGISTRY)/guojingneo/agi-fastapi-tts:base .

.PHONY: image_whisper
image_whisper:
	docker build \
	-f ./Dockerfile.whisper \
	--build-arg COMMIT_HASH=$$(git rev-parse HEAD) \
	--build-arg BRANCH_NAME=$$(git rev-parse --abbrev-ref HEAD) \
	-t $(REGISTRY)/guojingneo/agi-fastapi-whisper:$$(git rev-parse --short HEAD)-$$(git rev-parse --abbrev-ref HEAD) .

.PHONY: image_hf
image_hf:
	docker build \
	-f ./Dockerfile.hf \
	--build-arg COMMIT_HASH=$$(git rev-parse HEAD) \
	--build-arg BRANCH_NAME=$$(git rev-parse --abbrev-ref HEAD) \
	-t $(REGISTRY)/guojingneo/agi-fastapi-hf:$$(git rev-parse --short HEAD)-$$(git rev-parse --abbrev-ref HEAD) .

.PHONY: image_embd
image_embd:
	docker build \
	-f ./Dockerfile.embd \
	--build-arg COMMIT_HASH=$$(git rev-parse HEAD) \
	--build-arg BRANCH_NAME=$$(git rev-parse --abbrev-ref HEAD) \
	-t $(REGISTRY)/guojingneo/agi-fastapi-embd:$$(git rev-parse --short HEAD)-$$(git rev-parse --abbrev-ref HEAD) .