# Makefile

# 定义变量
PYTHON = python3
PIP = pip3
TEST_DIR = tests
TEST_LLMS = tests/llms
TEST_TASKS = tests/tasks
APP_DIR = agi
TARGET = agi.fastapi_agi:app
IMAGE_GEN = agi.apps.image.fast_api_image:app

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
.PHONY: run
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
	-t guojingneo/agi-fastapi-app:base .

.PHONY: image
image: image_base
	docker build \
	-f ./Dockerfile \
	--build-arg COMMIT_HASH=$$(git rev-parse HEAD) \
	--build-arg BRANCH_NAME=$$(git rev-parse --abbrev-ref HEAD) \
	-t guojingneo/agi-fastapi-app:$$(git rev-parse --short HEAD)-$$(git rev-parse --abbrev-ref HEAD) .

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

run_image:
	$(PYTHON) -m uvicorn $(IMAGE_GEN) --host 0.0.0.0 --port 8001 --reload


.PHONY: image_image_gen
image_image_gen:
	docker build \
	-f ./Dockerfile.image \
	--build-arg COMMIT_HASH=$$(git rev-parse HEAD) \
	--build-arg BRANCH_NAME=$$(git rev-parse --abbrev-ref HEAD) \
	-t guojingneo/agi-fastapi-image:$$(git rev-parse --short HEAD)-$$(git rev-parse --abbrev-ref HEAD) .

.PHONY: image_image_gen3.5
image_image_gen3:
	docker build \
	-f ./Dockerfile.image.3.5 \
	--build-arg COMMIT_HASH=$$(git rev-parse HEAD) \
	--build-arg BRANCH_NAME=$$(git rev-parse --abbrev-ref HEAD) \
	-t guojingneo/agi-fastapi-image:$$(git rev-parse --short HEAD)-$$(git rev-parse --abbrev-ref HEAD) .

.PHONY: image_tts
image_tts:
	docker build \
	-f ./Dockerfile.tts \
	--build-arg COMMIT_HASH=$$(git rev-parse HEAD) \
	--build-arg BRANCH_NAME=$$(git rev-parse --abbrev-ref HEAD) \
	-t guojingneo/agi-fastapi-tts:$$(git rev-parse --short HEAD)-$$(git rev-parse --abbrev-ref HEAD) .

.PHONY: image_tts_base
image_tts_base:
	docker build \
	-f ./Dockerfile.tts.base \
	-t guojingneo/agi-fastapi-tts:base .

.PHONY: image_whisper
image_whisper:
	docker build \
	-f ./Dockerfile.whisper \
	--build-arg COMMIT_HASH=$$(git rev-parse HEAD) \
	--build-arg BRANCH_NAME=$$(git rev-parse --abbrev-ref HEAD) \
	-t guojingneo/agi-fastapi-whisper:$$(git rev-parse --short HEAD)-$$(git rev-parse --abbrev-ref HEAD) .

.PHONY: image_hf
image_hf:
	docker build \
	-f ./Dockerfile.hf \
	--build-arg COMMIT_HASH=$$(git rev-parse HEAD) \
	--build-arg BRANCH_NAME=$$(git rev-parse --abbrev-ref HEAD) \
	-t guojingneo/agi-fastapi-hf:$$(git rev-parse --short HEAD)-$$(git rev-parse --abbrev-ref HEAD) .