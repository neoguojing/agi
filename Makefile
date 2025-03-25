# Makefile

# 定义变量
PYTHON = python3
PIP = pip3
TEST_DIR = tests
TEST_LLMS = tests/llms
TEST_TASKS = tests/tasks
APP_DIR = agi
TARGET = agi.fastapi_agi:app

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
	$(PYTHON) -m uvicorn $(TARGET) --host 0.0.0.0 --port 8000 --reload

# 清理目标
.PHONY: clean
clean:
	rm -rf dist/ build/ *.egg-info

.PHONY: image
image:
	docker build \
	--build-arg COMMIT_HASH=$(git rev-parse HEAD) \
	--build-arg BRANCH_NAME=$(git rev-parse --abbrev-ref HEAD) \
	-t agi-fastapi-app:$(git rev-parse --short HEAD)-$(git rev-parse --abbrev-ref HEAD) .


