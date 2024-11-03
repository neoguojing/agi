# Makefile

# 定义变量
PYTHON = python3
PIP = pip3
TEST_DIR = tests
APP_DIR = app
TARGET = app.main:app

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
	$(PYTHON) -m pytest -s $(TEST_DIR)

# 启动 FastAPI 服务
.PHONY: run
run:
	$(PYTHON) -m uvicorn $(TARGET) --host 0.0.0.0 --port 8000 --reload

# 清理目标
.PHONY: clean
clean:
	rm -rf dist/ build/ *.egg-info
