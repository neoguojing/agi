import os
import uuid
import shutil
from fastapi.testclient import TestClient
from fastapi import HTTPException
import pytest
from agi.fastapi_agi import app
from agi.config import CACHE_DIR
from agi.fast_api_file import ALLOWED_MIME_TYPES 
import logging
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
# 测试用文件和路径
TEST_FILE_PATH = "testfile.txt"
TEST_COLLECTION_NAME = "test_collection"

# 创建临时目录来模拟文件上传
@pytest.fixture(scope="module")
def setup_module():
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)
    yield
    # 清理临时目录
    for file in os.listdir(CACHE_DIR):
        file_path = os.path.join(CACHE_DIR, file)
        if os.path.isfile(file_path):
            os.remove(file_path)

# 创建测试客户端
client = TestClient(app)

# 测试列出文件 API
def test_list_files(setup_module):
    response = client.get("/v1/files")
    assert response.status_code == 200
    assert isinstance(response.json(), dict)
    assert "files" in response.json()

# 测试文件上传 API
def test_save_file(setup_module):
    with open(TEST_FILE_PATH, "w") as f:
        f.write("This is a test file.")

    with open(TEST_FILE_PATH, "rb") as f:
        response = client.post(
            "/v1/files",
            files={"file": ("testfile.txt", f, "text/plain")},
            data={"collection_name": TEST_COLLECTION_NAME,"user_id":"test"}
        )
    assert response.status_code == 200
    response_json = response.json()
    assert response_json["original_filename"] == "testfile.txt"
    assert "saved_filename" in response_json
    assert response_json["file_type"] == "text/plain"

    # 清理测试文件
    os.remove(TEST_FILE_PATH)

# 测试下载文件 API
def test_download_file(setup_module):
    # 上传文件
    with open(TEST_FILE_PATH, "w") as f:
        f.write("This is a test file.")
    
    with open(TEST_FILE_PATH, "rb") as f:
        response = client.post(
            "/v1/files",
            files={"file": ("testfile.txt", f, "text/plain")},
            data={"collection_name": TEST_COLLECTION_NAME}
        )
    saved_filename = response.json()["saved_filename"]
    log.debug(response.json())
    # 下载文件
    response = client.get(f"/v1/files/{saved_filename}")
    assert response.status_code == 200
    assert response.headers["Content-Disposition"].startswith("attachment")
    assert response.headers["Content-Type"] == "text/plain; charset=utf-8"
    assert response.content == b"This is a test file."
    
    # 清理测试文件
    os.remove(TEST_FILE_PATH)

# 测试删除文件 API
def test_delete_file(setup_module):
    # 上传文件
    with open(TEST_FILE_PATH, "w") as f:
        f.write("This is a test file.")
    
    with open(TEST_FILE_PATH, "rb") as f:
        response = client.post(
            "/v1/files",
            files={"file": ("testfile.txt", f, "text/plain")},
            data={"collection_name": TEST_COLLECTION_NAME}
        )
    saved_filename = response.json()["saved_filename"]
    log.debug(response.json())
    # 删除文件
    response = client.delete(f"/v1/files/{saved_filename}")
    assert response.status_code == 200
    assert response.json()["message"] == "File deleted successfully"
    
    # 尝试再次下载已删除的文件
    response = client.get(f"/v1/files/{saved_filename}")
    assert response.status_code == 200
    assert response.json() == {"error": "File not found"}

# 测试不支持的文件类型
def test_unsupported_file_type(setup_module):
    with open(TEST_FILE_PATH, "w") as f:
        f.write("This is a test file.")
    
    with open(TEST_FILE_PATH, "rb") as f:
        response = client.post(
            "/v1/files",
            files={"file": ("testfile.xyz", f, "application/xyz")},
            data={"collection_name": TEST_COLLECTION_NAME}
        )
    
    assert response.status_code == 400
    assert "不支持的文件类型" in response.json()["detail"]
    
    # 清理测试文件
    os.remove(TEST_FILE_PATH)
