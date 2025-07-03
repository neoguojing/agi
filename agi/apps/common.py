import os
import urllib.parse

from agi.config import API_KEY,BASE_URL,CACHE_DIR
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi import Depends, HTTPException

# 认证配置
security = HTTPBearer()

# 认证函数
async def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return credentials.credentials


def path_to_preview_url(file_path: str, base_url: str = BASE_URL) -> str:
        """
        将文件路径转换为图片预览 URL。
        
        Args:
            file_path (str): 服务器上的文件路径，例如 "uploads/picture.jpg"
            base_url (str): 服务器基地址，默认 "http://localhost:8000"
        
        Returns:
            str: 可用于预览的 URL，例如 "http://localhost:8000/files/picture.jpg"
        
        Raises:
            ValueError: 如果文件路径不在上传目录内
        """
        # 确保文件路径在 CACHE_DIR 内，防止目录遍历
        if not os.path.realpath(file_path).startswith(os.path.realpath(CACHE_DIR)):
            raise ValueError("File path is outside the upload directory")
        
        # 获取相对于 UPLOAD_DIR 的文件名
        file_name = os.path.basename(file_path)
        
        # 构建预览 URL
        preview_url = f"{base_url}/v1/files/{urllib.parse.quote(file_name)}"
        return preview_url