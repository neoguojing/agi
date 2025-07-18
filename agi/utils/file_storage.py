import fsspec
import os
import mimetypes
import time
from typing import List, Dict
import uuid
from agi.config import FILE_STORAGE_URL,BASE_URL
import urllib.parse

class StorageError(Exception):
    """通用存储错误基类"""
    pass

class FileNotFound(StorageError):
    """文件不存在"""
    pass

class InvalidFilename(StorageError):
    """非法文件名（可能是路径穿越）"""
    pass

class FSSpecStorage:
    def __init__(self, base_path: str):
        self.base_path = base_path.rstrip("/") + "/"
        self.fs, self.base = fsspec.core.url_to_fs(self.base_path)
        print(self.base_path,self.fs.protocol,self.base)

    def _full_path(self, filename: str) -> str:
        if not filename or ".." in filename or filename.startswith("/"):
            raise InvalidFilename(f"Invalid filename: {filename}")
        return os.path.join(self.base,filename)

    def to_local_path(self, filename: str) -> str:
        """
        如果当前存储后端是 file://，返回本地路径；否则抛出异常。
        """
        if "file" not in self.fs.protocol:
            raise RuntimeError("Only file:// protocol supports local path resolution")

        full_url = self._full_path(filename)
        parsed = urllib.parse.urlparse(full_url)
        return urllib.parse.unquote(parsed.path)
    
    def path_to_preview_url(self,filename: str, base_url: str = BASE_URL) -> str:
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
        path = self._full_path(filename)
        if not self.fs.exists(path):
            raise FileNotFound(f"File not found: {filename}")
        # 构建预览 URL
        preview_url = f"{base_url}/v1/files/{urllib.parse.quote(filename)}"
        return preview_url
    
    async def save(self, file_obj, filename: str) -> str:
        try:
            with self.fs.open(self._full_path(filename), 'wb') as f:
                f.write(await file_obj.read())
        except Exception as e:
            raise StorageError(f"Failed to save file: {e}")
        return filename

    async def load(self, filename: str) -> bytes:
        path = self._full_path(filename)
        if not self.fs.exists(path):
            raise FileNotFound(f"File not found: {filename}")
        try:
            with self.fs.open(path, 'rb') as f:
                return f.read()
        except Exception as e:
            raise StorageError(f"Failed to load file: {e}")

    async def delete(self, filename: str):
        path = self._full_path(filename)
        if not self.fs.exists(path):
            raise FileNotFound(f"File not found: {filename}")
        try:
            self.fs.rm(path)
        except Exception as e:
            raise StorageError(f"Failed to delete file: {e}")

    async def list_files(self) -> List[Dict[str, str]]:
        try:
            files = []
            for f in self.fs.ls(self.base, detail=True):
                files.append({
                    "name": os.path.basename(f["name"]),
                    "size": f.get("size", 0),
                    "last_modified": time.ctime(f.get("mtime", time.time()))
                })
            return files
        except Exception as e:
            raise StorageError(f"Failed to list files: {e}")

    def get_mime_type(self, filename: str) -> str:
        return mimetypes.guess_type(filename)[0] or "application/octet-stream"

class FileService:
    def __init__(self, storage: FSSpecStorage):
        self.storage = storage

    def generate_unique_filename(self, original_name: str) -> str:
        ext = os.path.splitext(original_name)[1]
        return f"{uuid.uuid4().hex}{ext}"

    async def save_file(self, file, original_name: str):
        unique_name = self.generate_unique_filename(original_name)
        await self.storage.save(file, unique_name)
        return unique_name

default_storage = FSSpecStorage(FILE_STORAGE_URL)
default_file_service = FileService(default_storage)