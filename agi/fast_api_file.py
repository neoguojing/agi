import os
import time
import mimetypes
import urllib.parse
from fastapi import FastAPI, HTTPException, File, UploadFile, Response,Form
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from agi.fastapi_agi import app
from agi.config import CACHE_DIR
from agi.tasks.task_factory import TaskFactory,TASK_DOC_DB
import shutil
import uuid

# 允许的 MIME 类型
ALLOWED_MIME_TYPES = {
    "image/jpeg", "image/png", "application/pdf", "text/plain",
    "application/msword",  # .doc
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",  # .docx
    "application/vnd.ms-excel",  # .xls
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",  # .xlsx
    "application/vnd.ms-powerpoint",  # .ppt
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",  # .pptx
    "application/json",  # .json
    "text/csv",  # .csv
    "text/markdown",  # .md
    "text/html",  # .html, .htm
    "application/epub+zip",  # .epub
    "message/rfc822",  # .msg (Outlook 邮件)
}


@ app.get("/files")
async def list_files():
    if not os.path.exists(CACHE_DIR):
        raise HTTPException(status_code=500, detail="Upload directory not found")
    files = []
    with os.scandir(CACHE_DIR) as entries:
        for entry in entries:
            if entry.is_file():
                file_info = {
                    "name": entry.name,
                    "size": entry.stat().st_size,
                    "last_modified": time.ctime(entry.stat().st_mtime)
                }
                files.append(file_info)
    return {"files": files}

def get_file_type(file: UploadFile):
    """根据文件扩展名或 MIME 类型检测文件类型"""
    mime_type, _ = mimetypes.guess_type(file.filename)
    return mime_type or "unknown"

def generate_unique_filename(filename: str):
    """生成不重复的文件名，保留原始扩展名"""
    ext = os.path.splitext(filename)[1]  # 获取文件扩展名
    unique_id = uuid.uuid4().hex  # 生成唯一 ID
    return f"{unique_id}{ext}"

@app.post("/files")
async def save_file(
    file: UploadFile = File(...),  # 接收上传的文件
    collection_name: str = Form(...),  # 接收 collection_name 作为表单字段
    ):
    # 获取文件类型
    file_type = get_file_type(file)

    if file_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(status_code=400, detail=f"不支持的文件类型: {file_type}")

    # 生成唯一文件名，防止重名
    unique_filename = generate_unique_filename(file.filename)
    file_path = os.path.join(CACHE_DIR,"upload", unique_filename)
    # 确保目标目录存在，如果不存在则创建
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    # 保存文件
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    
    if collection_name and not file_type.startswith("image/") and not file_type.startswith("audio/"):
        kmanager = TaskFactory.create_task(TASK_DOC_DB)
        param = {"filename" : file.filename}
        kmanager.store(collection_name,file_path,**param)
        
    return {"original_filename": file.filename, "saved_filename": unique_filename, "file_type": file_type, "message": "文件上传成功"}


@ app.get("/files/{file_name}")
async def download_file(file_name: str):
    file_path = os.path.join(CACHE_DIR,"upload", file_name)
    print("*******************",file_path)
    if not os.path.realpath(file_path).startswith(os.path.realpath(CACHE_DIR)):
        raise HTTPException(status_code=400, detail="Invalid file path")
    if not os.path.exists(file_path):
        return {"error": "File not found"}
    content_type = mimetypes.guess_type(file_name)[0] or "application/octet-stream"
    headers = {}
    if content_type.startswith("image/") or content_type.startswith("audio/"):
        headers["Content-Disposition"] = f"inline; filename={urllib.parse.quote(file_name)}"
    else:
        headers["Content-Disposition"] = f"attachment; filename={urllib.parse.quote(file_name)}"
    with open(file_path, "rb") as f:
        return Response(f.read(), media_type=content_type, headers=headers)

@ app.delete("/files/{file_name}")
async def delete_file(file_name: str):
    file_path = os.path.join(CACHE_DIR,"upload", file_name)
    if not os.path.realpath(file_path).startswith(os.path.realpath(CACHE_DIR)):
        raise HTTPException(status_code=400, detail="Invalid file path")
    if not os.path.exists(file_path):
        return {"error": "File not found"}
    try:
        os.remove(file_path)
    except OSError as e:
        raise HTTPException(status_code=500, detail=f"Error deleting file: {str(e)}")
    return {"message": "File deleted successfully"}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

