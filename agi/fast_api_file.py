from fastapi import HTTPException, File, UploadFile, Response,Form,APIRouter
from fastapi.responses import Response
from fastapi import Request
from agi.tasks.task_factory import TaskFactory
from agi.utils.file_storage import default_file_service,default_storage
from agi.utils.common import path_to_preview_url
from typing import Optional
from agi.config import log

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

router_file = APIRouter(prefix="/v1")

@router_file.get("/files")
async def list_files():
    return {"files": await default_storage.list_files()}

@router_file.post("/files")
async def save_file(
    file: UploadFile = File(...),  # 接收上传的文件
    user_id: str = Form(...),
    collection_name: Optional[str] = Form("default")
    ):
    # 获取文件类型
    file_type = file.content_type
    if file_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(status_code=400, detail=f"不支持的文件类型: {file_type}")

    
    unique_name = await default_file_service.save_file(file, file.filename)
    
    if collection_name and not file_type.startswith("image/") and not file_type.startswith("audio/"):
        kmanager = TaskFactory.get_knowledge_manager()
        param = {"filename" : file.filename}
        await kmanager.store(collection_name,default_storage.to_local_path(unique_name),tenant=user_id,**param)
        
    return {
        "original_filename": file.filename,
        "saved_filename": unique_name,
        "file_type": file_type,
        "message": "文件上传成功"
    }


@router_file.get("/files/{file_name}")
async def download_file(file_name: str, request: Request):
    def is_browser(user_agent: str) -> bool:
        # 简单判断是否为常见浏览器
        browsers = ["Chrome", "Firefox", "Safari", "Edge", "Opera"]
        return any(browser in user_agent for browser in browsers)
    
    # 获取User-Agent
    user_agent = request.headers.get("user-agent", "")
    disposition_type = "inline" if is_browser(user_agent) else "attachment"

    try:
        file_bytes = await default_storage.load(file_name)
        mime_type = default_storage.get_mime_type(file_name)
        return Response(file_bytes, media_type=mime_type, headers={
            "Content-Disposition": f"{disposition_type}; filename={file_name}"
        })
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="File not found")

@router_file.delete("/files/{file_name}")
async def delete_file(file_name: str):
    try:
        await default_storage.delete(file_name)
        return {"message": "File deleted successfully"}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="File not found")



