from fastapi import FastAPI, File, UploadFile, Form, HTTPException,Depends
import os
from agi.apps.common import verify_api_key
from agi.apps.whisper.speech2text import Speech2Text
from agi.config import FILE_UPLOAD_PATH

app = FastAPI(title="Custom Whisper API")

# 加载模型（你可以选择 tiny / base / medium / large）
model = Speech2Text() 

@app.post("/v1/audio/transcriptions")
async def transcribe_audio(
    file: UploadFile = File(...),
    model_name: str = Form(...),  # 必须是 "whisper-1"，但你可以忽略它
    prompt: str = Form(None),
    language: str = Form(None),
    response_format: str = Form("json"),
    temperature: float = Form(0.0),
    api_key: str = Depends(verify_api_key)
):
    if not file.filename.endswith((".mp3", ".wav", ".m4a", ".ogg", ".webm")):
        raise HTTPException(status_code=400, detail="Unsupported audio format.")

    # 保存音频到临时文件
    file_path = os.path.join(FILE_UPLOAD_PATH,file.filename)

    with open(file_path, "wb") as f:
        f.write(await file.read())

    # 转录
    text, info = model.invoke(file_path)


    # 返回响应
    if response_format == "json":
        return {"text": text}
    elif response_format == "text":
        return text
    elif response_format == "verbose_json":
        return {
            "text": text,
            "language": info.language
        }
    else:
        raise HTTPException(status_code=400, detail="Unsupported response_format.")
