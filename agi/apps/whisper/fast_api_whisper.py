from fastapi import FastAPI, File, UploadFile, Form, HTTPException,Depends
import os
from agi.apps.common import verify_api_key
from agi.apps.whisper.speech2text import Speech2Text
from agi.config import FILE_UPLOAD_PATH,log
from pydub import AudioSegment
import uuid
import traceback

app = FastAPI(title="Custom Whisper API")

# 加载模型（你可以选择 tiny / base / medium / large）
model = Speech2Text() 

@app.post("/v1/audio/transcriptions")
async def transcribe_audio(
    file: UploadFile = File(...),
    model: str = Form("whisper-1"),  # 必须是 "whisper-1"，但你可以忽略它
    prompt: str = Form(None),
    language: str = Form(None),
    response_format: str = Form("json"),
    temperature: float = Form(0.0),
    api_key: str = Depends(verify_api_key)
):
    if not file.filename.endswith((".mp3", ".wav", ".m4a", ".ogg", ".webm")):
        raise HTTPException(status_code=400, detail="Unsupported audio format.")
    try:
        ext = file.filename.split(".")[-1]
        id = uuid.uuid4()

        filename = f"{id}.{ext}"
        contents = file.file.read()

        file_dir = f"{FILE_UPLOAD_PATH}/audio"
        os.makedirs(file_dir, exist_ok=True)
        file_path = f"{file_dir}/{filename}"

        with open(file_path, "wb") as f:
            f.write(contents)

        file_path = compress_audio(file_path)
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
    
    except Exception as e:
        log.exception(e)
        print(traceback.format_exc())

        raise HTTPException(
            status_code=400,
            detail=str(e),
        )

MAX_FILE_SIZE_MB = 25
MAX_FILE_SIZE = MAX_FILE_SIZE_MB * 1024 * 1024
def compress_audio(file_path):
    if os.path.getsize(file_path) > MAX_FILE_SIZE:
        file_dir = os.path.dirname(file_path)
        audio = AudioSegment.from_file(file_path)
        audio = audio.set_frame_rate(16000).set_channels(1)  # Compress audio
        compressed_path = f"{file_dir}/{id}_compressed.opus"
        audio.export(compressed_path, format="opus", bitrate="32k")
        log.debug(f"Compressed audio to {compressed_path}")

        if (
            os.path.getsize(compressed_path) > MAX_FILE_SIZE
        ):  # Still larger than MAX_FILE_SIZE after compression
            raise Exception(f"file size greater than {MAX_FILE_SIZE_MB}MB")
        return compressed_path
    else:
        return file_path