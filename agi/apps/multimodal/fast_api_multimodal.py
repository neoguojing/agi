from fastapi import FastAPI, Request,Depends,HTTPException
from agi.apps.common import verify_api_key,ChatRequest
from agi.apps.multimodal.multi_modal import MultiModel
from agi.utils.common import detect_input_and_save
from agi.config import FILE_UPLOAD_PATH
from datetime import datetime
import uuid

app = FastAPI()
client = MultiModel()
# ======== 定义 OpenAI Chat API 输入格式 ========


# ======== 模拟处理图像 + 文本内容的逻辑 ========
@app.post("/v1/chat/completions")
async def chat_completion(
    request: ChatRequest,
    api_key: str = Depends(verify_api_key)
):
    if not request.messages:
        raise HTTPException(status_code=400, detail="No messages provided")

    msg = request.messages[-1]  # 只取最新的 user message
    text = ""
    media = None
    input_type = ""

    # 解析文本 + 图像URL
    for item in msg.content:
        if item.type == "text" and item.text:
            text = item.text
        elif item.type == "image_url" and item.image_url:
            media = item.image_url.url

    # 检测多模态资源类型
    audio = image = video = None
    if media:
        try:
            saved_path, input_type = detect_input_and_save(media, target_path=FILE_UPLOAD_PATH)
            if input_type == "audio":
                audio = saved_path
            elif input_type == "image":
                image = saved_path
            elif input_type == "video":
                video = saved_path
            else:
                raise ValueError(f"Unsupported media type: {input_type}")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Media load error: {str(e)}")

    # 调用自定义 client.invoke(text, audio=..., image=..., video=...)
    try:
        response_text, _, response_audio = client.invoke(text, audio=audio, image=image, video=video)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model invocation error: {str(e)}")

    # 组装结果（文字优先，音频路径作为补充）
    final_response = response_audio or response_text or "无内容返回"

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(datetime.utcnow().timestamp()),
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": final_response
            },
            "finish_reason": "stop"
        }]
    }