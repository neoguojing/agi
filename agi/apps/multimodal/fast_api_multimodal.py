from fastapi import FastAPI, Request,Depends,HTTPException
from agi.apps.common import verify_api_key,ChatCompletionRequest
from agi.apps.multimodal.multi_modal import MultiModel
from agi.config import FILE_UPLOAD_PATH,log
from datetime import datetime
import uuid
import traceback

app = FastAPI()
client = MultiModel()
# ======== 定义 OpenAI Chat API 输入格式 ========


# ======== 模拟处理图像 + 文本内容的逻辑 ========
@app.post("/v1/chat/completions")
async def chat_completion(
    request: ChatCompletionRequest,
    api_key: str = Depends(verify_api_key)
):
    log.info(request)
    try:
        if not request.messages:
            raise HTTPException(status_code=400, detail="No messages provided")

        msg = request.messages[-1]  # 只取最新的 user message
        text = ""
        audio = image = video = None

        # 解析文本 + 图像URL
        for item in msg.content:
            if item.type == "text" and item.text:
                text = item.text
            elif item.type == "audio" and item.audio:
                audio = item.audio
            elif item.type == "image" and item.image:
                image = item.image
            elif item.type == "video" and item.video:
                video = item.video

            response_text, _, response_audio = client.invoke(text, audio=audio, image=image, video=video,return_audio=request.need_speech)
    except Exception as e:
        print(traceback.format_exc())

        raise HTTPException(status_code=500, detail=f"multimodal error: {str(e)}")

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