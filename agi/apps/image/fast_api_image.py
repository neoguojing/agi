import io
from fastapi import FastAPI, UploadFile, File, HTTPException,Depends,Form
from pydantic import BaseModel, Field
from typing import Literal, Optional, List, Union
from PIL import Image
from agi.apps.image.text2image import Text2Image
from agi.apps.image.image2image import Image2Image
from agi.apps.common import verify_api_key,ChatCompletionRequest
from agi.utils.common import Media
from datetime import datetime
from agi.config import log
import uuid
import traceback

app = FastAPI(
    title="AGI IMAGE GEN API",
    description="兼容 OpenAI API 的 AGI 接口",
    version="1.0.0",
    # lifespan=lifespan
)

text2img = Text2Image()
image2img = Image2Image()

class ImageGenRequest(BaseModel):
    model: str
    prompt: str
    negative_prompt: Optional[str] = None
    n: int = Field(1, ge=1, le=10)
    size: Literal["256x256", "512x512", "1024x1024"] = "1024x1024"
    response_format: Literal["url", "b64_json"] = "url"
    user: Optional[str] = None
    moderation: Literal["auto", "low"] = "auto"

class UrlData(BaseModel):
    url: str

class B64Data(BaseModel):
    b64_json: str

class ImageGenResponse(BaseModel):
    created: int
    data: List[Union[UrlData, B64Data]]

@app.post("/v1/images/generations")
async def generate(req: ImageGenRequest,api_key: str = Depends(verify_api_key)):
    log.info(req)

    try:
        width, height = map(int, req.size.lower().split("x"))
        image = text2img.invoke(req.prompt,
                                model=req.model,
                                width=width,
                                height=height,
                                negative_prompt=req.negative_prompt,
                                resp_format=req.response_format)
            
    except Exception as e:
        log.error(e)
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

    resp_data = []
    if req.response_format == "url":
        resp_data.append(UrlData(url=image))
    else:
        resp_data.append(B64Data(b64_json=image))

    return ImageGenResponse(
        created=int(datetime.utcnow().timestamp()),
        data = resp_data
    )

# 用于修改图片
@app.post("/v1/chat/completions")
async def chat_completion(
    request: ChatCompletionRequest,
    api_key: str = Depends(verify_api_key)
):
    log.info(request)
    if not request.messages:
        raise HTTPException(status_code=400, detail="No messages provided")

    msg = request.messages[-1]  # 只取最新的 user message
    text = ""
    input_image = None

    # 解析文本 + 图像URL
    for item in msg.content:
        if item.type == "text" and item.text:
            text = item.text
        elif item.type == "image_url" and item.image_url:
            input_image = item.image_url.url

    if not input_image or not text:
        raise HTTPException(status_code=400, detail="No image or text provided")

    resp_image = None
    try:
        image_data = Media.from_data(input_image)
        resp_image = image2img.invoke(text, input_image=image_data.data)
    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Model invocation error: {str(e)}")

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(datetime.utcnow().timestamp()),
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": resp_image
            },
            "finish_reason": "stop"
        }]
    }
