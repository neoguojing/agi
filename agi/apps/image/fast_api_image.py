import io
from fastapi import FastAPI, UploadFile, File, HTTPException,Depends
from pydantic import BaseModel, Field
from typing import Literal, Optional, List, Union
from PIL import Image
from agi.apps.image.text2image import Text2Image
from agi.apps.image.image2image import Image2Image
from agi.apps.common import verify_api_key
from datetime import datetime

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
    try:
        image = text2img.invoke(req.prompt,resp_format=req.response_format)
            
    except Exception as e:
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

@app.post("/v1/images/edits")
async def edit(prompt: str, image: UploadFile = File(...),api_key: str = Depends(verify_api_key)):
    input_img = Image.open(io.BytesIO(await image.read())).convert("RGB")
    try:
        output = image2img.invoke(input=prompt, input_image=input_img)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return ImageGenResponse(
        created=int(datetime.utcnow().timestamp()),
        data = [UrlData(url=output)]
    )
