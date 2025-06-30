import io
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
import torch
from PIL import Image

app = FastAPI()

# 载入模型
device = "cuda" if torch.cuda.is_available() else "cpu"
txt2img = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5").to(device)
img2img = StableDiffusionImg2ImgPipeline.from_pretrained("runwayml/stable-diffusion-v1-5").to(device)

class GenerateRequest(BaseModel):
    prompt: str
    num_images: int = 1
    guidance_scale: float = 7.5
    width: int = 512
    height: int = 512
    strength: float = 0.75  # 图生图时用

@app.post("/api/v1/images/generations")
async def generate(req: GenerateRequest):
    try:
        images = txt2img(
            req.prompt, 
            guidance_scale=req.guidance_scale, 
            num_images=req.num_images, 
            height=req.height, 
            width=req.width).images
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    outputs = []
    for img in images:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        outputs.append(buf.read())

    return StreamingResponse(io.BytesIO(b"".join(outputs)), media_type="application/octet-stream")

@app.post("/api/v1/images/edits")
async def edit(prompt: str, image: UploadFile = File(...), strength: float = 0.75):
    input_img = Image.open(io.BytesIO(await image.read())).convert("RGB").resize((512,512))
    try:
        output = img2img(
            prompt=prompt, init_image=input_img, strength=strength).images[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    buf = io.BytesIO()
    output.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")
