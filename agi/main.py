from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from agi.api.routes_chat import router as router_chat
from agi.api.routes_models import router as router_models
from agi.api.fast_api_file import router_file

app = FastAPI(title="AGI API", version="2.0.0")

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

app.include_router(router_chat)
app.include_router(router_models)
app.include_router(router_file)