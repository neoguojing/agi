from fastapi import APIRouter, Depends
from pydantic import BaseModel
from typing import List
from agi.apps.common import verify_api_key

router = APIRouter()

class Model(BaseModel):
    id: str
    object: str
    created: int
    owned_by: str

class ModelListResponse(BaseModel):
    object: str
    data: List[Model]

@router.get("/v1/models", response_model=ModelListResponse)
async def list_models(api_key: str = Depends(verify_api_key)):
    return ModelListResponse(
        object="list",
        data=[Model(id="agi", object="model", created=1677654321, owned_by="agi")]
    )