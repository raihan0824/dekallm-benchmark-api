from fastapi import APIRouter,Depends
from pydantic import BaseModel
from typing import Optional,List
import os
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
from typing import List
from math import ceil
from server.utils.security import get_current_user


load_dotenv()

class Item(BaseModel):
    input: Optional[str] = None

generation_router=APIRouter(tags=['Generate response'],
                            dependencies=[Depends(get_current_user)]
                            )

general_router=APIRouter(tags=['General'])
@generation_router.post("/api/v1/generate/")
async def generate(item: Item):
    return {"response": ""}

@general_router.get("/api/v1/model/")
async def get_model_path():
    return {"model": ""}
