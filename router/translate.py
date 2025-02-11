from fastapi import APIRouter, Query, Body, Request
from fastapi.encoders import jsonable_encoder

from service import translate_service

import logging

logging.basicConfig(level=logging.INFO)

router = APIRouter()

@router.post("/translate-kor")
async def translate_kor(content: str = Body(..., media_type="text/plain")):
    return translate_service.translate_kor(content)

@router.post("/translate-eng")
async def translate_eng(content: str = Body(..., media_type="text/plain")):
    return translate_service.translate_eng(content)
