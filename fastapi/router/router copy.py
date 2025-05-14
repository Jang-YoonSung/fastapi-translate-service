# from fastapi import APIRouter, Query, Body, Request
# from fastapi.encoders import jsonable_encoder

# from router.tasks import celery_translate_kor, celery_translate_eng

# from celery_config import celery_app
# from celery.result import AsyncResult

# from pydantic import BaseModel
# from typing import Optional, Union

# import logging
# import time
# import json

# logging.basicConfig(level=logging.INFO)

# main_router = APIRouter()

# class RequestBody(BaseModel):
#     content: str

# class ResponseBody(BaseModel):
#     msg: str
#     body: Optional[Union[str, list]] = None

# @main_router.post("/translate_kor", response_model=ResponseBody(..., media_type="text/plain"))
# def translate_kor(request: RequestBody):
#     celery_result = celery_translate_kor.delay(request.content)
#     print(celery_result)
#     return ResponseBody(msg = celery_result.state, body=celery_result.id)

# @main_router.post("/translate_eng", response_model=ResponseBody)
# def translate_eng(request: RequestBody):
#     celery_result = celery_translate_eng.delay(request.content)
#     print(celery_result)
#     return ResponseBody(msg = celery_result.state, body=celery_result.id)

# @main_router.get("/result", response_model=ResponseBody)
# def get_result(task_id: str):
#     result = AsyncResult(task_id, app=celery_app)
#     logging.info(result.result)

#     if result.state == "PENDING":
#         return ResponseBody(msg="PENDING", body="작업이 아직 큐에 있습니다.")
#     elif result.state == "STARTED":
#         return ResponseBody(msg="STARTED", body="작업이 실행 중입니다.")
#     elif result.state == "SUCCESS":
#         return ResponseBody(msg="SUCCESS", body=result.result)
#     elif result.state == "FAILURE":
#         return ResponseBody(msg="FAILURE", body="작업이 실패 하였습니다.")
#     else:
#         return ResponseBody(msg=result.state, body="기타 상태")