from fastapi import HTTPException, File, UploadFile
from fastapi.responses import StreamingResponse, FileResponse

import logging
import os
import re
from pathlib import Path
import sys
import time
import logging
from datetime import datetime

# huggingface 관련 모델 라이브러리
from transformers import pipeline, BitsAndBytesConfig
import torch

from celery_config import celery_app
from celery.signals import worker_process_init

quantization_config=BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type='nf4'
)

# Load model directly
from transformers import T5ForConditionalGeneration, T5Tokenizer

@worker_process_init.connect
def load_llm_model(**kwargs):
    global model, tokenizer
    logging.basicConfig(level=logging.INFO)
    logging.info("Starting to load the model...")

    model_name = "/app/models/madlad400-3b-mt"
    model = T5ForConditionalGeneration.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="cuda:0"
        )

    logging.info("Model loaded successfully.")
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    logging.info("Tokenizer loaded successfully.")

def text_splitter(content):
    chunks = []
    split_num = 500
    while len(content) > split_num:
        # sub_chunk = content[split_num:].split("\n", 1)
        sub_chunk_raw = content[split_num:]
        if "." in sub_chunk_raw:
            sub_chunk = sub_chunk_raw.split(".", 1)
        elif "\n" in sub_chunk_raw:
            sub_chunk = sub_chunk_raw.split("\n", 1)
        else:
            sub_chunk = sub_chunk_raw
        chunk = content[:split_num] + sub_chunk[0]
        chunks.append(chunk.replace("\n", " ").strip())
        content = sub_chunk[1] if len(sub_chunk) > 1 else ''
    if content.strip():
        chunks.append(content.replace("\n", " ").strip())
    return chunks

conversion_dict = {
    "이다.": "입니다.",
    "았다.": "았습니다.",
    "었다.": "었습니다.",
    "있다.": "있습니다.",
    "없다.": "없습니다.",
    "된다.": "됩니다.",
    "하다.": "합니다.",
    "했다.": "했습니다.",
    "한다.": "합니다.",
    "왔다.": "왔습니다.",
    "친다.": "칩니다.",
    "낸다.": "냅니다.",
}

special_case = {
    "는다.": "습니다.",
}

pattern = re.compile("|".join(map(re.escape, conversion_dict.keys())))
special_pattern = re.compile(r"([가-힣]+)는다\.")

def convert_polite(sentence):
    sentence = sentence.strip()
    sentence = pattern.sub(lambda m: conversion_dict[m.group(0)], sentence)
    sentence = special_pattern.sub(r"\1%s" % special_case["는다."], sentence)
    return sentence

@celery_app.task
def celery_translate_kor(content:str):
    # if task_result.state == 'PENDING':
    #     time.sleep(1)
    #     celery_summ_result(task_id)
    #     logging.info(f"[LOCAL] | {__name__}.py | summarize\tProcessing | PENDING")
    # elif task_result.state == 'SUCCESS':
    #     return task_result.result
    # else:
    #     return task_result.state
    # while task_result.state == 'SUCCESS':
    #     time.sleep(1)
    #     task_result = summ_meet.AsyncResult(task_id)
    #     logging.info(f"[LOCAL] | {__name__}.py | summarize\tProcessing | PENDING")
        # return task_result.result
    translate_content = []
    chunks = text_splitter(content)
    for text in chunks:
        start = time.time()
        inputs = tokenizer("<2ko> " + text, return_tensors="pt").input_ids.to(model.device)
        outputs = model.generate(input_ids=inputs, max_length=512)
        end = time.time()
        elapsed_time = end - start
        result = tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        result = [convert_polite(s.strip() + '.') for s in result.split('.') if s.strip()]
        print(result)
        translate_content.extend(result)
    return translate_content

@celery_app.task
def celery_translate_eng(content:str):
    translate_content = []
    for text in content.split('\n'):
        start = time.time()
        inputs = tokenizer("<2en> " + text, return_tensors="pt").input_ids.to(model.device)
        outputs = model.generate(input_ids=inputs, max_length=512)
        end = time.time()
        elapsed_time = end - start
        print(tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=False))
        translate_content.append(tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=False))
    return translate_content
    # if task_result.state == 'PENDING':
    #     time.sleep(1)
    #     celery_summ_result(task_id)
    #     logging.info(f"[LOCAL] | {__name__}.py | summarize\tProcessing | PENDING")
    # elif task_result.state == 'SUCCESS':
    #     return task_result.result
    # else:
    #     return task_result.state
    # while task_result.state == 'SUCCESS':
    #     time.sleep(1)
    #     task_result = summ_meet.AsyncResult(task_id)
    #     logging.info(f"[LOCAL] | {__name__}.py | summarize\tProcessing | PENDING")
        # return task_result.result