from fastapi import HTTPException, File, UploadFile
from fastapi.responses import StreamingResponse, FileResponse

import logging
import os
import re
from pathlib import Path

import sys

# huggingface 관련 모델 라이브러리
from transformers import pipeline
import torch
import time
import logging
from datetime import datetime

from transformers import BitsAndBytesConfig

quantization_config=BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type='nf4'
)

# Load model directly
from transformers import T5ForConditionalGeneration, T5Tokenizer

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

logging.basicConfig(level=logging.INFO)
logging.info("Starting to load the model...")

model_name = "/app/translation/models/madlad400-3b-mt"
model = T5ForConditionalGeneration.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    device_map="cuda:0"
    )

logging.info("Model loaded successfully.")
tokenizer = T5Tokenizer.from_pretrained(model_name)

def translate_kor(content: str):
    translate_content = []
    for text in content.split('\n'):
        logging.info(text)
        start = time.time()
        inputs = tokenizer("<2ko> " + text, return_tensors="pt").input_ids.to(model.device)
        outputs = model.generate(input_ids=inputs, max_length=512, no_repeat_ngram_size=3)# 3-gram 반복 방지
        end = time.time()
        elapsed_time = end - start
        print(tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=False))
        translate_content.append(tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=False))
    return {"translate_contents" : translate_content}

def translate_eng(content: str):
    translate_content = []
    for text in content.split('\n'):
        logging.info(text)
        start = time.time()
        inputs = tokenizer("<2en> " + text, return_tensors="pt").input_ids.to(model.device)
        outputs = model.generate(input_ids=inputs, max_length=512, no_repeat_ngram_size=3)
        end = time.time()
        elapsed_time = end - start
        print(tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=False))
        translate_content.append(tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=False))
    return {"translate_contents" : translate_content}