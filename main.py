from fastapi import FastAPI
from router import translate

app = FastAPI()

app.include_router(translate.router)