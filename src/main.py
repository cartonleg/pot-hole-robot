from fastapi import FastAPI
from routes import process

app = FastAPI()

app.include_router(process.process_router)
