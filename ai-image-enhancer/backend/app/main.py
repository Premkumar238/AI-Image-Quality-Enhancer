"""Backend application entry point."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import APP_NAME, is_model_loaded

app = FastAPI(title=APP_NAME)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"message": "AI Image Quality Enhancer API is running"}


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "application_name": APP_NAME,
        "model_loaded": is_model_loaded(),
    }
