"""Backend application entry point."""

from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from app.config import APP_NAME, is_model_loaded
from app.ml_service import enhance_image_bytes, initialize_model, is_opencv_available


@asynccontextmanager
async def lifespan(_app: FastAPI):
    initialize_model()
    yield


app = FastAPI(title=APP_NAME, lifespan=lifespan)

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
        "opencv_available": is_opencv_available(),
    }


@app.post("/api/enhance")
async def enhance(
    file: UploadFile = File(...),
    scale_factor: int = Form(4),
):
    """Convert/enhance an uploaded image using denoise, upscale, and deblur."""
    if scale_factor not in {2, 3, 4}:
        raise HTTPException(
            status_code=400,
            detail="scale_factor must be 2, 3, or 4.",
        )

    filename = file.filename or "image.jpg"
    suffix = filename.lower()
    if not suffix.endswith((".jpg", ".jpeg", ".png")):
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Please upload a JPG, JPEG, or PNG image.",
        )

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")

    try:
        enhanced_bytes, media_type, original_size, output_size = enhance_image_bytes(
            image_bytes,
            filename=filename,
            scale_factor=scale_factor,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    output_name = f"enhanced_{filename}"
    return Response(
        content=enhanced_bytes,
        media_type=media_type,
        headers={
            "Content-Disposition": f'attachment; filename="{output_name}"',
            "X-Original-Size": f"{original_size[0]}x{original_size[1]}",
            "X-Output-Size": f"{output_size[0]}x{output_size[1]}",
            "Access-Control-Expose-Headers": "X-Original-Size, X-Output-Size",
        },
    )
