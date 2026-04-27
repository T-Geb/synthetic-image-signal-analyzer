from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from detector import ImageAnalysisError, ModelRuntimeUnavailable, analyze_image


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png"}
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}

app = FastAPI(
    title="Synthetic Media Signal Analyzer",
    description="Image-only MVP for synthetic likelihood analysis.",
    version="0.1.0",
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/analyze-image")
async def analyze_upload(file: UploadFile = File(...)):
    filename = file.filename or ""
    suffix = Path(filename).suffix.lower()

    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Upload a JPG, JPEG, or PNG image.",
        )

    if file.content_type and file.content_type not in ALLOWED_CONTENT_TYPES | {"application/octet-stream"}:
        raise HTTPException(
            status_code=400,
            detail="Upload a JPG, JPEG, or PNG image.",
        )

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="The uploaded image is empty.")

    try:
        return analyze_image(image_bytes)
    except ImageAnalysisError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ModelRuntimeUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Image analysis could not be completed.",
        ) from exc
