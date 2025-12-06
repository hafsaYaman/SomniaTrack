from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import soundfile as sf
import io, librosa

from utils.vision import analyze_frame, summarize_session

app = FastAPI(title="SomniaTrack Model API", version="0.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

class PredictResp(BaseModel):
    state: str    # "asleep" | "awake"
    score: float  # 0-100
    notes: str


class VisionResp(BaseModel):
    posture: str
    movement: str
    bed_exit: bool
    light_change: str
    note: str
    confidence: float | None = None

class VisionSummaryResp(BaseModel):
    summary: str | None = None
    key_events: list[str] | None = None
    posture_distribution: dict[str, float] | None = None
    notable_movements: list[str] | None = None
    recommendations: list[str] | None = None
    raw: str | None = None

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/predict", response_model=PredictResp)
async def predict(audio: UploadFile = File(...)):
    if not audio.filename.lower().endswith((".wav", ".flac", ".ogg")):
        raise HTTPException(status_code=400, detail="Please upload a WAV/FLAC/OGG file.")
    data = await audio.read()
    y, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = np.mean(y, axis=1)  # mono
    rms = float(librosa.feature.rms(y=y).mean())
    asleep = rms < 0.02                     # simple demo heuristic
    score  = max(0.0, 100.0 - rms * 4000)   # demo scoring
    return PredictResp(
        state="asleep" if asleep else "awake",
        score=round(score, 1),
        notes=f"Avg RMS={rms:.4f}. Lower RMS indicates quieter periods (more likely asleep)."
    )


@app.post("/vision-analyze", response_model=VisionResp)
async def vision_analyze(frame: UploadFile = File(...)):
    """
    Accept a single image frame (jpg/png) and return posture/movement context via GPT-4o.
    """
    if not frame.filename.lower().endswith((".jpg", ".jpeg", ".png")):
        raise HTTPException(status_code=400, detail="Please upload a JPG or PNG frame.")
    data = await frame.read()
    try:
        result = analyze_frame(
            data,
            api_key=None,  # uses env OPENAI_API_KEY
            image_media_type="png" if frame.filename.lower().endswith(".png") else "jpeg",
        )
    except Exception as exc:  # pylint: disable=broad-except
        raise HTTPException(status_code=500, detail=f"Vision analysis failed: {exc}") from exc

    return VisionResp(
        posture=result.get("posture", "unknown"),
        movement=result.get("movement", "unknown"),
        bed_exit=bool(result.get("bed_exit", False)),
        light_change=result.get("light_change", "unknown"),
        note=result.get("note", "")[:120],
        confidence=result.get("confidence", None),
    )


@app.post("/vision-summarize", response_model=VisionSummaryResp)
async def vision_summarize(events: list[dict]):
    """
    Summarize a session of per-frame events.
    Each event should include timestamp_iso, posture, movement, bed_exit, and note.
    """
    try:
        result = summarize_session(events, api_key=None)  # uses env OPENAI_API_KEY
    except Exception as exc:  # pylint: disable=broad-except
        raise HTTPException(status_code=500, detail=f"Vision summary failed: {exc}") from exc
    return result
