from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import soundfile as sf
import io, librosa

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
