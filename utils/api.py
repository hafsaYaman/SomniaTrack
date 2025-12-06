import os
from typing import Any, Dict, Optional

import requests
from dotenv import load_dotenv

# Load variables from .env so local dev picks up OPENAI/API_BASE without extra config
load_dotenv()

API_BASE = os.getenv("API_BASE", "http://127.0.0.1:8000")


class API:
    """Helper for talking to the FastAPI backend."""

    @staticmethod
    def health() -> Dict[str, Any]:
        resp = requests.get(f"{API_BASE}/health", timeout=10)
        resp.raise_for_status()
        return resp.json()

    @staticmethod
    def version() -> Dict[str, Any]:
        # FastAPI app doesn't expose version yet; return base to show connectivity.
        return {"api_base": API_BASE}

    @staticmethod
    def predict(
        file_bytes: bytes,
        filename: str = "audio.wav",
        is_shift_worker: Optional[bool] = None,
        avg_env_noise_db: Optional[float] = None,
        zipcode: Optional[str] = None,
    ) -> Dict[str, Any]:
        files = {"audio": (filename or "audio.wav", file_bytes, "application/octet-stream")}
        data: Dict[str, Any] = {}
        if is_shift_worker is not None:
            data["is_shift_worker"] = str(is_shift_worker).lower()
        if avg_env_noise_db is not None:
            data["avg_env_noise_db"] = avg_env_noise_db
        if zipcode:
            data["zipcode"] = zipcode

        resp = requests.post(f"{API_BASE}/predict", files=files, data=data, timeout=60)
        resp.raise_for_status()
        return resp.json()

