import base64
import json
import os
import textwrap
from datetime import datetime
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables early so OPENAI_API_KEY from .env is picked up in dev
load_dotenv()


def _client(api_key: Optional[str] = None) -> OpenAI:
    key = api_key or os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("Set OPENAI_API_KEY in .env or environment variables to use vision analysis.")
    return OpenAI(api_key=key)


def analyze_frame(
    image_bytes: bytes,
    api_key: Optional[str] = None,
    image_media_type: str = "jpeg",
) -> Dict[str, Any]:
    """
    Run a single-frame GPT-4o vision analysis.
    Returns a JSON-like dict with posture, movement, bed_exit, light_change, note, and confidence.
    """
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    media_type = image_media_type or "jpeg"
    system_prompt = textwrap.dedent(
        """
        You are a sleep posture observer analyzing a still frame from a night-vision webcam.
        Extract only visible context — do not guess beyond the image. Avoid personal identifiers.
        Respond strictly as JSON with keys:
        - posture: supine | side-left | side-right | prone | sitting | unknown
        - movement: none | minor | major
        - bed_exit: true | false
        - light_change: none | up | down | unknown
        - note: <= 120 characters, concise sleep-relevant observation (position, movement, bed exit, light/noise). No medical advice.
        - confidence: 0-1 float
        """
    ).strip()

    user_content = [
        {"type": "text", "text": "Analyze this single still frame for posture/movement context."},
        {"type": "image_url", "image_url": {"url": f"data:image/{media_type};base64,{b64}"}},
    ]

    resp = _client(api_key).chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        response_format={"type": "json_object"},
        temperature=0.2,
        max_tokens=300,
    )
    message = resp.choices[0].message.content
    try:
        return json.loads(message)
    except json.JSONDecodeError:
        return {"raw": message}


def summarize_session(events: List[Dict[str, Any]], api_key: Optional[str] = None) -> Dict[str, Any]:
    """
    Summarize a session given per-frame events.
    Each event should include timestamp_iso, posture, movement, bed_exit, and note fields.
    """
    if not events:
        return {}

    def _fmt_event(ev: Dict[str, Any]) -> str:
        ts = ev.get("timestamp_iso") or ev.get("timestamp") or ""
        posture = ev.get("posture", "unknown")
        movement = ev.get("movement", "unknown")
        bed_exit = ev.get("bed_exit", False)
        note = ev.get("note", "")
        return f"{ts} | posture={posture} | movement={movement} | bed_exit={bed_exit} | note={note}"

    lines = "\n".join(_fmt_event(e) for e in events)
    summary_prompt = textwrap.dedent(
        """
        You are summarizing a sleep session from periodic webcam snapshots.
        Create a concise overview (not medical advice). Emphasize posture trends, movements, and bed exits.
        Respond as JSON with keys:
        - summary: short paragraph (<= 120 words)
        - key_events: list of bullet strings
        - posture_distribution: object mapping posture -> percentage estimate
        - notable_movements: list of bullet strings
        - recommendations: 3 short suggestions tailored to observed issues (sleep hygiene, environment). No medical claims.
        """
    ).strip()

    resp = _client(api_key).chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": summary_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Observations by timestamp (iso):\n" + lines},
                ],
            },
        ],
        response_format={"type": "json_object"},
        temperature=0.3,
        max_tokens=500,
    )
    message = resp.choices[0].message.content
    try:
        return json.loads(message)
    except json.JSONDecodeError:
        return {"raw": message}
