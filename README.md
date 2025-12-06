# SomniaTrack
A sleep tracker AI app

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file (one is included with placeholders):
```
OPENAI_API_KEY=your-openai-api-key
API_BASE=http://127.0.0.1:8000
```

## Run the services

- **Model API (FastAPI)**: `uvicorn api.model_api:app --reload --host 0.0.0.0 --port 8000`
- **Unified Streamlit UI (audio + vision tabs)**: `streamlit run ui/app.py`

Make sure the API is running before hitting it from the UIs.

## Vision sleep session (new)

- Requires `OPENAI_API_KEY`.
- Prompts for webcam consent; captures frames periodically; sends them to GPT-4o Vision to log posture/movement context.
- On stop, it generates a concise summary with key events and suggestions. Access it from the “Vision Sleep Session” tab in `ui/app.py`.

> Privacy: frames stay in memory unless you export them; images are sent to OpenAI for analysis. Avoid capturing identifiable details.
