import os
import sys
import warnings
from pathlib import Path

import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

# Ensure repo root is on path so utils package resolves when Streamlit runs from other cwd
ROOT = Path(__file__).resolve().parent.parent
root_str = str(ROOT)
if root_str not in sys.path:
    sys.path.insert(0, root_str)

load_dotenv(dotenv_path=ROOT / ".env", override=False)

# Suppress noisy ScriptRunContext warnings
warnings.filterwarnings("ignore", message=".*ScriptRunContext.*")

API_BASE = os.getenv("API_BASE") or st.secrets.get("API_BASE", "http://127.0.0.1:8000")

st.set_page_config(page_title="SomniaTrack", page_icon="😴", layout="wide")

st.title("SomniaTrack")
st.caption("AI sleep for everyone — rest without barriers.")

with st.expander("How it works"):
    st.write(
        "- Upload a short audio clip (WAV/FLAC/OGG) recorded near your sleep area.\n"
        "- We estimate sleep state and a simple quality score (demo).\n"
        "- Use Sleep Equity Mode for shift schedules.\n"
        "- Vision tab: capture periodic webcam frames, log posture/movement, and get a summary."
    )

audio_tab, vision_tab = st.tabs(["Audio Analysis", "Vision Sleep Session (GPT-4o)"])


def render_audio_tab():
    st.subheader("Sleep Equity Mode")
    col_a, col_b = st.columns(2)
    shift_start = col_a.time_input("Shift start")
    shift_end = col_b.time_input("Shift end")
    if shift_start and shift_end:
        st.info("We'll tailor tips to your schedule (e.g., recovery window 30–120 min after shift).")
        st.markdown("**Equity Tips:**")
        st.markdown("- Darken the room for daytime sleep (curtains/towel).")
        st.markdown("- Keep a 10-minute pre-sleep routine, even on short windows.")
        st.markdown("- If noise is high, try a low-volume fan or white noise.")

    st.subheader("Analyze Your Sleep Audio")
    audio_file = st.file_uploader("Upload audio", type=["wav", "flac", "ogg"])
    btn_analyze = st.button("Analyze")
    btn_demo = st.button("Demo Mode")
    placeholder = st.empty()

    def call_predict(file_bytes, filename):
        files = {"audio": (filename, file_bytes, "application/octet-stream")}
        return requests.post(f"{API_BASE}/predict", files=files, timeout=30)

    if btn_analyze:
        if not audio_file:
            st.warning("Please upload a WAV/FLAC/OGG file first.")
        else:
            with st.spinner("Analyzing..."):
                try:
                    resp = call_predict(audio_file.getvalue(), audio_file.name)
                    if resp.ok:
                        data = resp.json()
                        with placeholder.container():
                            st.write("### Results")
                            st.success(f"State: **{data['state'].title()}**")
                            st.metric("Sleep Score", f"{data['score']}/100")
                            with st.container(border=True):
                                st.write("**Notes**")
                                st.write(data.get("notes", ""))

                            if "history" not in st.session_state:
                                st.session_state.history = []
                            st.session_state.history.append(
                                {"state": data["state"], "score": data["score"]}
                            )
                            st.subheader("Session History (this run)")
                            df = pd.DataFrame(st.session_state.history)
                            st.dataframe(df, use_container_width=True)
                    else:
                        st.error(f"API error: {resp.status_code} {resp.text}")
                except Exception as e:  # pylint: disable=broad-except
                    st.error(f"Request failed: {e}")

    if btn_demo:
        with placeholder.container():
            st.info("Demo output (no API call).")
            st.success("State: **Asleep**")
            st.metric("Sleep Score", "83/100")
            st.write("Low RMS and minimal spikes suggest sustained rest.")
            if "history" not in st.session_state:
                st.session_state.history = []
            st.session_state.history.append({"state": "asleep", "score": 83})
            st.subheader("Session History (this run)")
            df = pd.DataFrame(st.session_state.history)
            st.dataframe(df, use_container_width=True)


def render_vision_tab():
    st.subheader("Vision Sleep Session (embedded capture)")
    st.caption(
        "Webcam interval capture runs below and posts frames to the API for GPT-4o sleep-context analysis."
    )
    try:
        with open(ROOT / "vision_capture.html", "r", encoding="utf-8") as f:
            html = f.read()
        # Embed the standalone JS capture UI so everything runs on this page
        components.html(html, height=1050, scrolling=True)
    except Exception as exc:  # pylint: disable=broad-except
        st.error(f"Failed to load vision_capture.html: {exc}")


with audio_tab:
    render_audio_tab()

with vision_tab:
    render_vision_tab()

st.write("---")
st.caption("SomniaTrack • Audio + Vision demos for sleep tracking. Not medical advice.")
