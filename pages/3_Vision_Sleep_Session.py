import os
import time
from datetime import datetime
from typing import Any, Dict, List

import cv2
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from streamlit_webrtc import RTCConfiguration, WebRtcMode, webrtc_streamer

from utils.vision import analyze_frame, summarize_session

# Load .env so OPENAI_API_KEY is available locally
load_dotenv()

st.set_page_config(page_title="Vision Sleep Session", page_icon="📷", layout="wide")


RTC_CONFIG = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)


def _init_state() -> None:
    defaults = {
        "vision_running": False,
        "vision_consent": False,
        "vision_capture_interval": 20,
        "vision_max_frames": 60,
        "vision_start_time": None,
        "vision_end_time": None,
        "vision_frame_queue": [],  # pending raw frames to analyze
        "vision_events": [],  # analyzed frame metadata
        "vision_last_capture": 0.0,
        "vision_last_preview": None,
        "vision_summary": None,
        "vision_errors": [],
    }
    for key, val in defaults.items():
        st.session_state.setdefault(key, val)


_init_state()

api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
if api_key:
    # Make sure downstream helpers can see it even if only in secrets.toml
    os.environ.setdefault("OPENAI_API_KEY", api_key)

st.title("📷 Vision Sleep Session (GPT-4o)")
st.caption(
    "Capture periodic webcam frames, log posture/movement context with GPT-4o Vision, "
    "and generate a post-session summary."
)

with st.expander("Privacy & usage notes", expanded=False):
    st.markdown(
        "- Frames stay in-session only (memory) unless you export them manually.\n"
        "- Images are sent to OpenAI for analysis; avoid capturing identifiable details.\n"
        "- This is a demo, not medical advice."
    )

# Controls
consent = st.checkbox(
    "I consent to capturing webcam frames and sending them to OpenAI for analysis.",
    value=st.session_state.vision_consent,
)
st.session_state.vision_consent = consent

col_a, col_b, col_c = st.columns([1, 1, 1])
with col_a:
    st.session_state.vision_capture_interval = st.slider(
        "Capture every (seconds)", 5, 120, st.session_state.vision_capture_interval, 5
    )
with col_b:
    st.session_state.vision_max_frames = st.slider(
        "Max frames per session (to control cost)",
        10,
        200,
        st.session_state.vision_max_frames,
        10,
    )
with col_c:
    st.write(" ")
    st.write(" ")
    st.text(f"Frames logged: {len(st.session_state.vision_events)}")

col1, col2 = st.columns(2)
start_disabled = (
    st.session_state.vision_running
    or not consent
    or api_key is None
)
stop_disabled = not st.session_state.vision_running

if col1.button("Start session", type="primary", disabled=start_disabled):
    st.session_state.vision_running = True
    st.session_state.vision_start_time = datetime.utcnow()
    st.session_state.vision_end_time = None
    st.session_state.vision_frame_queue = []
    st.session_state.vision_events = []
    st.session_state.vision_summary = None
    st.session_state.vision_errors = []
    st.session_state.vision_last_capture = 0.0
    st.session_state.vision_last_preview = None

if col2.button("Stop session", disabled=stop_disabled):
    st.session_state.vision_running = False
    st.session_state.vision_end_time = datetime.utcnow()

# Show a small status header
status_cols = st.columns(3)
status_cols[0].metric(
    "Status", "Running" if st.session_state.vision_running else "Idle"
)
status_cols[1].metric(
    "Frames analyzed",
    f"{len(st.session_state.vision_events)}/{st.session_state.vision_max_frames}",
)
status_cols[2].metric(
    "Capture interval (s)", st.session_state.vision_capture_interval
)


def _video_frame_callback(frame):
    """
    Called on each incoming frame from the webcam. We sample at the configured interval,
    push bytes into a queue, and show a live preview in the sidebar.
    """
    img = frame.to_ndarray(format="bgr24")
    now = time.time()
    if (
        st.session_state.vision_running
        and len(st.session_state.vision_events) < st.session_state.vision_max_frames
        and consent
        and api_key
    ):
        if now - st.session_state.vision_last_capture >= st.session_state.vision_capture_interval:
            st.session_state.vision_last_capture = now
            ok, buf = cv2.imencode(".jpg", img)
            if ok:
                st.session_state.vision_frame_queue.append(
                    {"ts": now, "data": buf.tobytes()}
                )
                st.session_state.vision_last_preview = buf.tobytes()
    return frame


with st.container(border=True):
    st.subheader("Live capture")
    if not api_key:
        st.warning("Add OPENAI_API_KEY to .env or secrets to enable vision analysis.")
    webrtc_streamer(
        key="sleep-vision",
        mode=WebRtcMode.SENDONLY,
        rtc_configuration=RTC_CONFIG,
        media_stream_constraints={"video": True, "audio": False},
        video_frame_callback=_video_frame_callback,
    )
    if st.session_state.vision_last_preview:
        st.image(
            st.session_state.vision_last_preview,
            caption="Most recent captured frame",
            width=320,
        )


def _process_pending_frames() -> None:
    """
    Pull queued frames and run GPT-4o vision analysis (a few per rerun to stay responsive).
    """
    if not st.session_state.vision_frame_queue or not api_key:
        return

    processed = 0
    # Handle a few frames per rerun to avoid UI stalls if the queue is long.
    while st.session_state.vision_frame_queue and processed < 3:
        if len(st.session_state.vision_events) >= st.session_state.vision_max_frames:
            st.session_state.vision_running = False
            break

        item = st.session_state.vision_frame_queue.pop(0)
        try:
            result = analyze_frame(item["data"], api_key=api_key)
            ts_iso = datetime.fromtimestamp(item["ts"]).isoformat()
            st.session_state.vision_events.append(
                {
                    "timestamp_iso": ts_iso,
                    "posture": result.get("posture", "unknown"),
                    "movement": result.get("movement", "unknown"),
                    "bed_exit": bool(result.get("bed_exit", False)),
                    "light_change": result.get("light_change", "unknown"),
                    "note": result.get("note", "")[:120],
                    "confidence": result.get("confidence", None),
                }
            )
        except Exception as exc:  # pylint: disable=broad-except
            st.session_state.vision_errors.append(f"Frame analysis error: {exc}")
        processed += 1


_process_pending_frames()

if (
    not st.session_state.vision_running
    and st.session_state.vision_events
    and not st.session_state.vision_summary
):
    # If user manually stopped streaming (camera toggle), still produce summary
    if api_key:
        try:
            st.session_state.vision_summary = summarize_session(
                st.session_state.vision_events, api_key=api_key
            )
        except Exception as exc:  # pylint: disable=broad-except
            st.session_state.vision_errors.append(f"Summary error: {exc}")

st.divider()

col_log, col_summary = st.columns([1.3, 1])

with col_log:
    st.subheader("Event timeline")
    if st.session_state.vision_events:
        df = pd.DataFrame(st.session_state.vision_events)
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("No frames analyzed yet. Start a session to capture observations.")

    if st.session_state.vision_errors:
        st.error("Errors:\n- " + "\n- ".join(st.session_state.vision_errors))

with col_summary:
    st.subheader("Session summary")
    if st.session_state.vision_summary:
        summary = st.session_state.vision_summary
        st.write(summary.get("summary", ""))

        if summary.get("key_events"):
            st.markdown("**Key events**")
            for evt in summary["key_events"]:
                st.markdown(f"- {evt}")

        if summary.get("posture_distribution"):
            st.markdown("**Posture distribution**")
            dist_text = ", ".join(
                f"{k}: {v}" for k, v in summary["posture_distribution"].items()
            )
            st.text(dist_text)

        if summary.get("notable_movements"):
            st.markdown("**Notable movements**")
            for mv in summary["notable_movements"]:
                st.markdown(f"- {mv}")

        if summary.get("recommendations"):
            st.markdown("**Recommendations**")
            for rec in summary["recommendations"]:
                st.markdown(f"- {rec}")
    else:
        st.info("Summary appears after you stop the session.")
