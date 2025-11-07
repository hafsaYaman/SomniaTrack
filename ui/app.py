import streamlit as st
import requests
import pandas as pd

API_BASE = st.secrets.get("API_BASE", "http://127.0.0.1:8000")

st.set_page_config(page_title="SomniaTrack", page_icon="😴", layout="centered")

st.title("SomniaTrack")
st.caption("AI sleep for everyone — rest without barriers.")

with st.expander("How it works"):
    st.write(
        "- Upload a short audio clip (WAV) recorded near your sleep area.\n"
        "- We estimate sleep state and a simple quality score (demo).\n"
        "- Use Sleep Equity Mode for shift schedules."
    )

st.subheader("Sleep Equity Mode")
c1, c2 = st.columns(2)
shift_start = c1.time_input("Shift start")
shift_end   = c2.time_input("Shift end")
if shift_start and shift_end:
    st.info("We’ll tailor tips to your schedule (e.g., recovery window 30–120 min after shift).")
    st.markdown("**Equity tips:**")
    st.markdown("- Darken the room for daytime sleep.")
    st.markdown("- Keep a short pre-sleep routine.")
    st.markdown("- Use white noise if the environment is loud.")

st.subheader("Analyze Your Sleep Audio")
audio_file = st.file_uploader("Upload audio (WAV only)", type=["wav"])
btn_analyze = st.button("Analyze")
btn_demo = st.button("Demo Mode")
placeholder = st.empty()

def call_predict(file_bytes, filename):
    files = {"audio": (filename, file_bytes, "application/octet-stream")}
    return requests.post(f"{API_BASE}/predict", files=files, timeout=30)

if btn_analyze:
    if not audio_file:
        st.warning("Please upload a WAV file first.")
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
                        st.write(data.get("notes",""))
                else:
                    st.error(f"API error: {resp.status_code} {resp.text}")
            except Exception as e:
                st.error(f"Request failed: {e}")

if btn_demo:
    with placeholder.container():
        st.info("Demo output (no API call).")
        st.success("State: **Asleep**")
        st.metric("Sleep Score", "83/100")
        st.write("Low RMS and minimal spikes suggest sustained rest.")

st.write("---")
st.caption("SomniaTrack • SleepEquity Mode supports shift workers and underserved communities.")
