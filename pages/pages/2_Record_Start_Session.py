import streamlit as st
import numpy as np
from utils.api import API


st.set_page_config(page_title="Record / Start Session", page_icon="🎙️", layout="wide")


st.header("🎙️ Record / Start Session")


st.markdown("Upload a short nighttime audio clip (15–60 seconds) to analyze.")


left, right = st.columns([2, 1])
with left:
uploaded = st.file_uploader("Audio file (wav/mp3/flac)", type=["wav", "mp3", "flac"], accept_multiple_files=False)
is_shift = st.toggle("I'm a night‑shift worker", value=False)
zipcode = st.text_input("ZIP code (optional for community trends)")
manual_noise = st.slider("Avg environmental noise (RMS dB, optional)", min_value=-60.0, max_value=-5.0, value=-25.0, step=0.5)


if st.button("Analyze", type="primary"):
if not uploaded:
st.warning("Please upload an audio file.")
else:
try:
bytes_data = uploaded.read()
result = API.predict(
file_bytes=bytes_data,
filename=uploaded.name,
is_shift_worker=is_shift,
avg_env_noise_db=manual_noise,
zipcode=zipcode or None,
)
st.session_state["last_result"] = result
st.success("Analysis complete! Check the Results Dashboard page.")
except Exception as e:
st.error(f"Prediction failed: {e}")


with right:
st.subheader("Live Noise Level (demo)")
st.caption("Approximate indicator without mic access. Use native app/mobile for real‑time.")
# Simulated meter
level = np.random.uniform(0, 1)
st.progress(level, text=f"Level ~ {int(level*100)}% (simulated)")
