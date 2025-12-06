import streamlit as st
from utils.api import API


st.set_page_config(page_title="Onboarding & Consent", page_icon="🧾", layout="centered")


st.header("🧾 Onboarding & Consent")


st.markdown(
"""
Purpose: SomniaTrack analyzes short audio segments to estimate sleep state and quality.
We do not store raw audio by default. With consent, we can store summaries to help you track trends.
"""
)


with st.expander("What data is used?", expanded=False):
    st.markdown("- Audio features (e.g., mel spectrogram statistics) — not the raw audio contents")
    st.markdown("- Optional: shift schedule, ZIP for community trends, self‑reported noise levels")


consent = st.checkbox("I consent to processing my audio and anonymized summaries for research/improvement.")
store_summary = st.checkbox("Allow storing only **summary metrics** (not raw audio)")


st.session_state["consent"] = consent
st.session_state["store_summary"] = store_summary


col1, col2 = st.columns(2)
with col1:
    if st.button("Check API Health", use_container_width=True):
        try:
            st.success(API.health())
        except Exception as e:  # pylint: disable=broad-except
            st.error(f"API error: {e}")
with col2:
    if st.button("View API Version", use_container_width=True):
        try:
            st.info(API.version())
        except Exception as e:  # pylint: disable=broad-except
            st.error(f"API error: {e}")
