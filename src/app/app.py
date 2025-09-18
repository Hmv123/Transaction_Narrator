import os
import pandas as pd
import io
import contextlib
import streamlit as st

from src.input_check.Input_validation import schema_validation
from src.Anomaly_detector.Anomaly_detector import detect_anomalies
from src.Narrator.Narrative_generator import generate_narratives

# ---------- CONFIG ----------
INPUT_DIR = "data/Input"
OUTPUT_DIR = "data/Output"
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

st.set_page_config(page_title="Transaction Anomaly Narrator", layout="wide")
st.title("Transaction Anomaly Narrator")

# ---------- FILE UPLOAD ----------
uploaded_file = st.file_uploader("Upload the Transactions", type=["csv"], key="file_uploader")

if uploaded_file:
    if st.button("Upload File"):
        input_path = os.path.join(INPUT_DIR, uploaded_file.name)
        with open(input_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"File uploaded and saved to {input_path}")
        st.session_state["uploaded_file_path"] = input_path

# ---------- PAGE LAYOUT ----------
left_col, right_col = st.columns([2, 1])  # Wider left, narrower right

# ---------- LOG AREA ON RIGHT ----------
with right_col:
    st.subheader("Workflow Status")
    with st.expander("Show Logs", expanded=True):
        log_box = st.empty()

    def show_logs(buffer):
        """Helper to show logs in a scrollable text area"""
        log_box.text_area("Logs", buffer.getvalue(), height=500)

# ---------- MAIN WORKFLOW ON LEFT ----------
with left_col:
    st.subheader("File Validation")
    if st.button("Validation:▶️"):
        if "uploaded_file_path" in st.session_state:
            input_path = st.session_state["uploaded_file_path"]
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                schema_validation(input_path)
            show_logs(buffer)
        else:
            st.error("Please upload a file before running schema validation.")

    st.subheader("Anomaly Report")
    if st.button("Anomaly Report:▶️"):
        if "uploaded_file_path" in st.session_state:
            input_path = st.session_state["uploaded_file_path"]
            output_path = os.path.join(OUTPUT_DIR, "anomalies_output.csv")
            st.session_state["anomalies_output_path"] = output_path
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                detect_anomalies(input_path, output_path)
            show_logs(buffer)
            # Show download button for anomaly output
            if os.path.exists(output_path):
                with open(output_path, "rb") as f:
                    st.download_button(
                        label="Download Anomaly Output CSV",
                        data=f,
                        file_name="anomalies_output.csv",
                        mime="text/csv"
        )
        else:
            st.error("Please upload a file before running anomaly detection.")

    st.subheader("Transaction Narrator")
    if st.button("Narrator:▶️"):
        if "anomalies_output_path" in st.session_state:
            input_path = st.session_state["anomalies_output_path"]
            output_path = os.path.join(OUTPUT_DIR, "anomalies_with_narratives.csv")
            st.session_state["anomalies_with_narratives"] = output_path
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                generate_narratives(input_path, output_path)
            show_logs(buffer)
            # Show download button for Narrator output
            if os.path.exists(output_path):
                with open(output_path, "rb") as f:
                    st.download_button(
                        label="Download Anomaly Narrative Output CSV",
                        data=f,
                        file_name="anomalies_with_narratives.csv",
                        mime="text/csv")
        else:
            st.error("Please run anomaly detection before generating narratives.")

# ---------- QUIT BUTTON ----------
st.markdown("---")  # horizontal divider
st.subheader("Exit Session")

if st.button("Quit App"):
    # Clear session state
    for key in st.session_state.keys():
        del st.session_state[key]

    # Display exit message
    st.warning("Session ended. You may close the browser tab or refresh to start over.")
    
    # Stop further execution
    st.stop()
