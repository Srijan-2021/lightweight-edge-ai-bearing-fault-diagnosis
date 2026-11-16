from __future__ import annotations

import base64
import json
import tempfile
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src import paderborn_inference as pbi

ROOT = Path(__file__).resolve().parent
MODEL_B64 = ROOT / "models" / "paderborn_baseline" / "paderborn_baseline_pipeline.joblib.b64"
MODEL_META = ROOT / "models" / "paderborn_baseline" / "model_metadata.json"

st.set_page_config(
    page_title="Bearing Fault Diagnosis | Edge AI Research",
    page_icon="⚙️",
    layout="wide",
)

st.title("Lightweight Edge AI for Bearing Fault Diagnosis")
st.caption("M.Tech research demonstrator · Paderborn primary dataset · CWRU supplementary study")

@st.cache_resource(show_spinner="Loading the packaged Paderborn model...")
def load_model():
    if not MODEL_B64.exists():
        raise FileNotFoundError(f"Model artefact not found: {MODEL_B64}")
    model_bytes = base64.b64decode(MODEL_B64.read_text(encoding="ascii").strip())
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as handle:
            handle.write(model_bytes)
            temp_path = Path(handle.name)
        return pbi.load_model(temp_path)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)

def load_metadata():
    return json.loads(MODEL_META.read_text(encoding="utf-8"))

page = st.sidebar.radio(
    "Navigate",
    ["Overview", "Run diagnosis", "Experimental results", "Methodology"],
)

if page == "Overview":
    st.header("Project overview")
    st.write(
        "This project studies vibration-based rolling-bearing fault diagnosis. "
        "The research focuses on extracting useful signal information, comparing compact "
        "classifiers and sequence models, and evaluating whether predictions transfer to "
        "physical bearings that were not used for training."
    )
    a, b, c, d = st.columns(4)
    a.metric("Paderborn bearing folders", "32")
    b.metric("Usable V2 feature rows", "2,559")
    c.metric("V2 sequence length", "5 recordings")
    d.metric("Packaged baseline features", "34")
    st.subheader("The research workflow")
    st.code(
        "Paderborn vibration recordings\n"
        "        ↓\n"
        "MATLAB structure inspection and signal extraction\n"
        "        ↓\n"
        "Feature extraction and analysis\n"
        "        ↓\n"
        "Bearing-level train / validation / test split\n"
        "        ↓\n"
        "Classical ML and sequence-model experiments\n"
        "        ↓\n"
        "Generalisation analysis and compact-inference direction",
        language="text",
    )
    left, right = st.columns(2)
    with left:
        st.subheader("V1 — Initial experiments")
        st.write(
            "The first version established the signal-processing and model-training pipeline. "
            "Random sample-level splitting could place measurements from the same physical "
            "bearing in both training and test sets, making the evaluation easier than the "
            "intended unseen-bearing problem."
        )
    with right:
        st.subheader("V2 — Stronger evaluation design")
        st.write(
            "The second version kept each physical bearing in exactly one split, fitted "
            "preprocessing on training data only, retained bearing identity and chronology, "
            "and studied the difference between binary detection and detailed fault-type diagnosis."
        )
    st.warning(
        "Deployment scope: this app is intended to be hosted as a public Streamlit presentation "
        "demo. It is not a physical Raspberry Pi deployment, live sensor connection, or edge-device benchmark."
    )

elif page == "Run diagnosis":
    st.header("Interactive Paderborn prediction")
    st.write(
        "Upload one Paderborn MATLAB recording. The app extracts the same 34 features as the "
        "packaged binary baseline and predicts Healthy or Damaged."
    )
    st.info(
        "Input requirements: a Paderborn .mat file containing the named vibration_1 and speed "
        "channels. The inference module retains the project's nominal 64 kHz assumption and "
        "requires at least 64,000 vibration samples."
    )
    uploaded = st.file_uploader("Choose one Paderborn .mat recording", type=["mat"])
    if uploaded is None:
        st.info("Upload a recording to enable prediction.")
    else:
        st.caption(f"{uploaded.name} · {uploaded.size / (1024 * 1024):.2f} MB")
        if st.button("Run diagnosis", type="primary"):
            temp_path = None
            try:
                with st.spinner("Extracting signal features and running inference..."):
                    start = time.perf_counter()
                    with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as handle:
                        handle.write(uploaded.getvalue())
                        temp_path = Path(handle.name)
                    signal, measured_speed = pbi.load_signal_and_rpm(temp_path)
                    model = load_model()
                    result = pbi.predict_from_signal(signal, measured_speed, model)
                    elapsed_ms = (time.perf_counter() - start) * 1000
                st.session_state["prediction"] = {
                    "filename": uploaded.name,
                    "signal": signal,
                    "result": result,
                    "elapsed_ms": elapsed_ms,
                }
            except Exception as exc:
                st.error(f"Could not process this recording: {type(exc).__name__}: {exc}")
                st.caption(
                    "Confirm that the file is a valid Paderborn MAT recording with vibration_1 "
                    "and speed channels and at least 64,000 vibration samples."
                )
            finally:
                if temp_path is not None:
                    temp_path.unlink(missing_ok=True)

    saved = st.session_state.get("prediction")
    if saved:
        result = saved["result"]
        st.divider()
        if result["prediction"] == 1:
            st.error("Prediction: DAMAGED")
        else:
            st.success("Prediction: HEALTHY")
        x1, x2, x3 = st.columns(3)
        x1.metric("Healthy probability", f'{result["healthy_probability"] * 100:.2f}%')
        x2.metric("Damaged probability", f'{result["damaged_probability"] * 100:.2f}%')
        x3.metric("Processing time", f'{saved["elapsed_ms"]:.1f} ms')
        st.caption(
            f'Recording: {saved["filename"]} · {result["sample_count"]:,} samples · '
            f'Mean measured speed: {result["rpm"]:.1f} RPM'
        )
        signal = np.asarray(saved["signal"], dtype=float)
        sample_step = max(1, len(signal) // 5000)
        indices = np.arange(0, len(signal), sample_step)
        times = indices / pbi.FS
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(times, signal[indices], linewidth=0.6)
        ax.set(title="Vibration waveform", xlabel="Time (seconds; nominal 64 kHz)", ylabel="Amplitude")
        ax.grid(alpha=0.2)
        st.pyplot(fig)
        plt.close(fig)
        centered = signal - np.mean(signal)
        spectrum = np.abs(np.fft.rfft(centered))
        frequencies = np.fft.rfftfreq(len(centered), d=1 / pbi.FS)
        mask = frequencies <= 10000
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(frequencies[mask], spectrum[mask], linewidth=0.6)
        ax.set(title="Signal spectrum", xlabel="Frequency (Hz)", ylabel="FFT magnitude")
        ax.grid(alpha=0.2)
        st.pyplot(fig)
        plt.close(fig)
        st.subheader("Extracted 34-feature vector")
        feature_df = pd.DataFrame(
            {"Feature": list(result["features"].keys()), "Value": list(result["features"].values())}
        )
        st.dataframe(feature_df, use_container_width=True, hide_index=True, height=360)
        st.warning(
            "Model probabilities are outputs for this recording, not calibrated guarantees or "
            "maintenance decisions. This predictor is the separate 34-feature binary baseline, "
            "not the V2 sequence-model Stage 1 / Stage 2 system."
        )

elif page == "Experimental results":
    st.header("Experimental results")
    st.write("The project contains different experiments. Their models and metrics are kept separate.")
    st.subheader("Packaged Paderborn 34-feature baseline")
    metadata = load_metadata()
    k1, k2, k3 = st.columns(3)
    k1.metric("Reported accuracy", "99.06%")
    k2.metric("Balanced accuracy", "99.38%")
    k3.metric("ROC AUC", "1.00")
    st.write(
        f'Model: {metadata["scaling"]} + {metadata["classifier"]}. '
        f'Training set: {metadata["training_recordings"]:,} recordings from '
        f'{metadata["training_bearings"]} bearings.'
    )
    st.caption(
        "The reported 320-sample evaluation involved four physical bearings and was reused during "
        "iterative experimentation. It is not a pristine untouched external test."
    )
    st.subheader("Paderborn V2 sequence-model experiments")
    st.write("Stage 1 asks Healthy vs Damaged; Stage 2 asks Outer vs Inner vs Both.")
    st.dataframe(
        pd.DataFrame({
            "Stage 1 model": ["LSTM", "GRU", "Transformer", "Compact 1D CNN"],
            "Reported test accuracy (%)": [100.0, 100.0, 94.375, 100.0],
        }),
        use_container_width=True,
        hide_index=True,
    )
    st.dataframe(
        pd.DataFrame({
            "Stage 2 model": ["LSTM", "Transformer", "Compact 1D CNN"],
            "Reported test accuracy (%)": [43.75, 45.70, 53.52],
            "Reported macro-F1 (%)": [28.34, 35.52, 41.57],
        }),
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        "These are retrospective sequence-model results from a separate V2 pipeline. The Both "
        "class was difficult to generalise, and the final test subset had been seen during earlier "
        "project experimentation."
    )
    st.subheader("CWRU supplementary 48 kHz study")
    st.dataframe(
        pd.DataFrame({
            "Feature group": ["Time only", "Frequency only", "Envelope only", "All 37 features"],
            "Feature count": [9, 13, 15, 37],
            "LOLO pooled accuracy (%)": [97.86, 100.0, 100.0, 100.0],
        }),
        use_container_width=True,
        hide_index=True,
    )
    st.warning(
        "The selected CWRU recordings share fault-condition series across motor loads. These "
        "results do not establish generalisation to independent physical bearing specimens."
    )

else:
    st.header("Methodology")
    st.subheader("1. Dataset verification")
    st.write(
        "The Paderborn files are MATLAB .mat files with nested structures and multiple channels. "
        "The loader inspects the structure and finds the channel named vibration_1 instead of "
        "assuming a fixed channel position. A separate speed channel supplies measured RPM."
    )
    st.subheader("2. Feature extraction")
    st.write(
        "The V2 sequence-model study began with 26 general statistical and spectral features, "
        "removed four redundant features, and retained 22. The packaged binary baseline is a "
        "separate 34-feature pipeline: those 22 general features plus six BPFO and six BPFI "
        "envelope-energy features."
    )
    st.subheader("3. Bearing-level evaluation")
    st.write(
        "V2 used 22 training bearings, 5 validation bearings and 5 final test bearings. All "
        "measurements from one physical bearing were kept within one partition. Five consecutive "
        "recordings were combined into each sequence."
    )
    st.subheader("4. Two-stage diagnosis")
    st.write(
        "Stage 1 detects Healthy vs Damaged. Stage 2 attempts to identify Outer, Inner or Both "
        "faults among damaged samples. Stage 2 was substantially harder, especially for the "
        "combined Both class."
    )
    st.subheader("5. Intended lightweight direction")
    st.write(
        "The planned next steps include stronger bearing-level cross-validation, teacher selection, "
        "a compact TCN student, knowledge distillation and model compression. Do not describe "
        "these planned stages as completed results."
    )

st.divider()
st.caption(
    "Academic research demonstrator. Dataset-specific results and preprocessing assumptions apply. "
    "No raw dataset is bundled or committed by this application."
)
