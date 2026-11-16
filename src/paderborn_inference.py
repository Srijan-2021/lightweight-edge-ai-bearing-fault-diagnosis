
"""
Paderborn Bearing Fault Detection - Inference Module

This module reproduces the original 34-feature Logistic Regression
baseline developed in MTECH_PROJECT_V2.

Feature composition:
    22 statistical and spectral features (complete signal)
     6 BPFO features (first 64,000 samples)
     6 BPFI features (first 64,000 samples)

Classes:
    0 = Healthy
    1 = Damaged

Important:
    This deliberately preserves the original 64 kHz assumption and
    timestamp-independent feature calculations. It does not perform
    timestamp correction or resampling.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from scipy.io import loadmat
from scipy.stats import kurtosis, skew
from scipy.signal import butter, sosfiltfilt, hilbert
from scipy.fft import rfft, rfftfreq


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "paderborn_baseline"
    / "paderborn_baseline_pipeline.joblib"
)

FS = 64000

LOWCUT = 2000
HIGHCUT = 10000

BPFO_MAX_ENVELOPE_FREQ = 500
BPFI_MAX_ENVELOPE_FREQ = 700

HARMONIC_TOLERANCE = 3

# Bearing geometry: 6203
N_BALLS = 8
BALL_DIAMETER = 6.75
PITCH_DIAMETER = 29.05
PRESSURE_ANGLE_DEG = 0

COS_THETA = np.cos(
    np.deg2rad(PRESSURE_ANGLE_DEG)
)


# ============================================================
# 2. MODEL FEATURE ORDER
# ============================================================

FEATURES_22 = [
    "mean",
    "std",
    "min",
    "max",
    "kurtosis",
    "skewness",
    "crest_factor",
    "dominant_frequency",
    "dominant_magnitude",
    "spectral_centroid",
    "spectral_bandwidth",
    "band_power_0_500",
    "band_power_500_1000",
    "band_power_1000_2000",
    "band_power_2000_3000",
    "band_power_3000_4000",
    "band_power_4000_5000",
    "band_power_5000_6000",
    "band_power_6000_7000",
    "band_power_7000_8000",
    "band_power_8000_9000",
    "band_power_9000_10000",
]

BPFO_FEATURES = [
    f"bpfo_{h}x_energy_ratio"
    for h in range(1, 6)
] + ["bpfo_1to5_total"]

BPFI_FEATURES = [
    f"bpfi_{h}x_energy_ratio"
    for h in range(1, 6)
] + ["bpfi_1to5_total"]

FEATURE_NAMES = (
    FEATURES_22
    + BPFO_FEATURES
    + BPFI_FEATURES
)

assert len(FEATURE_NAMES) == 34


# ============================================================
# 3. LOAD THE TRAINED MODEL
# ============================================================

def load_model(model_path=None):
    """
    Load the packaged Logistic Regression pipeline.

    The function verifies the input feature count and,
    where available, the exact feature order.
    """

    if model_path is None:
        model_path = DEFAULT_MODEL_PATH

    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}"
        )

    model = joblib.load(model_path)

    if model.n_features_in_ != len(FEATURE_NAMES):
        raise ValueError(
            f"Model expects {model.n_features_in_} features, "
            f"but this module defines {len(FEATURE_NAMES)}."
        )

    if hasattr(model, "feature_names_in_"):

        model_features = list(model.feature_names_in_)

        if model_features != FEATURE_NAMES:

            raise ValueError(
                "Model feature order does not match "
                "the inference module."
            )

    if set(model.classes_) != {0, 1}:
        raise ValueError(
            f"Unexpected model classes: {model.classes_}"
        )

    return model


# ============================================================
# 4. EXTRACT CHANNEL FROM MATLAB STRUCTURE
# ============================================================

def extract_channel(data, channel_name):
    """
    Extract a named Y channel from a Paderborn MAT structure.
    """

    y_channels = np.asarray(data.Y).flatten()

    for channel in y_channels:

        name = channel.Name

        if isinstance(name, np.ndarray):

            name = name.squeeze()

            if name.size == 1:
                name = name.item()
            else:
                name = "".join(
                    map(str, name.flatten())
                )

        if isinstance(name, bytes):
            name = name.decode(
                "utf-8",
                errors="replace"
            )

        if str(name).strip() == channel_name:

            signal = np.asarray(
                channel.Data
            ).squeeze().astype(np.float64)

            return signal.reshape(-1)

    raise ValueError(
        f"Channel '{channel_name}' not found."
    )


# ============================================================
# 5. LOAD VIBRATION AND MEASURED SPEED FROM MAT FILE
# ============================================================

def load_signal_and_rpm(mat_file):
    """
    Load vibration_1 and measured speed from a Paderborn MAT file.

    Returns:
        vibration_signal, measured_speed_array
    """

    mat_file = Path(mat_file)

    if not mat_file.exists():
        raise FileNotFoundError(mat_file)

    mat = loadmat(
        mat_file,
        squeeze_me=False,
        struct_as_record=False
    )

    data_keys = [
        key for key in mat.keys()
        if not key.startswith("__")
    ]

    if len(data_keys) != 1:
        raise ValueError(
            f"Unexpected MATLAB structure: {mat_file}"
        )

    measurement = np.asarray(
        mat[data_keys[0]],
        dtype=object
    )

    if measurement.size != 1:
        raise ValueError(
            f"Unexpected MATLAB measurement dimensions: "
            f"{measurement.shape}"
        )

    data = measurement.reshape(-1)[0]

    vibration_signal = extract_channel(
        data,
        "vibration_1"
    )

    speed = extract_channel(
        data,
        "speed"
    )

    return vibration_signal, speed


# ============================================================
# 6. ORIGINAL 22 STATISTICAL AND SPECTRAL FEATURES
# ============================================================

def extract_vibration_features(signal, fs=FS):
    """
    Extract the original 26 features from the complete signal.

    The inference pipeline subsequently selects the original
    22 features used by the trained model.
    """

    signal = np.asarray(
        signal,
        dtype=np.float64
    ).flatten()

    if len(signal) == 0:
        raise ValueError("Signal is empty.")

    if not np.all(np.isfinite(signal)):
        raise ValueError(
            "Signal contains NaN or infinite values."
        )

    # --------------------------------------------------------
    # Time-domain features
    # --------------------------------------------------------

    mean_value = np.mean(signal)
    std_value = np.std(signal)

    rms_value = np.sqrt(
        np.mean(signal ** 2)
    )

    min_value = np.min(signal)
    max_value = np.max(signal)

    peak_to_peak = max_value - min_value

    kurtosis_value = kurtosis(
        signal,
        fisher=True,
        bias=False
    )

    skewness_value = skew(
        signal,
        bias=False
    )

    crest_factor = (
        np.max(np.abs(signal)) / rms_value
        if rms_value != 0 else 0
    )

    # --------------------------------------------------------
    # Frequency-domain features
    # --------------------------------------------------------

    n = len(signal)

    centered_signal = signal - mean_value

    fft_values = np.fft.rfft(centered_signal)

    frequencies = np.fft.rfftfreq(
        n,
        d=1 / fs
    )

    magnitude = np.abs(fft_values)

    magnitude_no_dc = magnitude.copy()

    if len(magnitude_no_dc) > 1:
        magnitude_no_dc[0] = 0

    dominant_index = np.argmax(
        magnitude_no_dc
    )

    dominant_frequency = frequencies[
        dominant_index
    ]

    dominant_magnitude = magnitude[
        dominant_index
    ]

    power = magnitude ** 2

    total_power = np.sum(power)

    if total_power > 0:

        spectral_centroid = (
            np.sum(frequencies * power)
            / total_power
        )

        spectral_bandwidth = np.sqrt(
            np.sum(
                ((frequencies - spectral_centroid) ** 2)
                * power
            ) / total_power
        )

        probability = power / total_power

        spectral_entropy = -np.sum(
            probability[probability > 0]
            * np.log2(probability[probability > 0])
        )

    else:

        spectral_centroid = 0.0
        spectral_bandwidth = 0.0
        spectral_entropy = 0.0

    spectral_rms = np.sqrt(
        np.mean(power)
    )

    # --------------------------------------------------------
    # Frequency band powers
    # --------------------------------------------------------

    bands = [
        (0, 500),
        (500, 1000),
        (1000, 2000),
        (2000, 3000),
        (3000, 4000),
        (4000, 5000),
        (5000, 6000),
        (6000, 7000),
        (7000, 8000),
        (8000, 9000),
        (9000, 10000)
    ]

    features = {
        "mean": mean_value,
        "std": std_value,
        "rms": rms_value,
        "min": min_value,
        "max": max_value,
        "peak_to_peak": peak_to_peak,
        "kurtosis": kurtosis_value,
        "skewness": skewness_value,
        "crest_factor": crest_factor,
        "dominant_frequency": dominant_frequency,
        "dominant_magnitude": dominant_magnitude,
        "spectral_centroid": spectral_centroid,
        "spectral_bandwidth": spectral_bandwidth,
        "spectral_rms": spectral_rms,
        "spectral_entropy": spectral_entropy
    }

    for low, high in bands:

        mask = (
            (frequencies >= low)
            & (frequencies < high)
        )

        features[
            f"band_power_{low}_{high}"
        ] = np.sum(power[mask])

    return features


# ============================================================
# 7. ENVELOPE SPECTRUM
# ============================================================

def envelope_spectrum(
    signal,
    fs,
    lowcut,
    highcut,
    max_freq
):
    """
    Reproduce the original envelope-spectrum calculation.

    Only the first fs samples are used.
    At 64 kHz, this represents approximately one second.
    """

    signal = np.asarray(
        signal,
        dtype=np.float64
    ).flatten()

    if len(signal) < fs:
        raise ValueError(
            f"At least {fs} samples are needed for "
            "the one-second envelope analysis."
        )

    # Preserve original one-second segment definition.
    signal = signal[:fs].copy()

    signal = signal - np.mean(signal)

    sos = butter(
        4,
        [lowcut, highcut],
        btype="bandpass",
        fs=fs,
        output="sos"
    )

    filtered = sosfiltfilt(
        sos,
        signal
    )

    analytic_signal = hilbert(filtered)

    envelope = np.abs(analytic_signal)

    envelope = envelope - np.mean(envelope)

    spectrum = np.abs(
        rfft(envelope)
    )

    frequencies = rfftfreq(
        len(envelope),
        d=1 / fs
    )

    mask = frequencies <= max_freq

    return frequencies[mask], spectrum[mask]


# ============================================================
# 8. BEARING FAULT FREQUENCIES
# ============================================================

def calculate_fault_frequencies(rpm):
    """
    Calculate shaft, BPFO, BPFI, FTF and BSF frequencies.
    """

    rpm = float(rpm)

    if not np.isfinite(rpm) or rpm <= 0:
        raise ValueError(
            f"Invalid RPM: {rpm}"
        )

    fr = rpm / 60.0

    bpfo = (
        N_BALLS / 2
        * fr
        * (
            1
            - (BALL_DIAMETER / PITCH_DIAMETER)
            * COS_THETA
        )
    )

    bpfi = (
        N_BALLS / 2
        * fr
        * (
            1
            + (BALL_DIAMETER / PITCH_DIAMETER)
            * COS_THETA
        )
    )

    ftf = (
        0.5
        * fr
        * (
            1
            - (BALL_DIAMETER / PITCH_DIAMETER)
            * COS_THETA
        )
    )

    bsf = (
        PITCH_DIAMETER / (2 * BALL_DIAMETER)
        * fr
        * (
            1
            - (BALL_DIAMETER / PITCH_DIAMETER) ** 2
            * COS_THETA ** 2
        )
    )

    return fr, bpfo, bpfi, ftf, bsf


# ============================================================
# 9. BPFO ENERGY FEATURES
# ============================================================

def extract_bpfo_energy_features(signal, rpm):

    frequencies, spectrum = envelope_spectrum(
        signal,
        FS,
        LOWCUT,
        HIGHCUT,
        BPFO_MAX_ENVELOPE_FREQ
    )

    spectral_power = spectrum ** 2

    total_band = (
        (frequencies >= 20)
        & (frequencies <= BPFO_MAX_ENVELOPE_FREQ)
    )

    total_energy = np.sum(
        spectral_power[total_band]
    )

    _, bpfo, _, _, _ = calculate_fault_frequencies(rpm)

    features = {}

    for harmonic in range(1, 6):

        target = harmonic * bpfo

        band = (
            np.abs(frequencies - target)
            <= HARMONIC_TOLERANCE
        )

        harmonic_energy = np.sum(
            spectral_power[band]
        )

        features[
            f"bpfo_{harmonic}x_energy_ratio"
        ] = (
            harmonic_energy
            / (total_energy + 1e-12)
        )

    features["bpfo_1to5_total"] = sum(
        features[
            f"bpfo_{harmonic}x_energy_ratio"
        ]
        for harmonic in range(1, 6)
    )

    return features


# ============================================================
# 10. BPFI ENERGY FEATURES
# ============================================================

def extract_bpfi_energy_features(signal, rpm):

    frequencies, spectrum = envelope_spectrum(
        signal,
        FS,
        LOWCUT,
        HIGHCUT,
        BPFI_MAX_ENVELOPE_FREQ
    )

    spectral_power = spectrum ** 2

    total_band = (
        (frequencies >= 20)
        & (frequencies <= BPFI_MAX_ENVELOPE_FREQ)
    )

    total_energy = np.sum(
        spectral_power[total_band]
    )

    _, _, bpfi, _, _ = calculate_fault_frequencies(rpm)

    features = {}

    for harmonic in range(1, 6):

        target = harmonic * bpfi

        band = (
            np.abs(frequencies - target)
            <= HARMONIC_TOLERANCE
        )

        harmonic_energy = np.sum(
            spectral_power[band]
        )

        features[
            f"bpfi_{harmonic}x_energy_ratio"
        ] = (
            harmonic_energy
            / (total_energy + 1e-12)
        )

    features["bpfi_1to5_total"] = sum(
        features[
            f"bpfi_{harmonic}x_energy_ratio"
        ]
        for harmonic in range(1, 6)
    )

    return features


# ============================================================
# 11. EXTRACT ALL 34 FEATURES
# ============================================================

def extract_34_features(vibration_signal, measured_speed):
    """
    Extract all 34 original model features.

    Statistical/spectral features:
        Entire supplied vibration signal.

    BPFO/BPFI features:
        First 64,000 vibration samples.

    measured_speed:
        Scalar RPM or array of measured RPM values.
    """

    signal = np.asarray(
        vibration_signal,
        dtype=np.float64
    ).reshape(-1)

    if signal.size < FS:
        raise ValueError(
            f"Signal needs at least {FS} samples."
        )

    if not np.all(np.isfinite(signal)):
        raise ValueError(
            "Vibration signal contains invalid values."
        )

    speed = np.asarray(
        measured_speed,
        dtype=np.float64
    ).reshape(-1)

    if speed.size == 0 or not np.all(np.isfinite(speed)):
        raise ValueError(
            "Invalid measured speed."
        )

    rpm = float(np.mean(speed))

    if rpm <= 0:
        raise ValueError("RPM must be positive.")

    # Original 26 features, then select 22.
    features_26 = extract_vibration_features(signal)

    features_22 = {
        name: features_26[name]
        for name in FEATURES_22
    }

    bpfo_features = extract_bpfo_energy_features(
        signal,
        rpm
    )

    bpfi_features = extract_bpfi_energy_features(
        signal,
        rpm
    )

    features = {
        **features_22,
        **bpfo_features,
        **bpfi_features
    }

    if list(features.keys()) != FEATURE_NAMES:
        raise RuntimeError(
            "Generated feature order differs from model order."
        )

    feature_values = np.asarray(
        list(features.values()),
        dtype=np.float64
    )

    if not np.all(np.isfinite(feature_values)):
        raise ValueError(
            "Generated features contain NaN or infinity."
        )

    return features


# ============================================================
# 12. PREDICT FROM A RAW VIBRATION SIGNAL
# ============================================================

def predict_from_signal(
    vibration_signal,
    measured_speed,
    model
):
    """
    Predict Healthy/Damaged from raw vibration samples.

    Load the model once and reuse it for multiple predictions.
    """

    features = extract_34_features(
        vibration_signal,
        measured_speed
    )

    X = pd.DataFrame(
        [[features[name] for name in FEATURE_NAMES]],
        columns=FEATURE_NAMES
    )

    prediction = int(
        model.predict(X)[0]
    )

    probabilities = model.predict_proba(X)[0]

    classes = list(model.classes_)

    healthy_index = classes.index(0)
    damaged_index = classes.index(1)

    return {

        "prediction": prediction,

        "classification": (
            "Damaged" if prediction == 1 else "Healthy"
        ),

        "healthy_probability": float(
            probabilities[healthy_index]
        ),

        "damaged_probability": float(
            probabilities[damaged_index]
        ),

        "rpm": float(np.mean(measured_speed)),

        "sample_count": len(vibration_signal),

        "features": features,

        "feature_vector": X
    }


# ============================================================
# 13. PREDICT DIRECTLY FROM A PADERBORN MAT FILE
# ============================================================

def predict_from_mat(mat_file, model):
    """
    Load a Paderborn MAT file and predict its bearing condition.
    """

    vibration_signal, speed = load_signal_and_rpm(
        mat_file
    )

    result = predict_from_signal(
        vibration_signal,
        speed,
        model
    )

    result["file"] = Path(mat_file).name

    result["path"] = str(
        Path(mat_file).resolve()
    )

    return result