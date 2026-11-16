
"""
Paderborn Edge Replay Application

Simulates chunk-based vibration input and performs
bearing fault detection using the packaged baseline.

This is accelerated offline replay, not a real-time
sensor or Raspberry Pi hardware implementation.
"""

import argparse
import base64
import json
import tempfile
import time

from pathlib import Path

import numpy as np

import paderborn_inference as pbi


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "results" / "edge_replay"
)


# ============================================================
# 2. SIGNAL BUFFER
# ============================================================

class SignalBuffer:
    """
    Accumulates incoming signal chunks and reconstructs
    the complete vibration signal.
    """

    def __init__(self):

        self.chunks = []
        self.total_samples = 0
        self.closed = False

    def add_chunk(self, chunk):

        if self.closed:
            raise RuntimeError(
                "Cannot add data to a finalized buffer."
            )

        chunk = np.asarray(
            chunk,
            dtype=np.float64
        ).reshape(-1)

        if chunk.size == 0:
            return

        if not np.all(np.isfinite(chunk)):
            raise ValueError(
                "Invalid values in incoming signal chunk."
            )

        self.chunks.append(chunk.copy())

        self.total_samples += len(chunk)

    def finish(self):

        if self.closed:
            raise RuntimeError(
                "Buffer has already been finalized."
            )

        if not self.chunks:
            raise ValueError(
                "No signal chunks were received."
            )

        complete_signal = np.concatenate(
            self.chunks
        )

        self.closed = True

        return complete_signal


# ============================================================
# 3. COMMAND-LINE ARGUMENTS
# ============================================================

def parse_arguments():

    parser = argparse.ArgumentParser(
        description=(
            "Paderborn bearing fault detection "
            "using simulated signal replay."
        )
    )

    parser.add_argument(
        "--mat-file",
        required=True,
        help=(
            "Path to the Paderborn MAT file. "
            "Relative paths are resolved from project root."
        )
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=4096,
        help="Number of vibration samples per chunk."
    )

    return parser.parse_args()


# ============================================================
# 4. MAIN APPLICATION
# ============================================================

def main():

    args = parse_arguments()

    if args.chunk_size <= 0:
        raise ValueError(
            "Chunk size must be greater than zero."
        )

    print("=" * 70)
    print("PADERBORN BEARING FAULT DETECTION")
    print("SIMULATED EDGE REPLAY")
    print("=" * 70)

    # --------------------------------------------------------
    # A. Resolve input file
    # --------------------------------------------------------

    mat_path = Path(args.mat_file)

    if not mat_path.is_absolute():
        mat_path = PROJECT_ROOT / mat_path

    mat_path = mat_path.resolve()

    if not mat_path.exists():
        raise FileNotFoundError(
            f"MAT file not found: {mat_path}"
        )

    print("\nInput recording:")
    print(mat_path)

    # --------------------------------------------------------
    # B. Load packaged model
    # --------------------------------------------------------

    model_start = time.perf_counter()

    encoded_model_path = (
        PROJECT_ROOT
        / "models"
        / "paderborn_baseline"
        / "paderborn_baseline_pipeline.joblib.b64"
    )

    if not encoded_model_path.exists():
        raise FileNotFoundError(
            f"Encoded model artefact not found: {encoded_model_path}"
        )

    model_bytes = base64.b64decode(
        encoded_model_path.read_text(encoding="ascii").strip(),
        validate=True
    )

    temporary_model_path = None

    try:
        with tempfile.NamedTemporaryFile(
            suffix=".joblib",
            delete=False
        ) as model_file:
            model_file.write(model_bytes)
            temporary_model_path = Path(model_file.name)

        model = pbi.load_model(temporary_model_path)

    finally:
        if temporary_model_path is not None:
            temporary_model_path.unlink(missing_ok=True)

    model_load_time = (
        time.perf_counter() - model_start
    )

    print("\nPackaged model loaded.")

    print(
        "Model input features:",
        model.n_features_in_
    )

    # --------------------------------------------------------
    # C. Load original MAT recording
    # --------------------------------------------------------

    load_start = time.perf_counter()

    vibration_signal, measured_speed = (
        pbi.load_signal_and_rpm(mat_path)
    )

    mat_load_time = (
        time.perf_counter() - load_start
    )

    print("\nSource signal information:")
    print("-" * 55)

    print(
        "Vibration samples:",
        len(vibration_signal)
    )

    print(
        "Measured RPM:",
        f"{np.mean(measured_speed):.3f}"
    )

    # --------------------------------------------------------
    # D. Simulate chunk-based input
    # --------------------------------------------------------

    buffer = SignalBuffer()

    expected_chunks = int(
        np.ceil(
            len(vibration_signal) / args.chunk_size
        )
    )

    print("\nStarting simulated signal replay...")
    print("-" * 55)

    replay_start = time.perf_counter()

    for start in range(
        0,
        len(vibration_signal),
        args.chunk_size
    ):

        end = min(
            start + args.chunk_size,
            len(vibration_signal)
        )

        incoming_chunk = vibration_signal[start:end]

        buffer.add_chunk(incoming_chunk)

    buffered_signal = buffer.finish()

    buffering_time = (
        time.perf_counter() - replay_start
    )

    # --------------------------------------------------------
    # E. Verify buffer integrity
    # --------------------------------------------------------

    buffer_integrity = np.array_equal(
        vibration_signal,
        buffered_signal
    )

    if not buffer_integrity:

        raise RuntimeError(
            "Signal reconstruction failed."
        )

    print("\nBuffer status:")
    print("-" * 55)

    print(
        "Expected chunks:",
        expected_chunks
    )

    print(
        "Received chunks:",
        len(buffer.chunks)
    )

    print(
        "Total received samples:",
        buffer.total_samples
    )

    print(
        "Buffer integrity:",
        buffer_integrity
    )

    # --------------------------------------------------------
    # F. Perform prediction
    # --------------------------------------------------------

    inference_start = time.perf_counter()

    result = pbi.predict_from_signal(
        vibration_signal=buffered_signal,
        measured_speed=measured_speed,
        model=model
    )

    inference_time = (
        time.perf_counter() - inference_start
    )

    # --------------------------------------------------------
    # G. Display prediction
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("BEARING CONDITION PREDICTION")
    print("=" * 70)

    print(
        "Recording:",
        mat_path.name
    )

    print(
        "Classification:",
        result["classification"]
    )

    print(
        "Healthy probability:",
        f"{result['healthy_probability']:.8f}"
    )

    print(
        "Damaged probability:",
        f"{result['damaged_probability']:.8f}"
    )

    print(
        "Measured RPM:",
        f"{result['rpm']:.3f}"
    )

    # --------------------------------------------------------
    # H. Processing performance
    # --------------------------------------------------------

    total_processing_time = (
        model_load_time
        + mat_load_time
        + buffering_time
        + inference_time
    )

    print("\n" + "=" * 70)
    print("PROCESSING PERFORMANCE")
    print("=" * 70)

    print(
        f"Model loading:       "
        f"{model_load_time * 1000:.3f} ms"
    )

    print(
        f"MAT file loading:    "
        f"{mat_load_time * 1000:.3f} ms"
    )

    print(
        f"Signal buffering:    "
        f"{buffering_time * 1000:.3f} ms"
    )

    print(
        f"Feature extraction + "
        f"inference: {inference_time * 1000:.3f} ms"
    )

    print(
        f"Total:               "
        f"{total_processing_time * 1000:.3f} ms"
    )

    # --------------------------------------------------------
    # I. Save inference result
    # --------------------------------------------------------

    DEFAULT_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    report = {

        "recording": mat_path.name,

        "classification": result["classification"],

        "prediction": result["prediction"],

        "healthy_probability": (
            result["healthy_probability"]
        ),

        "damaged_probability": (
            result["damaged_probability"]
        ),

        "measured_rpm": result["rpm"],

        "sample_count": result["sample_count"],

        "chunk_size": args.chunk_size,

        "chunks_received": len(buffer.chunks),

        "buffer_integrity": bool(buffer_integrity),

        "timing_ms": {

            "model_loading": model_load_time * 1000,

            "mat_loading": mat_load_time * 1000,

            "buffering": buffering_time * 1000,

            "inference": inference_time * 1000,

            "total": total_processing_time * 1000
        },

        "execution_mode": "accelerated_offline_replay"

    }

    output_path = (
        DEFAULT_OUTPUT_DIR
        / f"{mat_path.stem}_result.json"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            report,
            file,
            indent=4
        )

    print("\nInference report saved:")
    print(output_path)

    print("\n" + "=" * 70)
    print("EDGE REPLAY COMPLETED SUCCESSFULLY")
    print("=" * 70)


# ============================================================
# 5. APPLICATION ENTRY POINT
# ============================================================

if __name__ == "__main__":

    try:
        main()

    except Exception as error:

        print("\nAPPLICATION ERROR:")
        print(f"{type(error).__name__}: {error}")

        raise