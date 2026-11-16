"""Smoke test for the packaged Paderborn inference baseline."""
import base64
import tempfile
from pathlib import Path

import numpy as np

from src import paderborn_inference as pbi

ROOT = Path(__file__).resolve().parent
encoded_model = (
    ROOT
    / "models"
    / "paderborn_baseline"
    / "paderborn_baseline_pipeline.joblib.b64"
)
model_bytes = base64.b64decode(
    encoded_model.read_text(encoding="ascii").strip(),
    validate=True,
)

temporary_model_path = None
try:
    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as handle:
        handle.write(model_bytes)
        temporary_model_path = Path(handle.name)

    model = pbi.load_model(temporary_model_path)
    assert model.n_features_in_ == 34, model.n_features_in_
    assert set(model.classes_) == {0, 1}, model.classes_

    # Synthetic input checks that feature extraction and prediction execute.
    # This is a software smoke test, not an accuracy or real-data validation.
    n_samples = pbi.FS
    time_axis = np.arange(n_samples, dtype=np.float64) / pbi.FS
    rng = np.random.default_rng(2026)
    synthetic_signal = (
        0.5 * np.sin(2 * np.pi * 3000 * time_axis)
        + 0.15 * np.sin(2 * np.pi * 5000 * time_axis)
        + 0.02 * rng.normal(size=n_samples)
    )
    result = pbi.predict_from_signal(
        synthetic_signal,
        np.full(100, 1500.0),
        model,
    )
    assert len(result["features"]) == 34
    assert result["classification"] in {"Healthy", "Damaged"}
    print("PASS: model loaded and 34-feature synthetic-signal inference completed.")
finally:
    if temporary_model_path is not None:
        temporary_model_path.unlink(missing_ok=True)
