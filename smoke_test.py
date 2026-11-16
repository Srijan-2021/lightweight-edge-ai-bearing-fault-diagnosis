"""Smoke test for the packaged Paderborn baseline and feature schema."""
import base64
import tempfile
from pathlib import Path

from src import paderborn_inference as pbi

ROOT = Path(__file__).resolve().parent
encoded_model = ROOT / "models" / "paderborn_baseline" / "paderborn_baseline_pipeline.joblib.b64"
model_bytes = base64.b64decode(encoded_model.read_text(encoding="ascii").strip())

temp_path = None
try:
    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as handle:
        handle.write(model_bytes)
        temp_path = Path(handle.name)
    model = pbi.load_model(temp_path)
    assert model.n_features_in_ == 34, model.n_features_in_
    assert set(model.classes_) == {0, 1}, model.classes_
    print("PASS: model loads, expects 34 features, and contains Healthy/Damaged classes.")
finally:
    if temp_path is not None:
        temp_path.unlink(missing_ok=True)
