from __future__ import annotations
import json
import os
import joblib
from typing import Dict, List, Any
from .features import FEATURES, to_dataframe

ARTIFACTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "artifacts")
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "model.joblib")
FEATURE_CONFIG_PATH = os.path.join(ARTIFACTS_DIR, "feature_config.json")
LABEL_MAP_PATH = os.path.join(ARTIFACTS_DIR, "label_map.json")

def ensure_artifacts() -> None:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Train a model first: "
            f"`python -m src.train --rows 50000 --preset baseline`"
        )
    if not os.path.exists(FEATURE_CONFIG_PATH):
        raise FileNotFoundError("feature_config.json missing – train again.")
    if not os.path.exists(LABEL_MAP_PATH):
        raise FileNotFoundError("label_map.json missing – train again.")

def load_artifacts():
    ensure_artifacts()
    model = joblib.load(MODEL_PATH)
    with open(FEATURE_CONFIG_PATH, "r") as f:
        feat_cfg = json.load(f)
    with open(LABEL_MAP_PATH, "r") as f:
        label_map = json.load(f)
    return model, feat_cfg, label_map

def predict_from_records(model, records: List[Dict[str, Any]]):
    df = to_dataframe(records)
    proba = model.predict_proba(df)[:, 1]
    yhat = (proba >= 0.5).astype(int)
    return yhat, proba, df
