from __future__ import annotations
import argparse, os, json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, f1_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
import joblib
from .data_simulator import simulate_dataset
from .features import FEATURES

ARTIFACTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

def build_model(preset: str):
    preset = preset.lower()
    if preset == "baseline":
        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_split=2,
            n_jobs=-1,
            random_state=42,
        )
    elif preset == "best":
        hgb = HistGradientBoostingClassifier(
            max_depth=None,
            learning_rate=0.07,
            max_iter=350,
            random_state=42,
        )
        rf = RandomForestClassifier(
            n_estimators=250,
            random_state=42,
            n_jobs=-1,
        )
        meta = LogisticRegression(max_iter=1000)
        model = StackingClassifier(
            estimators=[("hgb", hgb), ("rf", rf)],
            final_estimator=meta,
            passthrough=False,
            n_jobs=-1,
        )
    else:
        raise ValueError("preset must be 'baseline' or 'best'")
    return model

def main(rows: int, preset: str):
    print(f"Simulating {rows} rows...")
    df = simulate_dataset(n_rows=rows, reel_ratio=0.5)

    X = df[FEATURES]
    y = df["label"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = build_model(preset)
    print(f"Training model preset: {preset}")
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    yhat = (proba >= 0.5).astype(int)

    metrics = {
        "preset": preset,
        "rows": rows,
        "accuracy": float(accuracy_score(y_test, yhat)),
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "f1": float(f1_score(y_test, yhat)),
        "report": classification_report(y_test, yhat, target_names=["non_reel", "reel"]),
    }
    print("\n=== Metrics ===")
    for k in ["accuracy", "roc_auc", "f1"]:
        print(f"{k}: {metrics[k]:.4f}")
    print(metrics["report"])

    # Save artifacts
    joblib.dump(model, os.path.join(ARTIFACTS_DIR, "model.joblib"))
    with open(os.path.join(ARTIFACTS_DIR, "feature_config.json"), "w") as f:
        json.dump({"feature_order": FEATURES}, f, indent=2)
    with open(os.path.join(ARTIFACTS_DIR, "label_map.json"), "w") as f:
        json.dump({"0": "non_reel", "1": "reel"}, f, indent=2)
    with open(os.path.join(ARTIFACTS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved artifacts to: {ARTIFACTS_DIR}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=50000)
    parser.add_argument("--preset", type=str, default="baseline", choices=["baseline", "best"])
    args = parser.parse_args()
    main(rows=args.rows, preset=args.preset)
