"""
Retrain + shadow-deploy pipeline. Responsibilities:
  1. Pull recent, LABELED predictions from prediction_logs — this is real
     live data with known correct answers, not the original training set.
  2. Train a new challenger model on it.
  3. Save it as the next version (model_v2.joblib, model_v3.joblib, ...).
  4. Tell the serving API to run this new version in shadow mode via
     POST /admin/set_shadow.

This NEVER touches production. It has no code path that could promote a
model to serve live traffic — that only happens in step 4, and only after
a human approves it.

Run:
    python retrain/retrain_pipeline.py --n-recent 500
"""
import argparse
import json
import os
import re
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

from serving.db import SessionLocal, PredictionLog
from data.house_price_preprocessor import HousePricePreprocessor, TARGET_COL
from models.train_baseline import FINAL_GB_PARAMS

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
API_URL = os.environ.get("SERVING_API_URL", "http://localhost:8000")


def get_next_version():
    """Scans models/ for model_v<N>.joblib and returns the next version name."""
    existing = [p.stem for p in MODELS_DIR.glob("model_v*.joblib")]
    numbers = [int(m.group(1)) for name in existing if (m := re.match(r"model_v(\d+)", name))]
    next_n = (max(numbers) + 1) if numbers else 2
    return f"v{next_n}"


def load_recent_labeled_data(n_recent):
    """
    Pulls the most recent N *labeled* production predictions — real live
    data with known true outcomes — to train the challenger on. This is
    deliberately NOT the original training_data.csv: the whole point is to
    learn from the data distribution the model is actually seeing now.
    """
    db = SessionLocal()
    try:
        rows = (
            db.query(PredictionLog)
            .filter(
                PredictionLog.is_shadow == False,  # noqa: E712
                PredictionLog.true_label.isnot(None),
            )
            .order_by(PredictionLog.id.desc())
            .limit(n_recent)
            .all()
        )
    finally:
        db.close()

    if not rows:
        return None

    records = []
    for r in rows:
        row = dict(r.features)
        row[TARGET_COL] = r.true_label
        records.append(row)
    return pd.DataFrame(records)


def train_challenger(df, version_name):
    """
    Fits a FRESH preprocessor on this recent live data (not reusing the
    original v1 preprocessor) — the whole point of retraining is to adapt
    to whatever the data looks like now, including any new category values
    or shifted medians drift may have introduced. Uses the same tuned
    GradientBoostingRegressor configuration and log1p target transform as
    the original notebook, so the challenger is comparable apples-to-apples
    with production.
    """
    y = df[TARGET_COL]
    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        df, y, test_size=0.2, random_state=0
    )

    preprocessor = HousePricePreprocessor()
    preprocessor.fit(X_train_raw)

    X_train = preprocessor.transform(X_train_raw)
    X_val = preprocessor.transform(X_val_raw)

    y_train_log = np.log1p(y_train)
    y_val_log = np.log1p(y_val)

    model = GradientBoostingRegressor(**FINAL_GB_PARAMS)
    model.fit(X_train, y_train_log)

    val_pred = np.expm1(model.predict(X_val))
    metrics = {
        "log_rmse": float(np.sqrt(mean_squared_error(y_val_log, model.predict(X_val)))),
        "mae": float(mean_absolute_error(y_val, val_pred)),  # dollars
        "trained_at": time.time(),
        "n_train_rows": len(X_train),
        "trained_on": "recent_live_labeled_data",
    }

    model_path = MODELS_DIR / f"model_{version_name}.joblib"
    meta_path = MODELS_DIR / f"model_{version_name}.meta.json"
    preproc_path = MODELS_DIR / f"preprocessor_{version_name}.joblib"
    joblib.dump(model, model_path)
    preprocessor.save(preproc_path)
    meta_path.write_text(json.dumps(metrics, indent=2))
    return model_path, metrics


def deploy_shadow(version_name):
    resp = requests.post(f"{API_URL}/admin/set_shadow", json={"version": version_name}, timeout=5)
    resp.raise_for_status()
    return resp.json()


def run_retrain(n_recent):
    """
    Callable entry point (used by the CLI below and by the automation loop).
    Returns the new version name on success, or None if there wasn't enough
    labeled data yet to retrain on.
    """
    print(f"Pulling the {n_recent} most recent labeled predictions from prediction_logs...")
    df = load_recent_labeled_data(n_recent)
    if df is None or len(df) < 50:
        n = 0 if df is None else len(df)
        print(f"Not enough labeled data yet ({n} rows). Aborting — need more traffic first.")
        return None

    version = get_next_version()
    print(f"Training challenger {version} on {len(df)} rows of recent live data...")
    model_path, metrics = train_challenger(df, version)
    print(f"Saved {model_path.name} | Log RMSE={metrics['log_rmse']:.4f} | MAE=${metrics['mae']:,.0f}")

    print(f"Deploying {version} to shadow mode (production traffic unaffected)...")
    result = deploy_shadow(version)
    print(f"Serving API confirms: shadow_version = {result['shadow_version']}")
    print("Production is still serving the original model. Nothing has been promoted.")
    return version


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-recent", type=int, default=500,
                         help="How many recent labeled predictions to train the challenger on")
    args = parser.parse_args()
    run_retrain(args.n_recent)


if __name__ == "__main__":
    main()
