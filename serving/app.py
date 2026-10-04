"""
The serving service. Responsibilities:
  1. Load the current production model (and, if one is running, a shadow
     challenger model) at startup.
  2. Serve predictions on the current production model only.
  3. Log every prediction (features, prediction, model version, timestep)
     to the DB — this log is what the drift monitor and evaluation gate
     read from.
  4. Accept true labels later (delayed-label problem) via /label.
  5. Never auto-promote anything — /admin/promote is a separate,
     human-triggered concern (added in the evaluation-gate step).

Run with: uvicorn serving.app:app --reload --port 8000
"""
import json
import time
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
from fastapi import FastAPI, Depends, HTTPException, UploadFile, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session

from serving.db import init_db, get_session, PredictionLog, PromotionRequest, DriftEvent
from data.house_price_preprocessor import HousePricePreprocessor

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

app = FastAPI(title="Drift-Watchdog Serving Service")

# Allows the React dashboard (running on a different local port, e.g. 5173)
# to call this API from the browser. Fine for local development; if this
# ever goes further than your own machine, lock allow_origins down.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- in-memory model registry -------------------------------------------------
# Each version now needs a MODEL + a PREPROCESSOR loaded together: a raw
# feature dict (e.g. {"OverallQual": 7, "Neighborhood": "CollgCr", ...})
# has to go through that version's fitted preprocessor before the model can
# score it. They're always saved and loaded as a pair.
STATE = {
    "production_version": "v1",
    "shadow_version": None,
    "models": {},  # version -> {"model": ..., "preprocessor": ...}
}


def load_model(version: str):
    if version not in STATE["models"]:
        model_path = MODELS_DIR / f"model_{version}.joblib"
        preproc_path = MODELS_DIR / f"preprocessor_{version}.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"No model file for version {version} at {model_path}")
        if not preproc_path.exists():
            raise FileNotFoundError(f"No preprocessor file for version {version} at {preproc_path}")
        STATE["models"][version] = {
            "model": joblib.load(model_path),
            "preprocessor": HousePricePreprocessor.load(preproc_path),
        }
    return STATE["models"][version]


@app.on_event("startup")
def startup():
    init_db()
    load_model(STATE["production_version"])


# --- schemas -------------------------------------------------------------------
class PredictRequest(BaseModel):
    features: dict  # e.g. {"feature_0": 0.12, "feature_1": -1.3, ...}
    timestep: Optional[int] = None


class PredictResponse(BaseModel):
    prediction: float
    model_version: str
    log_id: int
    latency_ms: Optional[float] = None


class LabelRequest(BaseModel):
    log_id: int
    true_label: float


# --- endpoints -------------------------------------------------------------------
@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest, db: Session = Depends(get_session)):
    prod_version = STATE["production_version"]
    prod = load_model(prod_version)

    t0 = time.perf_counter()
    X = prod["preprocessor"].transform_single(req.features)
    pred_log = prod["model"].predict(X)[0]
    pred = float(np.expm1(pred_log))  # model predicts log1p(SalePrice); invert for a real dollar figure
    prod_latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)

    log = PredictionLog(
        timestep=req.timestep,
        features=req.features,
        prediction=pred,
        model_version=prod_version,
        is_shadow=False,
        latency_ms=prod_latency_ms,
    )
    db.add(log)
    db.commit()
    db.refresh(log)

    # if a shadow challenger is active, score it too (logged, never returned
    # to the caller, never affects the served prediction)
    if STATE["shadow_version"]:
        try:
            t_shadow_0 = time.perf_counter()
            shadow = load_model(STATE["shadow_version"])
            shadow_X = shadow["preprocessor"].transform_single(req.features)
            shadow_pred = float(np.expm1(shadow["model"].predict(shadow_X)[0]))
            shadow_latency_ms = round((time.perf_counter() - t_shadow_0) * 1000.0, 2)

            shadow_log = PredictionLog(
                timestep=req.timestep,
                features=req.features,
                prediction=shadow_pred,
                model_version=STATE["shadow_version"],
                is_shadow=True,
                latency_ms=shadow_latency_ms,
            )
            db.add(shadow_log)
            db.commit()
        except FileNotFoundError:
            pass  # shadow model file not written yet; skip silently

    return PredictResponse(
        prediction=pred,
        model_version=prod_version,
        log_id=log.id,
        latency_ms=prod_latency_ms,
    )


@app.post("/label")
def submit_label(req: LabelRequest, db: Session = Depends(get_session)):
    """
    Attaches a true label to a previously-logged prediction, once it's
    known (this is the delayed-label problem — e.g. actual delivery time
    is only known after the delivery happens). Also back-fills the
    matching shadow prediction for the same timestep, if one exists.
    """
    log = db.query(PredictionLog).filter(PredictionLog.id == req.log_id).first()
    if not log:
        raise HTTPException(404, "log_id not found")
    log.true_label = req.true_label

    if log.timestep is not None and not log.is_shadow:
        shadow_log = (
            db.query(PredictionLog)
            .filter(
                PredictionLog.timestep == log.timestep,
                PredictionLog.is_shadow == True,  # noqa: E712
            )
            .first()
        )
        if shadow_log:
            shadow_log.true_label = req.true_label

    db.commit()
    return {"status": "ok", "log_id": req.log_id}


@app.get("/status")
def status():
    return {
        "production_version": STATE["production_version"],
        "shadow_version": STATE["shadow_version"],
    }


@app.get("/admin/predictions")
def list_predictions(limit: int = 50, db: Session = Depends(get_session)):
    """Most recent predictions (production + shadow), newest first."""
    rows = db.query(PredictionLog).order_by(PredictionLog.id.desc()).limit(limit).all()
    return [
        {
            "id": r.id,
            "timestep": r.timestep,
            "model_version": r.model_version,
            "is_shadow": r.is_shadow,
            "prediction": r.prediction,
            "true_label": r.true_label,
            "latency_ms": r.latency_ms,
            "served_at": r.served_at,
        }
        for r in rows
    ]


@app.get("/admin/drift_events")
def list_drift_events(limit: int = 20, db: Session = Depends(get_session)):
    """Most recent drift-monitor checks, newest first."""
    rows = db.query(DriftEvent).order_by(DriftEvent.id.desc()).limit(limit).all()
    return [
        {
            "id": r.id,
            "checked_at": r.checked_at,
            "window_start_timestep": r.window_start_timestep,
            "window_end_timestep": r.window_end_timestep,
            "drift_score": r.drift_score,
            "method": r.method,
            "threshold": r.threshold,
            "triggered": r.triggered,
            "details": r.details,
            "explanation": r.explanation,
        }
        for r in rows
    ]


class SetShadowRequest(BaseModel):
    version: str


@app.post("/admin/set_shadow")
def set_shadow(req: SetShadowRequest):
    """
    Points the shadow slot at a new challenger model version. Deliberately
    has NO equivalent for production — promoting to production only ever
    happens through a separate, explicitly human-approved path (step 4).
    """
    model_path = MODELS_DIR / f"model_{req.version}.joblib"
    preproc_path = MODELS_DIR / f"preprocessor_{req.version}.joblib"
    if not model_path.exists() or not preproc_path.exists():
        raise HTTPException(404, f"Missing model or preprocessor file for version {req.version}")
    load_model(req.version)  # eagerly load + validate it actually loads
    STATE["shadow_version"] = req.version
    return {"status": "ok", "shadow_version": req.version}


class PromotionDecisionRequest(BaseModel):
    promotion_request_id: int
    decision: str  # "approve" or "reject" — nothing else is accepted
    decided_by: Optional[str] = None


@app.get("/admin/promotion_requests")
def list_promotion_requests(db: Session = Depends(get_session)):
    """Lists pending promotion requests for a human to review."""
    rows = db.query(PromotionRequest).order_by(PromotionRequest.id.desc()).all()
    return [
        {
            "id": r.id,
            "challenger_version": r.challenger_version,
            "production_version_at_request": r.production_version_at_request,
            "challenger_mae": r.challenger_mae,
            "production_mae": r.production_mae,
            "challenger_latency_ms": r.challenger_latency_ms,
            "production_latency_ms": r.production_latency_ms,
            "n_labels_compared": r.n_labels_compared,
            "status": r.status,
            "created_at": r.created_at,
            "explanation": r.explanation,
        }
        for r in rows
    ]


@app.post("/admin/promote")
def decide_promotion(req: PromotionDecisionRequest, db: Session = Depends(get_session)):
    """
    THE ONLY code path in this entire system that can change which model
    serves production traffic. Requires an explicit human decision on an
    existing promotion request created by the evaluation gate — there is no
    way to promote a model that hasn't already passed the statistical gate,
    and no automatic path that calls this endpoint on its own.
    """
    if req.decision not in ("approve", "reject"):
        raise HTTPException(400, "decision must be 'approve' or 'reject'")

    promo = db.query(PromotionRequest).filter(PromotionRequest.id == req.promotion_request_id).first()
    if not promo:
        raise HTTPException(404, "promotion request not found")
    if promo.status != "pending":
        raise HTTPException(400, f"promotion request is already '{promo.status}'")

    promo.status = "approved" if req.decision == "approve" else "rejected"
    promo.decided_at = time.time()
    promo.decided_by = req.decided_by
    db.commit()

    if req.decision == "approve":
        load_model(promo.challenger_version)  # confirm it loads before switching
        STATE["production_version"] = promo.challenger_version
        STATE["shadow_version"] = None  # clear shadow slot; nothing left to test against itself

    return {
        "status": "ok",
        "promotion_status": promo.status,
        "production_version": STATE["production_version"],
        "shadow_version": STATE["shadow_version"],
    }


# --- aggregate stats (KPI strip) -----------------------------------------------
@app.get("/admin/stats")
def get_stats(db: Session = Depends(get_session)):
    """Aggregate numbers for the dashboard KPI strip."""
    total_predictions = db.query(PredictionLog).filter(
        PredictionLog.is_shadow == False  # noqa: E712
    ).count()
    total_labeled = db.query(PredictionLog).filter(
        PredictionLog.is_shadow == False,  # noqa: E712
        PredictionLog.true_label.isnot(None),
    ).count()
    shadow_predictions = db.query(PredictionLog).filter(
        PredictionLog.is_shadow == True  # noqa: E712
    ).count()

    recent_labeled = (
        db.query(PredictionLog)
        .filter(
            PredictionLog.is_shadow == False,  # noqa: E712
            PredictionLog.true_label.isnot(None),
        )
        .order_by(PredictionLog.id.desc())
        .limit(200)
        .all()
    )
    recent_prod = (
        db.query(PredictionLog)
        .filter(
            PredictionLog.is_shadow == False,  # noqa: E712
        )
        .order_by(PredictionLog.id.desc())
        .limit(100)
        .all()
    )
    production_latency_ms = None
    if recent_prod:
        lats = [r.latency_ms for r in recent_prod if r.latency_ms is not None]
        if lats:
            production_latency_ms = round(sum(lats) / len(lats), 1)

    production_mae = None
    if recent_labeled:
        errors = [abs(r.prediction - r.true_label) for r in recent_labeled]
        production_mae = sum(errors) / len(errors)

    return {
        "total_predictions": total_predictions,
        "total_labeled": total_labeled,
        "production_mae": production_mae,
        "production_latency_ms": production_latency_ms,
        "shadow_predictions": shadow_predictions,
    }


# --- action endpoints (dashboard buttons) --------------------------------------
@app.post("/admin/upload_stream")
async def upload_stream(file: UploadFile):
    """Accepts a CSV upload and saves it as the new stream data file."""
    import pandas as pd
    data_dir = Path(__file__).resolve().parent.parent / "data"
    dest = data_dir / "stream_data.csv"
    contents = await file.read()
    dest.write_bytes(contents)
    df = pd.read_csv(dest)
    return {"status": "ok", "rows": len(df), "filename": file.filename}


@app.post("/admin/train_baseline")
def train_baseline_endpoint(background_tasks: BackgroundTasks):
    """Triggers baseline model (v1) training in the background."""
    def _train():
        from models.train_baseline import train
        train("v1")
        STATE["models"].pop("v1", None)
        load_model("v1")
    background_tasks.add_task(_train)
    return {"status": "started", "message": "Baseline training started in background. Check /status shortly."}


@app.post("/admin/run_drift_check")
def run_drift_check_endpoint():
    """Runs the drift monitor once and returns the result."""
    from monitor.drift_monitor import run_once
    result = run_once(window_size=200)
    if result is None:
        return {"status": "skipped", "message": "Not enough data or no new predictions since last check."}
    return {
        "status": "ok",
        "triggered": result["triggered"],
        "score": result["overall_score"],
        "n_drifted": result["n_drifted_features"],
    }


@app.post("/admin/retrain")
def retrain_endpoint(background_tasks: BackgroundTasks):
    """Triggers challenger retraining and shadow deployment in the background."""
    def _retrain():
        from retrain.retrain_pipeline import run_retrain
        run_retrain(n_recent=500)
    background_tasks.add_task(_retrain)
    return {"status": "started", "message": "Retrain started in background. Watch /status for the new shadow version."}
