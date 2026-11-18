"""PathSnap FastAPI backend (owner: C). Skeleton decoder = per-observation argmax.

Run (Laptop 1):
    PYTHONPATH=src uvicorn api.main:app --host 0.0.0.0 --port 8000

Ground truth stays server-side: metrics are aggregate only, geometry is predicted only.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # <repo>/src

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from pathsnap import baseline, io as psio, metrics, pipeline
from pathsnap.config import artifacts_dir, load_config

CFG = load_config()
MODEL_PATH = artifacts_dir() / "model_bundle.joblib"
_MODEL = None

app = FastAPI(title="PathSnap API", version="0.1.0-phase1")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten at final deployment if required
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_model():
    global _MODEL
    if _MODEL is None and MODEL_PATH.exists():
        _MODEL = joblib.load(MODEL_PATH)
    return _MODEL


class PredictRequest(BaseModel):
    trip_id: str


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": MODEL_PATH.exists()}


@app.get("/trips")
def trips():
    try:
        ids = psio.trip_ids(CFG["splits"]["student_test"])
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    return {"trips": ids}


@app.post("/predict")
def predict(req: PredictRequest):
    t0 = time.perf_counter()
    model = get_model()
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="No model bundle. Run scripts/train_baseline.py first.",
        )

    split = CFG["splits"]["student_test"]
    try:
        df, diag = pipeline.get_features(split, req.trip_id, CFG)
        trip = pipeline.get_trip(split, req.trip_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Unknown trip '{req.trip_id}'")

    t_feat = time.perf_counter()

    scores = baseline.predict_scores(model, df)
    chosen = baseline.choose_one_per_observation(df, scores)
    pred_arcs = chosen["arc_id"].to_numpy(dtype=np.int64)

    # ---- decoder hook (owner: C) -------------------------------------------
    # Replace this argmax with HMM/Viterbi over the candidate lattice:
    #   states  = candidate arcs per observation
    #   emission= the `scores` above (calibrated probabilities)
    #   transition = arc adjacency from trip["arcs"] + distance
    #   keep candidate <-> arc correspondence; handle gaps / missing candidates.
    # Then compare classifier-only vs HMM in the report (§5).
    decoder = "argmax-baseline"
    # ------------------------------------------------------------------------

    t_inf = time.perf_counter()

    nodes, arcs = trip["nodes"], trip["arcs"]
    pred_line = pipeline.line_from_arcs(pred_arcs, nodes, arcs)

    label_arc = chosen["label_arc"].to_numpy()
    result = metrics.classification_metrics(label_arc, pred_arcs)
    result.update(metrics.candidate_prf(df["label"].to_numpy(), scores))
    result["candidate_log_loss"] = metrics.candidate_log_loss(
        df["label"].to_numpy(), scores
    )
    result["candidate_miss_rate"] = diag["candidate_miss_rate"]
    result["route_continuity"] = metrics.route_continuity(pred_arcs, arcs)

    track = trip["track"]
    return {
        "trip_id": req.trip_id,
        "decoder": decoder,
        "n_observations": int(len(track)),
        "observations": np.column_stack(
            [track["lon"].to_numpy(float), track["lat"].to_numpy(float)]
        ).round(7).tolist(),
        "predicted_route": pred_arcs.tolist(),
        "predicted_line": pred_line.round(7).tolist(),
        "metrics": result,
        "timings": {
            "features_ms": round((t_feat - t0) * 1000, 1),
            "inference_ms": round((t_inf - t_feat) * 1000, 1),
            "total_ms": round((time.perf_counter() - t0) * 1000, 1),
        },
    }
