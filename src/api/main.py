"""Laptop 1: PYTHONPATH=src python -m uvicorn api.main:app --host 0.0.0.0 --port 8000."""
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field
from pathsnap.config import ROOT, load_config
from .service import Service


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    trip_id: str = Field(min_length=1, max_length=100, pattern=r'^[A-Za-z0-9_-]+$')


def create_app(service=None):
    @asynccontextmanager
    async def lifespan(app):
        app.state.service = service
        if service is None:
            path = Path(os.getenv('PATHSNAP_MODEL', str(ROOT / 'artifacts/model_bundle.joblib')))
            if path.exists():
                app.state.service = Service(joblib.load(path), load_config())
        yield

    app = FastAPI(title='PathSnap API', version='0.2.0-hmm', lifespan=lifespan)
    app.add_middleware(CORSMiddleware,
        allow_origins=[x.strip() for x in os.getenv('PATHSNAP_CORS', 'http://localhost:8501').split(',') if x.strip()],
        allow_methods=['GET', 'POST'], allow_headers=['Content-Type'])

    @app.get('/health')
    def health():
        loaded = app.state.service is not None
        return {'status': 'ok' if loaded else 'not_ready', 'model_loaded': loaded}

    @app.get('/trips')
    def trips():
        try:
            if app.state.service is not None:
                return {'trips': app.state.service.trip_ids()}
            from pathsnap import io
            return {'trips': io.trip_ids(load_config()['splits']['student_test'])}
        except FileNotFoundError:
            raise HTTPException(503, 'Data unavailable; configure PATHSNAP_DATA_DIR') from None

    @app.post('/predict')
    def predict(request: PredictRequest):
        svc = app.state.service
        if svc is None:
            raise HTTPException(503, 'No model loaded. Supply the model bundle and restart the server.')
        try:
            if request.trip_id not in svc.trip_ids():
                raise HTTPException(404, 'Unknown trip_id')
            return svc.predict(request.trip_id)
        except HTTPException:
            raise
        except FileNotFoundError:
            raise HTTPException(503, 'Trip data unavailable on server') from None
        except Exception:
            logging.exception('Prediction failed for %s', request.trip_id)
            raise HTTPException(500, 'Prediction failed; inspect server logs') from None

    return app


app = create_app()
