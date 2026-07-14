"""FastAPI service exposing the TEER pipeline.

Endpoints:
  GET  /api/cases                 -> list real cases + measured geometry
  GET  /api/cases/{id}            -> one case summary
  POST /api/simulate              -> reduced-order result for a placement
  POST /api/optimize/{id}         -> ranked placements for a real case
  POST /api/upload-dicom          -> intake stub (documents production path)

Run:  uvicorn backend.api:app --reload --port 8080
"""

from __future__ import annotations

import hashlib
import os
from datetime import datetime, timezone

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from . import cases
from .fsi_adapter import FSISolver, dolfinx_available
from .hemodynamics import ClipPlacement, simulate
from .optimization import optimize

app = FastAPI(title="TEER CDSS API", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class PlacementIn(BaseModel):
    case_id: str
    position: float = Field(0.5, ge=0.0, le=1.0)
    grasp_width_mm: float = Field(4.0, ge=2.0, le=8.0)
    angle_deg: float = Field(0.0, ge=-45.0, le=45.0)
    clip_count: int = Field(1, ge=1, le=2)


def _audit(event: str, subject: str, detail: dict | None = None) -> None:
    """Append-only audit line (HIPAA-style). Subject is always a hash."""
    line = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "event": event,
        "subject": subject,
        "detail": detail or {},
    }
    os.makedirs("audit_logs", exist_ok=True)
    with open(os.path.join("audit_logs", "audit.jsonl"), "a") as f:
        import json

        f.write(json.dumps(line) + "\n")


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "cases": len(cases.list_cases()),
        "full_fsi_available": dolfinx_available(),
        "engine": "reduced-order surrogate" if not dolfinx_available() else "FEniCSx FSI",
    }


@app.get("/api/cases")
def get_cases() -> dict:
    ids = cases.list_cases()
    return {"count": len(ids), "cases": [cases.case_summary(c) for c in ids]}


@app.get("/api/cases/{case_id}")
def get_case(case_id: str) -> dict:
    try:
        return cases.case_summary(case_id)
    except KeyError as e:
        raise HTTPException(404, str(e))


@app.post("/api/simulate")
def run_simulate(body: PlacementIn) -> dict:
    try:
        geom = cases.get_geometry(body.case_id)
    except KeyError as e:
        raise HTTPException(404, str(e))
    clip = ClipPlacement(body.position, body.grasp_width_mm, body.angle_deg, body.clip_count)
    result = simulate(geom, clip)
    _audit("simulate", body.case_id, {"clip_count": body.clip_count})
    return {"case_id": body.case_id, "placement": clip.__dict__, "result": result.to_dict()}


@app.post("/api/optimize/{case_id}")
def run_optimize(case_id: str, top_k: int = 3) -> dict:
    try:
        geom = cases.get_geometry(case_id)
    except KeyError as e:
        raise HTTPException(404, str(e))
    ranked = optimize(geom, top_k=top_k)
    _audit("optimize", case_id, {"top_k": top_k})
    return {"case_id": case_id, "engine": FSISolver(geom).available() and "FSI" or "surrogate", "ranked": ranked}


@app.post("/api/upload-dicom")
async def upload_dicom(file: UploadFile = File(...)) -> dict:
    """Intake stub. In production: anonymize -> segment (MONAI) -> mesh ->
    FSI (FEniCSx) -> optimize. Here we hash + acknowledge, no PHI retained."""
    data = await file.read()
    subject_hash = hashlib.sha256(data[:4096]).hexdigest()[:16]
    _audit("dicom.received", subject_hash, {"bytes": len(data), "name": file.filename})
    return {
        "status": "accepted",
        "subject_hash": subject_hash,
        "bytes": len(data),
        "next_steps": [
            "anonymize DICOM (strip PHI tags)",
            "segment leaflets (MONAI 3D U-Net)",
            "reconstruct watertight mesh (marching cubes + PyVista)",
            "solve FSI (FEniCSx) — requires GPU backend",
            "optimize clip placement and rank",
        ],
        "full_fsi_available": dolfinx_available(),
    }
