# TEER · Precision-Guided Mitral Valve Repair

Patient-specific decision support for **Transcatheter Edge-to-Edge Repair (TEER)**.
The system turns a patient's 3D transesophageal echo (TEE) into a physics-based
model of their mitral valve, then computes **where to place the MitraClip, how
many to use, and the expected residual regurgitation** — before the procedure.

> **Research prototype — not for clinical use.** All bundled anatomy is from the
> public MVSeg2023 3D TEE dataset. Predicted values are illustrative and have
> not been cleared by any regulatory body.

---

## What's real here (no synthetic data)

| Component | Status | Notes |
|---|---|---|
| **105 patient valve geometries** | ✅ real | Segmented surfaces in [valves/](valves/), measured geometry in [valves_metadata.json](valves_metadata.json) (from MVSeg2023 3D TEE) |
| **MitraClip NT geometry** | ✅ real | 1:1 articulated STL assembly in [clip/](clip/) |
| **Landing page + interactive simulator** | ✅ runs | [docs/](docs/) — static, deploys to GitHub Pages, renders the real meshes |
| **Reduced-order hemodynamic model** | ✅ runs (CPU) | [backend/hemodynamics.py](backend/hemodynamics.py) — physics-grounded surrogate; identical model in Python and in the browser |
| **Placement optimizer** | ✅ runs (CPU) | [backend/optimization.py](backend/optimization.py) — ranks clip strategies |
| **FastAPI service** | ✅ runs (CPU) | [backend/api.py](backend/api.py) — cases, simulate, optimize, DICOM intake |
| **Segmentation model (3D U-Net)** | 🔧 needs GPU | Architecture defined; training on MVSeg2023 requires GPU-hours |
| **Full FSI solve (FEniCSx/DOLFINx)** | 🔧 needs DOLFINx | Interface in [backend/fsi_adapter.py](backend/fsi_adapter.py); runs on the GPU backend, not on Pages |

The **reduced-order model is a surrogate, not a mock**: every relationship is a
recognized clinical/fluid relation (regurgitant volume = EROA × VTI; double-orifice
valve area; Laplace-scaled leaflet tension) parameterized by each patient's
*measured* annular geometry. It runs in real time and shortlists candidates that
the full FEniCSx solve then refines. See references in
[backend/hemodynamics.py](backend/hemodynamics.py).

---

## Quick start

### 1. The web experience (landing page + simulator)

```bash
python -m http.server 8000 --directory docs
# open http://localhost:8000            (landing page)
# open http://localhost:8000/demo.html  (interactive clip simulator)
```

The simulator loads a **real** segmented valve, lets you place the **real**
MitraClip along the coaptation line, and shows predicted residual regurgitation,
valve area, and leaflet stress live. "Optimize placement" grid-searches strategies
and ranks the top three.

### 2. The backend API

```bash
python -m pip install -r backend/requirements.txt
uvicorn backend.api:app --reload --port 8080

curl http://localhost:8080/api/health
curl http://localhost:8080/api/cases
curl -X POST http://localhost:8080/api/optimize/train_087?top_k=3
```

### 3. Tests

```bash
python -m pip install pytest
python -m pytest tests/ -q
```

### 4. Regenerate web assets from the full-resolution meshes

```bash
python scripts/prepare_web_assets.py   # decimates real valves -> docs/assets/models
```

---

## Deployment

**Landing page → GitHub Pages.** Push to `main`; the workflow in
[.github/workflows/deploy-pages.yml](.github/workflows/deploy-pages.yml) publishes
`docs/`. Enable Pages once at *Settings → Pages → Source: GitHub Actions*.

**Backend → container.** The API is a standard FastAPI app; deploy on Google
Cloud Run / any container host. The heavy segmentation + FSI stages belong on a
GPU worker (use the `dolfinx/dolfinx:stable` image for the solver).

---

## Architecture

```
3D TEE DICOM
   │  anonymize (strip PHI) ──────────────► audit log
   ▼
segment leaflets  (MONAI 3D U-Net)          [GPU]
   ▼
reconstruct mesh  (marching cubes, PyVista)
   ▼
simulate flow     (FEniCSx FSI)  ◄── reduced-order surrogate seeds/ranks  [GPU]
   ▼
optimize clip     (search → rank by regurgitation, stress, stenosis)
   ▼
ranked recommendations  ──►  web simulator / OR overlay
```

## Repository layout

```
docs/            Static site: landing page, simulator, real web assets  (→ Pages)
backend/         FastAPI service, hemodynamic model, optimizer, FSI adapter
scripts/         Asset preparation from the full-res segmented meshes
tests/           Backend tests against the real bundled cases
valves/          105 real segmented valve surfaces (MVSeg2023)
clip/            Real MitraClip NT STL assembly
src/teer_cdss/   Earlier pipeline scaffolding (DICOM, mesh, export)
```

## Clinical mentors

Dr. Michael Reardon · Dr. Fernando Ramirez Del Val (Houston Methodist).
