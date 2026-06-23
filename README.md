# TEER CDSS Framework

`teer-cdss` is a Python scaffold for a Transcatheter Edge-to-Edge Repair decision-support pipeline focused on mitral valve intervention planning.

The codebase is structured so clinical schemas, orchestration, and algorithmic modules can evolve independently as validated models and simulation components are added.

## Scope

The current package layout covers these domains:

- dataset acquisition and PyTorch data loading
- DICOM ingestion and preprocessing
- 3D mitral segmentation
- mesh generation and MitraClip geometry creation
- fluid-structure interaction orchestration
- clip placement optimization
- physician-facing export payloads

## Package Layout

```text
src/teer_cdss/
  __init__.py
  acquisition.py
  cli.py
  clinical.py
  exceptions.py
  export.py
  fsi.py
  mesh.py
  optimization.py
  orchestrator.py
  schemas.py
  segmentation.py
```

## Requirements

- Python 3.9+
- Dependencies declared in [pyproject.toml](/Users/josh/Documents/TEER/pyproject.toml)

## Install

```bash
pip install -e .
```

## Usage

```bash
teer-pipeline --help
teer-pipeline prepare-data
teer-pipeline data-summary
teer-pipeline run /path/to/dicom1.dcm /path/to/dicom2.dcm --workspace ./run-output
teer-pipeline mock-mitral-fsi mitral_valve_with_chordae.obj --workspace .
python -m compileall src
```

## Mock Mitral URIS-FSI Case

The repo now includes a runnable `svMultiPhysics` mock case that:

- morphs the upstream validated URIS-FSI pipe example into an LV-like chamber
- places `mitral_valve_with_chordae.obj` as an immersed mitral valve surface
- synthesizes open/close leaflet motion data around the detected annulus
- runs `svmultiphysics` and writes outputs to `artifacts/mock_mitral_uris_fsi/`

Run it with:

```bash
teer-pipeline mock-mitral-fsi mitral_valve_with_chordae.obj --workspace .
```

Important outputs:

- `artifacts/mock_mitral_uris_fsi/solver.xml`
- `artifacts/mock_mitral_uris_fsi/1-procs/result_005.vtu`
- `artifacts/mock_mitral_uris_fsi/1-procs/result_uris_MitralValve_005.vtu`
- `artifacts/mock_mitral_uris_fsi/summary.json`

## Dataset Adaptation: MVSeg2023 3D TEE

The bundled `data/` directory contains the MICCAI 2023 MVSeg dataset archives:

```text
data/
  train.zip
  val.zip
  test.zip
```

Each archive contains single-frame 3D transesophageal echocardiography NIfTI volumes and matching leaflet labels:

```text
train/train_001-US.nii.gz
train/train_001-label.nii.gz
```

Native label IDs:

```text
0 background
1 posterior_leaflet
2 anterior_leaflet
```

Prepare the local extracted dataset with:

```bash
teer-pipeline prepare-data
```

This extracts the splits into:

```text
data/mvseg2023/
  train/
  val/
  test/
```

Then verify pair counts with:

```bash
teer-pipeline data-summary
```

The default config points directly at these local archives:

```python
DatasetResource(
    name="MVSeg2023",
    uri="local://data",
    local_root=workspace / "data" / "mvseg2023",
    image_suffix="-US.nii.gz",
    label_suffix="-label.nii.gz",
    split_archives={
        "train": workspace / "data" / "train.zip",
        "val": workspace / "data" / "val.zip",
        "test": workspace / "data" / "test.zip",
    },
)
```

The pipeline still supports generic local zip imports for other manually downloaded 3D TEE datasets through `LocalArchiveDatasetFetcher`.

## Interactive 3D Mitral Valve FSI Simulation (TEER Web Dashboard)

The repository includes a real-time, interactive 3D web dashboard for Transcatheter Edge-to-Edge Repair (TEER) simulation and clinical planning.

### Features:
- **Interactive 3D Graphics**: Orbit, pan, and zoom to inspect high-resolution geometries (`segmented_valve_mesh_smoothed.stl`, `mitral_valve_with_chordae.obj`, and `mitral_valve.obj`).
- **Anatomical Mitral Valve Motion**: Fully dynamic leaflet opening (diastole) and closing/doming (systole) based on cardiac cycle state.
- **Interactive Blood Flow Simulator**: Real-time particle system simulating 3D forward flow (with recirculation toroidal vortices) and narrow regurgitant systolic jets.
- **MitraClip G4 Integration**: Enable and place a high-resolution 3D MitraClip (NT G4 surrogate) along the coaptation line to dynamically pinch the leaflets together and create a double-orifice mitral valve flow.
- **Real-time Waveforms**: Live Canvas charts showing Left Atrial (LA) and Left Ventricular (LV) pressure-time curves, with active telemetry tracking flow rates and regurgitation volumes.
- **Camera Preset Views**: Switch instantly between Atrial (Top), Ventricular (Bottom), Side Cutaway, and Isometric views.

### Running the Web Server:

You can run the lightweight Python web server with:

```bash
python server.py
```

This starts a local CORS-enabled HTTP server serving static files. Open the URL in your web browser:
[http://localhost:8000](http://localhost:8000)
