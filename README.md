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

- Python 3.10+
- Dependencies declared in [pyproject.toml](/Users/josh/Documents/TEER/pyproject.toml)

## Install

```bash
pip install -e .
```

## Usage

```bash
teer-pipeline --help
python -m compileall src
```
