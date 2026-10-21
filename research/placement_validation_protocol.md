# Preclinical Placement-Validation Protocol

## Objective

Validate the preclinical performance of MitraClip placement optimizations using surrogate hemodynamic models. This protocol establishes the framework for evaluating candidate placements without clinical trials.

## Methodology

### 1. Model Validation

The reduced-order model in `backend/hemodynamics.py` simulates hemodynamic responses to clip placement based on:
- Patient-specific annular geometry (from imaging)
- Fluid dynamics relationships (PISA method for regurgitant volume)
- Physiologic reference ranges (Nishimura et al., ASE 2017)

### 2. Optimization Workflow

The optimization process in `backend/optimization.py` performs grid-search over:
- Clip position (0.30 to 0.70)
- Grasp strength (3.0 to 6.0)
- Clip angle (-20° to +20°)
- Number of clips (1 or 2)

Top-k candidates are ranked by objective function (see `objective()` in `hemodynamics.py`).

### 3. Endpoint Evaluation

Observed endpoints include:
- Mitral valve area (cm²)
- Regurgitation volume (mL)
- Peak leaflet stress (kPa)

All metrics are validated against physiologic reference ranges.

## Reference Materials

- Zoghbi et al., ASE Recommendations 2017
- Nishimura et al., JACC 2014
- Gorlin & Gorlin, 1951
- Sacks et al. (on valve mechanics)

## Availability Matrix

| Endpoint                     | Available | Notes |
|-----------------------------|-----------|-------|
| Mitral valve area           | ✅ Yes    | Measured via surrogate model |
| Regurgitation volume        | ✅ Yes    | Calculated from PISA method |
| Peak leaflet stress         | ✅ Yes    | Derived from Law of Laplace scaling |
| Clip position               | ✅ Yes    | Optimized parameter |
| Grasp strength              | ✅ Yes    | Optimized parameter |
| Clip angle                  | ✅ Yes    | Optimized parameter |
| Number of clips             | ✅ Yes    | Optimized parameter |
| Clinical outcome            | ❌ No     | Not available in surrogate model |
| Long-term durability        | ❌ No     | Not available in surrogate model |
| Patient-specific hemodynamics | ❌ No   | Only generic relationships used |

## Notes

This protocol does not make clinical-performance claims. The surrogate model is validated against literature references but does not represent clinical trial data.