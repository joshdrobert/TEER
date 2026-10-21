# TEER Project Plan

## Objective
turn validated geometry into research-ready features, perform QC/outlier analysis, develop meaningful geometric representations, connect geometry to clip-placement/TEER decision-support objectives, establish baselines, validate any predictive/simulation models rigorously

## Current Milestone
Inspect current codebase and define concrete technical tasks.

## Active Task
1. Analyze project state and build rigorous technical roadmap aligned with objectives.

## Next Milestone
Develop rigorous technical roadmap for TEER project implementation.

## Tasks

### Task 1: Implement Quality Control Pipeline for Geometry Features
- **File**: `scripts/extract_features.py`
- **Complexity**: Medium (3-5 days)
- **Dependencies**: None
- **Description**: Add deterministic QC pipeline to extract_features.py that computes per-geometry validity checks (face count, volume, surface area sanity), flags outliers by IQR on each metric, and writes QC report CSV.

### Task 2: Develop Geometric Representation Module
- **File**: `src/teer_cdss/geometric_features.py`
- **Complexity**: High (5-7 days)
- **Dependencies**: Task 1 (QC pipeline)
- **Description**: Implement geometric representation module that computes signed/unsigned distance field samples on a regular grid over the mitral annulus ROI, plus Laplace-Beltrami spectrum features (first 10 eigenvalues) from STL mesh.

### Task 3: Connect Geometry to Clip-Placement Decision Support
- **File**: `src/teer_cdss/clip_placement.py` (to be created)
- **Complexity**: Medium (4-6 days)
- **Dependencies**: Task 2 (geometric features)
- **Description**: Develop decision-support module that connects computed geometric features to clip-placement recommendations for TEER procedures.

### Task 4: Establish Baselines and Validation Framework
- **File**: `tests/test_baseline_validation.py` (to be created)
- **Complexity**: Medium (3-5 days)
- **Dependencies**: Tasks 1-3
- **Description**: Establish baselines using existing real cases and validate any predictive/simulation models rigorously.

## Success Criteria

### Milestone Completion
- [ ] All tasks have concrete implementations with defined file paths
- [ ] Each task has complexity estimates and dependencies clearly identified
- [ ] Success criteria for each milestone are well-defined and measurable
- [ ] Code is validated against existing test suite
- [ ] All new functionality preserves existing verified work

### Task-Specific Success Metrics
1. **QC Pipeline**: 
   - [ ] Per-geometry validity checks implemented
   - [ ] Outlier detection via IQR method functional
   - [ ] QC report CSV output working
   - [ ] Existing feature columns preserved

2. **Geometric Features Module**:
   - [ ] Distance field samples computed correctly
   - [ ] Laplace-Beltrami spectrum features calculated
   - [ ] Consistent numpy array outputs with defined shapes
   - [ ] Integration with existing pipeline verified

3. **Decision Support Module**:
   - [ ] Geometric features connected to clip placement recommendations
   - [ ] TEER decision-support objectives met
   - [ ] Simulation models validated against real cases

4. **Baseline Validation**:
   - [ ] Baselines established for key metrics
   - [ ] Predictive models validated rigorously
   - [ ] Simulation models demonstrate physiologic behavior
   - [ ] All existing tests continue to pass

