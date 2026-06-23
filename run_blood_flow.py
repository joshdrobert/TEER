#!/usr/bin/env python
"""Run the full blood-flow simulation pipeline and render a professional MP4.

Usage:
    .venv/bin/python run_blood_flow.py
"""

from pathlib import Path
import sys
import time


def main() -> int:
    workspace = Path(__file__).resolve().parent
    stl_path = workspace / "segmented_valve_mesh_smoothed.stl"

    if not stl_path.exists():
        print(f"ERROR: STL file not found at {stl_path}", file=sys.stderr)
        return 1

    print("=" * 70)
    print("  Mitral Valve Blood Flow Simulation — Professional MP4 Pipeline")
    print("=" * 70)
    print()

    # --- Step 1: Run simulation -------------------------------------------
    print("[1/2] Running simulation (180 timesteps, 3 cardiac cycles)...")
    t0 = time.time()

    from teer_cdss.mitral_mock import generate_and_run_mock_case

    result = generate_and_run_mock_case(
        valve_obj=stl_path,
        workspace=workspace,
        time_steps=180,
        static_diastole=False,
    )
    sim_time = time.time() - t0
    print(f"       Simulation complete in {sim_time:.1f}s")
    print(f"       Case dir: {result.case_dir}")
    print()

    # --- Step 2: Render MP4 -----------------------------------------------
    print("[2/2] Rendering professional MP4...")
    t0 = time.time()

    from teer_cdss.blood_flow_mp4 import render_blood_flow_mp4

    mp4_result = render_blood_flow_mp4(
        case_dir=result.case_dir,
        fps=30,
        dpi=150,
    )
    render_time = time.time() - t0
    print(f"       Render complete in {render_time:.1f}s")
    print()

    print("=" * 70)
    print(f"  MP4 OUTPUT: {mp4_result.mp4_path}")
    print(f"  Frames: {mp4_result.n_frames}  Duration: {mp4_result.duration_seconds:.1f}s")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
    sys.exit(main())
