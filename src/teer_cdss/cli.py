"""Command-line entrypoint for the TEER pipeline scaffold."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from .schemas import TEERPipelineConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TEER decision-support pipeline scaffold.")
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="Run the clinical DICOM pipeline scaffold.")
    _add_run_arguments(run_parser)

    prepare_parser = subparsers.add_parser("prepare-data", help="Extract and validate the bundled MVSeg2023 dataset.")
    prepare_parser.add_argument("--workspace", type=Path, default=Path.cwd(), help="Project workspace root.")

    summary_parser = subparsers.add_parser("data-summary", help="Summarize prepared MVSeg2023 image/label pairs.")
    summary_parser.add_argument("--workspace", type=Path, default=Path.cwd(), help="Project workspace root.")

    mock_parser = subparsers.add_parser("mock-mitral-fsi", help="Build and run a mock mitral URIS-FSI case.")
    mock_parser.add_argument("valve_obj", type=Path, help="Path to a mitral OBJ surface mesh.")
    mock_parser.add_argument("--workspace", type=Path, default=Path.cwd(), help="Project workspace root.")

    render_parser = subparsers.add_parser("render-mock-mitral-fsi", help="Render images and a GIF for a completed mock mitral case.")
    render_parser.add_argument("--case-dir", type=Path, default=None, help="Case directory. Defaults to workspace artifacts path.")
    render_parser.add_argument("--workspace", type=Path, default=Path.cwd(), help="Project workspace root.")

    view_parser = subparsers.add_parser("view-mock-mitral-fsi", help="Open an interactive 3D PyVista viewer for a completed mock mitral case.")
    view_parser.add_argument("--case-dir", type=Path, default=None, help="Case directory. Defaults to workspace artifacts path.")
    view_parser.add_argument("--workspace", type=Path, default=Path.cwd(), help="Project workspace root.")
    view_parser.add_argument("--cutaway", action="store_true", help="Enable cutaway view of the chambers and flow.")

    mp4_parser = subparsers.add_parser("render-blood-flow-mp4", help="Render a professional blood flow MP4 animation.")
    mp4_parser.add_argument("--case-dir", type=Path, default=None, help="Case directory. Defaults to workspace artifacts path.")
    mp4_parser.add_argument("--workspace", type=Path, default=Path.cwd(), help="Project workspace root.")
    mp4_parser.add_argument("--fps", type=int, default=30, help="Frames per second for MP4 output.")
    mp4_parser.add_argument("--dpi", type=int, default=150, help="Resolution DPI for MP4 frames.")
    return parser


def build_legacy_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the TEER decision-support pipeline scaffold.")
    _add_run_arguments(parser)
    return parser


def _add_run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("dicom_paths", nargs="*", type=Path, help="Input DICOM paths for one TEE series.")
    parser.add_argument("--workspace", type=Path, default=Path.cwd(), help="Artifact workspace root.")
    parser.add_argument("--operator", type=str, default="system", help="Operator identifier for anonymization audit.")


def main(argv: Sequence[str] | None = None) -> int:
    raw_args = list(argv) if argv is not None else sys.argv[1:]
    command_names = {"run", "prepare-data", "data-summary", "mock-mitral-fsi", "render-mock-mitral-fsi", "view-mock-mitral-fsi", "render-blood-flow-mp4"}
    parser = build_parser() if raw_args and raw_args[0] in command_names else build_legacy_parser()
    args = parser.parse_args(raw_args)

    if getattr(args, "command", None) == "prepare-data":
        return _prepare_data(args.workspace)
    if getattr(args, "command", None) == "data-summary":
        return _data_summary(args.workspace)
    if getattr(args, "command", None) == "mock-mitral-fsi":
        return _mock_mitral_fsi(args.valve_obj, args.workspace)
    if getattr(args, "command", None) == "render-mock-mitral-fsi":
        return _render_mock_mitral_fsi(args.case_dir, args.workspace)
    if getattr(args, "command", None) == "view-mock-mitral-fsi":
        return _view_mock_mitral_fsi(args.case_dir, args.workspace, getattr(args, "cutaway", False))
    if getattr(args, "command", None) == "render-blood-flow-mp4":
        return _render_blood_flow_mp4(args.case_dir, args.workspace,
                                      getattr(args, "fps", 30),
                                      getattr(args, "dpi", 150))

    if not args.dicom_paths:
        parser.print_help()
        return 0
    from .orchestrator import TEERPipelineOrchestrator

    config = TEERPipelineConfig.default(args.workspace)
    orchestrator = TEERPipelineOrchestrator(config=config, workspace=args.workspace)
    summary = orchestrator.run(args.dicom_paths, operator=args.operator)
    print(json.dumps(
        {
            "subject_id": summary.subject_id,
            "artifacts": {key: str(value) for key, value in summary.artifacts.items()},
            "candidate_count": len(summary.candidate_rankings),
        },
        indent=2,
    ))
    return 0


def _mvseg_resource(workspace: Path):
    config = TEERPipelineConfig.default(workspace)
    for resource in config.datasets:
        if resource.name == "MVSeg2023":
            return resource
    raise RuntimeError("MVSeg2023 resource is not configured.")


def _prepare_data(workspace: Path) -> int:
    from .acquisition import MVSeg2023DatasetPreparer

    summaries = MVSeg2023DatasetPreparer(_mvseg_resource(workspace)).prepare()
    _print_split_summaries(summaries)
    return 0


def _data_summary(workspace: Path) -> int:
    from .acquisition import MVSeg2023DatasetPreparer

    summaries = MVSeg2023DatasetPreparer(_mvseg_resource(workspace)).summarize()
    _print_split_summaries(summaries)
    return 0


def _mock_mitral_fsi(valve_obj: Path, workspace: Path) -> int:
    from .mitral_mock import generate_and_run_mock_case

    summary = generate_and_run_mock_case(valve_obj=valve_obj, workspace=workspace)
    print(json.dumps(summary.to_dict(), indent=2))
    return 0


def _render_mock_mitral_fsi(case_dir: Path | None, workspace: Path) -> int:
    from .mitral_mock import render_mock_case

    resolved_case_dir = case_dir or (workspace / "artifacts" / "mock_mitral_uris_fsi")
    summary = render_mock_case(resolved_case_dir)
    print(json.dumps(summary.to_dict(), indent=2))
    return 0


def _view_mock_mitral_fsi(case_dir: Path | None, workspace: Path, cutaway: bool = False) -> int:
    from .mitral_mock import view_mock_case

    resolved_case_dir = case_dir or (workspace / "artifacts" / "mock_mitral_uris_fsi")
    view_mock_case(resolved_case_dir, cutaway=cutaway)
    return 0


def _render_blood_flow_mp4(case_dir: Path | None, workspace: Path,
                           fps: int = 30, dpi: int = 150) -> int:
    from .blood_flow_mp4 import render_blood_flow_mp4

    resolved_case_dir = case_dir or (workspace / "artifacts" / "mock_mitral_uris_fsi")
    result = render_blood_flow_mp4(resolved_case_dir, fps=fps, dpi=dpi)
    print(json.dumps(result.to_dict(), indent=2))
    return 0


def _print_split_summaries(summaries) -> None:
    print(json.dumps(
        {
            "dataset": "MVSeg2023",
            "splits": [
                {
                    "name": summary.name,
                    "root": str(summary.root),
                    "pair_count": summary.pair_count,
                    "image_suffix": summary.image_suffix,
                    "label_suffix": summary.label_suffix,
                }
                for summary in summaries
            ],
        },
        indent=2,
    ))


if __name__ == "__main__":
    import os
    os._exit(main())
