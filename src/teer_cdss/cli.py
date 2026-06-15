"""Command-line entrypoint for the TEER pipeline scaffold."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .schemas import TEERPipelineConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the TEER decision-support pipeline scaffold.")
    parser.add_argument("dicom_paths", nargs="*", type=Path, help="Input DICOM paths for one TEE series.")
    parser.add_argument("--workspace", type=Path, default=Path.cwd(), help="Artifact workspace root.")
    parser.add_argument("--operator", type=str, default="system", help="Operator identifier for anonymization audit.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
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


if __name__ == "__main__":
    raise SystemExit(main())
