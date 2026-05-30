#!/usr/bin/env python3
"""Run all baseline policies for each (workflow, quality) experiment — 18 jobs total."""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.baselines import BASELINE_SOLVERS  # noqa: E402

EXPERIMENT_RUNS: list[tuple[str, str]] = [
    (workflow, quality)
    for workflow in ("workflow1", "workflow2")
    for quality in ("Q1", "Q2", "Q3")
]

BASELINE_METHODS = sorted(BASELINE_SOLVERS)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        description=(
            f"Run {len(EXPERIMENT_RUNS) * len(BASELINE_METHODS)} baseline jobs: "
            "each workflow×quality × each of single_cloud, logical_optimal, greedy."
        )
    )
    parser.add_argument(
        "--skip-data-gen",
        action="store_true",
        help="Do not regenerate synthetic CSVs before running",
    )
    parser.add_argument(
        "--heldout-eval",
        action="store_true",
        help="Use calibration scenarios for deployment selection and held-out scenarios for metrics",
    )
    args = parser.parse_args()

    if not args.skip_data_gen:
        subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "generate_synthetic_data.py")],
            check=True,
            cwd=str(ROOT),
        )
        subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "populate_from_measurements.py")],
            check=True,
            cwd=str(ROOT),
        )

    script = ROOT / "scripts" / "run_simulation.py"
    python = sys.executable
    total = len(EXPERIMENT_RUNS) * len(BASELINE_METHODS)
    step = 0

    for workflow, quality in EXPERIMENT_RUNS:
        for method in BASELINE_METHODS:
            step += 1
            cmd = [
                python,
                str(script),
                "--workflow",
                workflow,
                "--quality",
                quality,
                "--method",
                method,
            ]
            if args.heldout_eval:
                cmd.append("--heldout-eval")
            logging.info("[%d/%d] %s", step, total, " ".join(cmd))
            subprocess.run(cmd, check=True, cwd=str(ROOT))

    logging.info("All %d baseline runs finished.", total)


if __name__ == "__main__":
    main()
