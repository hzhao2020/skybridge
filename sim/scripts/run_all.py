#!/usr/bin/env python3
"""Run the 6 paper experiments: one isolated run per (workflow, quality)."""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import CONFIG_DIR, RESULTS_DIR, load_yaml  # noqa: E402

WORKFLOWS = ["workflow1", "workflow2"]
QUALITIES = ["Q1", "Q2", "Q3"]

# Exactly 6 runs: workflow1×Q1–Q3 and workflow2×Q1–Q3 (not crossed with multiple methods).
EXPERIMENT_RUNS: list[tuple[str, str]] = [
    (workflow, quality) for workflow in WORKFLOWS for quality in QUALITIES
]


def _default_method() -> str:
    cfg = load_yaml(CONFIG_DIR / "default.yaml")
    return str(cfg.get("default_solver_method", "decomposition"))


def _run_dir(workflow: str, quality: str, method: str) -> Path:
    return RESULTS_DIR / f"{workflow}_{quality}" / method


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        description=(
            "Run 6 SkyFlow experiments (workflow1/2 × Q1/Q2/Q3). "
            "Each job uses only that workflow's queries and writes to its own results folder."
        )
    )
    parser.add_argument(
        "--method",
        choices=["full_milp", "decomposition"],
        default=None,
        help=f"Solver method for all 6 runs (default: { _default_method()!r} from configs/default.yaml)",
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
    method = args.method or _default_method()

    if not args.skip_data_gen:
        gen_script = ROOT / "scripts" / "generate_synthetic_data.py"
        subprocess.run([sys.executable, str(gen_script)], check=True, cwd=str(ROOT))
        pop_script = ROOT / "scripts" / "populate_from_measurements.py"
        subprocess.run([sys.executable, str(pop_script)], check=True, cwd=str(ROOT))

    script = ROOT / "scripts" / "run_simulation.py"
    python = sys.executable

    logging.info(
        "Starting %d experiments (method=%s): %s",
        len(EXPERIMENT_RUNS),
        method,
        ", ".join(f"{w}/{q}" for w, q in EXPERIMENT_RUNS),
    )

    for i, (workflow, quality) in enumerate(EXPERIMENT_RUNS, start=1):
        out = _run_dir(workflow, quality, method)
        cmd = [
            python,
            str(script),
            "--workflow",
            workflow,
            "--quality",
            quality,
            "--method",
            method,
            "--results-dir",
            str(out),
        ]
        if args.heldout_eval:
            cmd.append("--heldout-eval")
        logging.info("[%d/6] %s", i, " ".join(cmd))
        subprocess.run(cmd, check=True, cwd=str(ROOT))

    logging.info(
        "All 6 experiments finished. Outputs under %s/<workflow>_<Q*>/<method>/",
        RESULTS_DIR,
    )


if __name__ == "__main__":
    main()
