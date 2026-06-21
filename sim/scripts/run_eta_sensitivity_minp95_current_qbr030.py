#!/usr/bin/env python3
"""Run eta sensitivity analysis on the current min-P95 budget setup."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

ETAS = ("0.075", "0.10", "0.125", "0.15")
WORKFLOWS = ("workflow1", "workflow2")
QUALITIES = ("Q1", "Q2", "Q3")
BASELINES = ("single_cloud", "greedy", "dpgm", "mtgp")
RESULTS_ROOT = ROOT / "results" / "sensitivity_eta_minp95_current_qbr030"
FAILURES_PATH = ROOT / "results" / "experiment_logs" / "sensitivity_eta_minp95_current_qbr030_failures.csv"


def eta_slug(eta: str) -> str:
    return eta.replace(".", "p")


def run_command(cmd: list[str], label: str, failures: list[dict[str, str]]) -> None:
    print(f"START {label}: {' '.join(cmd)}", flush=True)
    t0 = time.perf_counter()
    try:
        subprocess.run(cmd, cwd=str(ROOT), check=True, env={**os.environ, "MPLCONFIGDIR": "/private/tmp/matplotlib-cache"})
    except subprocess.CalledProcessError as exc:
        elapsed = time.perf_counter() - t0
        print(f"FAIL {label}: exit={exc.returncode} elapsed={elapsed:.1f}s", flush=True)
        failures.append(
            {
                "label": label,
                "returncode": str(exc.returncode),
                "elapsed_sec": f"{elapsed:.3f}",
            }
        )
        FAILURES_PATH.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(failures).to_csv(FAILURES_PATH, index=False)
        return
    print(f"DONE {label}: elapsed={time.perf_counter() - t0:.1f}s", flush=True)


def main() -> None:
    failures: list[dict[str, str]] = []
    for eta in ETAS:
        eta_root = RESULTS_ROOT / f"eta_{eta_slug(eta)}"
        for workflow in WORKFLOWS:
            for quality in QUALITIES:
                for method in BASELINES:
                    out = eta_root / f"{workflow}_{quality}" / method
                    if (out / "selected_plan.json").exists():
                        print(f"SKIP eta={eta} {workflow}/{quality}/{method}", flush=True)
                        continue
                    run_command(
                        [
                            sys.executable,
                            "scripts/run_simulation.py",
                            "--workflow",
                            workflow,
                            "--quality",
                            quality,
                            "--method",
                            method,
                            "--heldout-eval",
                            "--results-dir",
                            str(out),
                            "--eta",
                            eta,
                        ],
                        f"eta={eta} {workflow}/{quality}/{method}",
                        failures,
                    )

                out = eta_root / f"{workflow}_{quality}" / "decomposition"
                if (out / "selected_plan.json").exists():
                    print(f"SKIP eta={eta} {workflow}/{quality}/decomposition", flush=True)
                    continue
                run_command(
                    [
                        sys.executable,
                        "scripts/run_simulation.py",
                        "--workflow",
                        workflow,
                        "--quality",
                        quality,
                        "--method",
                        "decomposition",
                        "--heldout-eval",
                        "--results-dir",
                        str(out),
                        "--eta",
                        eta,
                        "--initial-active-fraction",
                        "0.3",
                        "--initial-active-strategy",
                        "qbr",
                        "--disable-initializer-selection",
                    ],
                    f"eta={eta} {workflow}/{quality}/decomposition",
                    failures,
                )

    print("ETA_SENSITIVITY_DONE", flush=True)
    if failures:
        print(f"Failures written to {FAILURES_PATH}", flush=True)


if __name__ == "__main__":
    main()
