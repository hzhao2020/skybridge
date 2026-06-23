#!/usr/bin/env python3
"""Run SLA multiplier sensitivity for the current min-P95 qbr030 setup."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

MULTIPLIERS = ("0.8", "0.9", "1.1", "1.2")
WORKFLOWS = ("workflow1", "workflow2")
QUALITIES = ("Q1", "Q2", "Q3")
RESULTS_ROOT = ROOT / "results" / "sensitivity_sla_multiplier_minp95_current_qbr030"
LOG_DIR = ROOT / "results" / "experiment_logs"
FAILURES_PATH = LOG_DIR / "sensitivity_sla_multiplier_minp95_current_qbr030_failures.csv"


def multiplier_slug(multiplier: str) -> str:
    return multiplier.replace(".", "p")


def build_jobs(multipliers: tuple[str, ...]) -> list[dict[str, str]]:
    jobs: list[dict[str, str]] = []
    for multiplier in multipliers:
        mult_root = RESULTS_ROOT / f"sla_{multiplier_slug(multiplier)}"
        for workflow in WORKFLOWS:
            for quality in QUALITIES:
                out = mult_root / f"{workflow}_{quality}" / "decomposition"
                jobs.append(
                    {
                        "multiplier": multiplier,
                        "workflow": workflow,
                        "quality": quality,
                        "results_dir": str(out),
                    }
                )
    return jobs


def run_job(job: dict[str, str]) -> dict[str, str]:
    out = Path(job["results_dir"])
    label = f"sla={job['multiplier']} {job['workflow']}/{job['quality']}/decomposition"
    if (out / "selected_plan.json").exists():
        return {
            **job,
            "label": label,
            "status": "SKIP",
            "returncode": "0",
            "elapsed_sec": "0.000",
        }

    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    cmd = [
        sys.executable,
        "scripts/run_simulation.py",
        "--workflow",
        job["workflow"],
        "--quality",
        job["quality"],
        "--method",
        "decomposition",
        "--heldout-eval",
        "--results-dir",
        str(out),
        "--eta",
        "0.05",
        "--sla-multiplier",
        job["multiplier"],
        "--initial-active-fraction",
        "0.3",
        "--initial-active-strategy",
        "qbr",
        "--disable-initializer-selection",
    ]
    log_name = (
        f"sla_{multiplier_slug(job['multiplier'])}_"
        f"{job['workflow']}_{job['quality']}_decomposition"
    )
    stdout_path = LOG_DIR / f"{log_name}.stdout.log"
    stderr_path = LOG_DIR / f"{log_name}.stderr.log"
    env = {**os.environ, "MPLCONFIGDIR": str(ROOT / "results" / "matplotlib-cache")}
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr:
        proc = subprocess.run(cmd, cwd=str(ROOT), env=env, stdout=stdout, stderr=stderr)
    elapsed = time.perf_counter() - t0
    return {
        **job,
        "label": label,
        "status": "DONE" if proc.returncode == 0 else "FAIL",
        "returncode": str(proc.returncode),
        "elapsed_sec": f"{elapsed:.3f}",
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--multipliers",
        default=",".join(MULTIPLIERS),
        help="Comma-separated SLA multipliers to run",
    )
    args = parser.parse_args()

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    multipliers = tuple(
        item.strip() for item in args.multipliers.split(",") if item.strip()
    )
    jobs = build_jobs(multipliers)
    rows: list[dict[str, str]] = []
    print(
        f"SLA_MULTIPLIER_SENSITIVITY_START jobs={len(jobs)} workers={args.workers}",
        flush=True,
    )
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [pool.submit(run_job, job) for job in jobs]
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(
                f"{row['status']} {row['label']} exit={row['returncode']} "
                f"elapsed={row['elapsed_sec']}s",
                flush=True,
            )
            pd.DataFrame(rows).to_csv(
                LOG_DIR / "sensitivity_sla_multiplier_minp95_current_qbr030_progress.csv",
                index=False,
            )

    failures = [row for row in rows if row["status"] == "FAIL"]
    if failures:
        pd.DataFrame(failures).to_csv(FAILURES_PATH, index=False)
        raise SystemExit(f"{len(failures)} SLA multiplier runs failed; see {FAILURES_PATH}")
    print("SLA_MULTIPLIER_SENSITIVITY_DONE", flush=True)


if __name__ == "__main__":
    main()
