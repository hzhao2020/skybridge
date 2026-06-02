#!/usr/bin/env python3
"""Append synthetic queries/scenarios so each workflow-quality group reaches Q=2000."""

from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import DATA_DIR, RESULTS_DIR, load_default_config  # noqa: E402
from src.data_generator import _generate_queries  # noqa: E402


def _query_number(query_id: str) -> int:
    return int(query_id.rsplit("q", 1)[1])


def _backup(path: Path, backup_dir: Path) -> None:
    backup_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, backup_dir / path.name)


def _append_queries(
    queries: pd.DataFrame,
    *,
    target: int,
    seed: int,
) -> pd.DataFrame:
    cfg = load_default_config()
    cfg["num_queries_per_workflow_quality"] = target
    rng = np.random.default_rng(seed)
    generated = _generate_queries(rng, cfg)

    additions: list[pd.DataFrame] = []
    for (workflow, quality), group in queries.groupby(["workflow", "quality_level"]):
        existing_ids = set(group["query_id"].astype(str))
        current = len(group)
        if current >= target:
            continue

        gen_group = generated[
            (generated["workflow"] == workflow)
            & (generated["quality_level"] == quality)
        ].copy()
        gen_group = gen_group[gen_group["query_id"].map(_query_number) >= current]
        gen_group = gen_group.head(target - current).copy()
        if len(gen_group) != target - current:
            raise RuntimeError(
                f"could not generate enough queries for {workflow}/{quality}: "
                f"need {target - current}, got {len(gen_group)}"
            )
        gen_group["query_id"] = [
            f"{workflow}_{quality}_q{i:04d}" for i in range(current, target)
        ]
        if existing_ids.intersection(set(gen_group["query_id"])):
            raise RuntimeError(f"duplicate query ids while extending {workflow}/{quality}")
        additions.append(gen_group[queries.columns])

    if not additions:
        return queries
    return pd.concat([queries, *additions], ignore_index=True)


def _append_scenarios(
    scenarios: pd.DataFrame,
    queries_before: pd.DataFrame,
    queries_after: pd.DataFrame,
) -> pd.DataFrame:
    additions: list[pd.DataFrame] = []
    scenario_counts = scenarios.groupby("query_id").size()
    source_by_query = {
        str(query_id): group.copy()
        for query_id, group in scenarios.groupby("query_id", sort=False)
    }
    existing_query_ids = set(queries_before["query_id"].astype(str))

    for (workflow, quality), after_group in queries_after.groupby(["workflow", "quality_level"]):
        before_group = queries_before[
            (queries_before["workflow"] == workflow)
            & (queries_before["quality_level"] == quality)
        ].sort_values("query_id")
        before_ids = list(before_group["query_id"].astype(str))
        if not before_ids:
            raise RuntimeError(f"no source scenarios for {workflow}/{quality}")

        new_ids = [
            qid
            for qid in after_group["query_id"].astype(str)
            if qid not in existing_query_ids
        ]
        for offset, new_id in enumerate(new_ids):
            source_id = before_ids[offset % len(before_ids)]
            source = source_by_query[source_id].copy()
            expected = int(scenario_counts.loc[source_id])
            if len(source) != expected:
                raise RuntimeError(f"incomplete source scenarios for {source_id}")
            source["query_id"] = new_id
            source["scenario_id"] = [f"{new_id}_s{i:03d}" for i in range(len(source))]
            additions.append(source[scenarios.columns])

    if not additions:
        return scenarios
    combined = pd.concat([scenarios, *additions], ignore_index=True)
    counts_after = combined.groupby("query_id").size()
    too_small = counts_after[counts_after < counts_after.max()]
    if not too_small.empty:
        raise RuntimeError(f"scenario count mismatch after extension: {too_small.head()}")
    return combined


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=int, default=2000)
    parser.add_argument("--seed-offset", type=int, default=100_000)
    parser.add_argument("--no-backup", action="store_true")
    args = parser.parse_args()

    cfg = load_default_config()
    seed = int(cfg.get("random_seed", 42)) + args.seed_offset

    queries_path = DATA_DIR / "queries.csv"
    scenarios_path = DATA_DIR / "scenarios.csv"
    queries_before = pd.read_csv(queries_path)
    scenarios_before = pd.read_csv(scenarios_path)

    if not args.no_backup:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = RESULTS_DIR / "experiment_logs" / f"pre_q2000_extension_{stamp}"
        _backup(queries_path, backup_dir)
        _backup(scenarios_path, backup_dir)
        print(f"Backed up current CSVs to {backup_dir}")

    queries_after = _append_queries(queries_before, target=args.target, seed=seed)
    scenarios_after = _append_scenarios(
        scenarios_before,
        queries_before,
        queries_after,
    )

    queries_after.to_csv(queries_path, index=False)
    scenarios_after.to_csv(scenarios_path, index=False)

    print(
        "queries:",
        len(queries_before),
        "->",
        len(queries_after),
        "; scenarios:",
        len(scenarios_before),
        "->",
        len(scenarios_after),
    )
    print(queries_after.groupby(["workflow", "quality_level"]).size().to_string())


if __name__ == "__main__":
    main()
