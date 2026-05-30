"""Helpers for leakage-resistant simulation experiments."""

from __future__ import annotations

from collections import defaultdict

from src.schemas import Scenario


def split_scenarios_by_query(
    scenarios: list[Scenario],
    *,
    calibration_fraction: float = 0.5,
) -> tuple[list[Scenario], list[Scenario]]:
    """Split each query's scenario samples into calibration and held-out test sets."""
    by_query: dict[str, list[Scenario]] = defaultdict(list)
    for scenario in scenarios:
        by_query[scenario.query_id].append(scenario)

    calibration: list[Scenario] = []
    test: list[Scenario] = []
    for query_id in sorted(by_query):
        group = sorted(by_query[query_id], key=lambda s: s.scenario_id)
        if len(group) == 1:
            calibration.extend(group)
            continue
        n_calib = max(1, min(len(group) - 1, round(len(group) * calibration_fraction)))
        calibration.extend(group[:n_calib])
        test.extend(group[n_calib:])

    return calibration, test
