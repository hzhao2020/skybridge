from pathlib import Path

from skybridge_prototype.optimizer import load_runtime_profile, plan_workflow


def test_greedy_planner_returns_broker_overrides() -> None:
    profile, config = load_runtime_profile(Path("configs/planner.mock.json"))

    result = plan_workflow("greedy", profile, config)

    assert set(result.selected_providers) == {"shot_detection", "split_sample", "caption", "qa"}
    assert result.status == "optimal"
    assert result.avg_latency > 0
    assert result.expected_cost > 0


def test_decomposition_planner_runs_on_runtime_profile() -> None:
    profile, config = load_runtime_profile(Path("configs/planner.mock.json"))

    result = plan_workflow("decomposition", profile, config)

    assert set(result.selected_providers) == {"shot_detection", "split_sample", "caption", "qa"}
    assert result.num_iterations >= 1
    assert result.active_scenario_count >= 1
    assert result.per_query_scenario_metrics


def test_mtgp_planner_runs_on_runtime_profile() -> None:
    profile, config = load_runtime_profile(Path("configs/planner.mock.json"))

    result = plan_workflow("mtgp", profile, config)

    assert set(result.selected_providers) == {"shot_detection", "split_sample", "caption", "qa"}
    assert result.num_iterations == config.mtgp_generations
    assert result.convergence_history
