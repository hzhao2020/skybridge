"""Workflow 2：Database DAG 路径仿真、Sky CVaR–MILP、Baselines 与评估。"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .baseline import (
    BaselineResultWf2,
    deterministic_optimal_baseline_wf2,
    logical_optimal_baseline_wf2,
    mc_violation_counts_wf2,
    run_all_baselines_wf2,
    single_cloud_baseline_wf2,
)
from .candidates import candidates_for_logical_op, enumerate_candidates_wf2
from .evaluation import (
    EmpiricalDeploymentMetricsWf2,
    evaluate_deployment_empirical_wf2,
    format_metrics_lines_wf2,
    print_metrics_report_wf2,
)
from .utils import (
    WF2LogicalOp,
    WF2ParallelModality,
    WF2PathId,
    WF2PhysicalNode,
    WF2_PATH_CAPTION,
    WF2_PATH_LABEL,
    WF2_PATH_OCR,
    WF2_PATH_SPEECH,
    default_weights_for_path,
    end_to_end_cost_exclusive_path,
    end_to_end_latency_exclusive_path,
    generate_realistic_queries_wf2,
    path_logical_ops,
    plugin_mean_data_conversion_ratios_wf2,
    propagate_path_sizes,
    reference_deployment_exclusive_path,
    sample_wf2_logical_ratio,
    sample_wf2_path_rho,
    validate_exclusive_path_nodes,
    wf2_node_utility,
)

if TYPE_CHECKING:
    from .sky import (
        DecompositionResultWf2,
        JointScenarioWf2,
        MilpSolutionWf2,
        build_joint_scenarios_wf2,
        compute_linear_aggregate_wf2,
        locality_greedy_warm_start_indices_wf2,
        prepare_coefficients_wf2,
        run_sky_deployment_wf2,
        scenario_adaptive_decomposition_wf2,
        sky_ablation_settings_wf2,
    )

_SKY_EXPORTS: frozenset[str] = frozenset(
    {
        "DecompositionResultWf2",
        "JointScenarioWf2",
        "MilpSolutionWf2",
        "build_joint_scenarios_wf2",
        "compute_linear_aggregate_wf2",
        "locality_greedy_warm_start_indices_wf2",
        "prepare_coefficients_wf2",
        "run_sky_deployment_wf2",
        "scenario_adaptive_decomposition_wf2",
        "sky_ablation_settings_wf2",
    }
)


def __getattr__(name: str) -> Any:
    """按需加载 ``sky``（依赖 gurobipy），避免顶层强依赖。"""
    if name in _SKY_EXPORTS:
        from . import sky as _sky

        return getattr(_sky, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "BaselineResultWf2",
    "DecompositionResultWf2",
    "JointScenarioWf2",
    "MilpSolutionWf2",
    "WF2LogicalOp",
    "WF2ParallelModality",
    "WF2PathId",
    "WF2PhysicalNode",
    "WF2_PATH_CAPTION",
    "WF2_PATH_LABEL",
    "WF2_PATH_OCR",
    "WF2_PATH_SPEECH",
    "build_joint_scenarios_wf2",
    "candidates_for_logical_op",
    "compute_linear_aggregate_wf2",
    "default_weights_for_path",
    "deterministic_optimal_baseline_wf2",
    "end_to_end_cost_exclusive_path",
    "end_to_end_latency_exclusive_path",
    "EmpiricalDeploymentMetricsWf2",
    "enumerate_candidates_wf2",
    "evaluate_deployment_empirical_wf2",
    "format_metrics_lines_wf2",
    "generate_realistic_queries_wf2",
    "locality_greedy_warm_start_indices_wf2",
    "logical_optimal_baseline_wf2",
    "mc_violation_counts_wf2",
    "path_logical_ops",
    "plugin_mean_data_conversion_ratios_wf2",
    "prepare_coefficients_wf2",
    "print_metrics_report_wf2",
    "propagate_path_sizes",
    "reference_deployment_exclusive_path",
    "run_all_baselines_wf2",
    "run_sky_deployment_wf2",
    "sample_wf2_logical_ratio",
    "sample_wf2_path_rho",
    "scenario_adaptive_decomposition_wf2",
    "single_cloud_baseline_wf2",
    "sky_ablation_settings_wf2",
    "validate_exclusive_path_nodes",
    "wf2_node_utility",
]
