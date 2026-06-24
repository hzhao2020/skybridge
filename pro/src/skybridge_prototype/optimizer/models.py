from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RuntimeEndpoint:
    endpoint_id: str
    role: str
    provider_key: str
    provider: str
    region: str
    base_latency_sec: float
    latency_per_mb: float
    fixed_cost: float
    cost_per_mb: float
    storage_cost_per_mb: float = 0.0
    capability: float = 1.0
    model_name: str | None = None


@dataclass(frozen=True)
class RelayLink:
    endpoint_id: str
    upload_bandwidth_mb_per_sec: float
    download_bandwidth_mb_per_sec: float
    upload_rtt_sec: float
    download_rtt_sec: float
    upload_cost_per_gb: float = 0.0
    download_cost_per_gb: float = 0.0


@dataclass(frozen=True)
class WorkflowSpec:
    name: str
    nodes: list[str]
    edges: list[tuple[str, str]]


@dataclass(frozen=True)
class PlanningQuery:
    query_id: str
    video_size_mb: float
    sla_sec: float


@dataclass(frozen=True)
class PlanningScenario:
    scenario_id: str
    query_id: str
    node_latency_multiplier: dict[str, float] = field(default_factory=dict)
    upload_bandwidth_multiplier: dict[str, float] = field(default_factory=dict)
    download_bandwidth_multiplier: dict[str, float] = field(default_factory=dict)
    output_ratio: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class RuntimeProfile:
    workflow: WorkflowSpec
    endpoints: list[RuntimeEndpoint]
    relay_links: dict[str, RelayLink]
    queries: list[PlanningQuery]
    scenarios: list[PlanningScenario]


@dataclass(frozen=True)
class PlannerConfig:
    eta: float = 0.05
    initial_active_fraction: float = 0.25
    active_batch_fraction: float = 0.10
    initial_active_strategy: str = "qbu"
    violation_tolerance: float = 1e-6
    max_iterations: int = 30
    random_seed: int = 42
    gurobi_time_limit_sec: float = 60.0
    gurobi_mip_gap: float = 0.01
    latency_tiebreaker_weight: float = 1e-6
    mtgp_population_size: int = 28
    mtgp_generations: int = 16
    mtgp_elite_count: int = 2
    mtgp_mutation_rate: float = 0.15


@dataclass(frozen=True)
class PlanningAssignment:
    role: str
    endpoint_id: str
    provider_key: str
    provider: str
    region: str
    model_name: str | None = None


@dataclass(frozen=True)
class PlanningResult:
    method: str
    status: str
    assignments: list[PlanningAssignment]
    selected_providers: dict[str, str]
    objective_value: float
    expected_cost: float
    avg_latency: float
    p95_latency: float
    p99_latency: float
    violation_rate: float
    cvar_value: float
    solver_runtime_sec: float
    num_iterations: int = 1
    active_scenario_count: int = 0
    convergence_history: list[dict[str, Any]] = field(default_factory=list)
    per_query_scenario_metrics: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return {key: to_jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    return value
