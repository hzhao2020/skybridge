"""Pydantic schemas for SkyFlow simulation."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class Endpoint(BaseModel):
    endpoint_id: str
    logical_operation: str
    provider: str
    region: str
    quality_level: str
    model_name: str | None = None
    base_latency_sec: float
    latency_per_mb: float
    fixed_cost: float
    cost_per_mb: float
    storage_cost_per_mb: float
    is_virtual: bool = False


class NetworkLink(BaseModel):
    src_endpoint_id: str
    dst_endpoint_id: str
    bandwidth_mb_per_sec: float
    rtt_sec: float
    egress_cost_per_gb: float


class LogicalNode(BaseModel):
    name: str
    is_virtual: bool = False


class LogicalEdge(BaseModel):
    src: str
    dst: str


class WorkflowDAG(BaseModel):
    name: str
    description: str = ""
    nodes: list[LogicalNode]
    edges: list[LogicalEdge]
    virtual_nodes: list[str] = Field(default_factory=list)
    query_metadata_targets: list[str] = Field(default_factory=list)

    def node_names(self) -> list[str]:
        return [n.name for n in self.nodes]

    def is_virtual(self, node: str) -> bool:
        return node in self.virtual_nodes

    def compute_nodes(self) -> list[str]:
        return [n.name for n in self.nodes if not self.is_virtual(n.name)]

    def predecessors(self, node: str) -> list[str]:
        return [e.src for e in self.edges if e.dst == node]

    def successors(self, node: str) -> list[str]:
        return [e.dst for e in self.edges if e.src == node]


class Query(BaseModel):
    query_id: str
    workflow: str
    quality_level: str
    split: str = "train"
    video_size_mb: float
    video_duration_sec: float
    fps: float
    sla_sec: float


class Scenario(BaseModel):
    query_id: str
    scenario_id: str
    rho: dict[str, float] = Field(default_factory=dict)
    database_output_tokens: float | None = None
    q_a_output_tokens: float | None = None
    caption_output_tokens_per_frame: float | None = None
    exec_latency_multiplier: dict[str, float] = Field(default_factory=dict)
    bandwidth_multiplier: dict[str, float] = Field(default_factory=dict)
    rtt_multiplier: dict[str, float] = Field(default_factory=dict)
    exec_stress: float = 1.0
    bw_stress: float = 1.0
    rtt_stress: float = 1.0


class AblationConfig(BaseModel):
    enable_cvar: bool = True
    enable_network_latency: bool = True
    enable_network_cost: bool = True
    enable_storage_cost: bool = True
    enable_client_upload_download: bool = True
    enable_decomposition: bool = True
    fixed_provider: str | None = None
    fixed_region: str | None = None


class SolverConfig(BaseModel):
    random_seed: int = 42
    eta: float = 0.05
    top_k: int = 0
    initial_active_fraction: float = 0.20
    initial_active_strategy: str = "qbr"
    initializer_validation_fraction: float = 0.20
    initializer_selection_candidates: list[str] = Field(
        default_factory=lambda: [
            "qbr",
        ]
    )
    active_batch_fraction: float = 0.05
    max_iterations: int = 100
    gurobi_time_limit_sec: float = 300.0
    gurobi_mip_gap: float = 0.01
    latency_tiebreaker_weight: float = 0.0
    endpoint_tiebreaker_weight: float = 0.0
    mtgp_population_size: int = 12
    mtgp_generations: int = 6
    mtgp_max_depth: int = 4
    mtgp_tournament_size: int = 3
    mtgp_elite_count: int = 2
    mtgp_crossover_rate: float = 0.80
    mtgp_mutation_rate: float = 0.15
    ablation: AblationConfig = Field(default_factory=AblationConfig)


class DeploymentAssignment(BaseModel):
    logical_node: str
    endpoint_id: str
    provider: str
    region: str
    model_name: str | None = None


class OptimizationResult(BaseModel):
    workflow: str
    quality_level: str
    method: str
    assignments: list[DeploymentAssignment]
    objective_value: float
    expected_cost: float
    avg_latency: float
    p95_latency: float
    p99_latency: float
    violation_rate: float
    cvar_value: float
    solver_runtime_sec: float
    status: str
    num_iterations: int = 1
    active_scenario_count: int = 0
    active_path_cut_count: int = 0
    selected_initializer: str | None = None
    initializer_selection_history: list[dict[str, Any]] = Field(default_factory=list)
    convergence_history: list[dict[str, Any]] = Field(default_factory=list)
    per_query_scenario_metrics: list[dict[str, Any]] = Field(default_factory=list)


class ConvergenceRecord(BaseModel):
    workflow: str
    quality_level: str
    iteration: int
    active_scenario_count: int
    active_path_cut_count: int = 0
    objective_value: float
    max_violation: float
    max_new_cut_violation: float = 0.0
    num_violated_scenarios: int
    num_violated_cuts: int = 0
    active_cut_violation_count: int = 0
    active_cut_batch_size: int = 0
    runtime_sec: float
