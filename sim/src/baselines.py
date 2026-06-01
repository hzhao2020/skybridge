"""Baseline deployment policies: Single-Cloud, Murakkab-style, and Greedy."""

from __future__ import annotations

import logging
import time
from typing import Callable

import numpy as np

from src.capability import endpoint_capability_mu
from src.cost_latency import (
    compute_cvar_value,
    endpoints_by_operation,
    execution_cost,
    execution_latency,
    filter_endpoints,
    network_latency,
    network_transfer_cost,
    storage_cost,
)
from src.data_loader import get_virtual_endpoint_map, load_network_links
from src.data_propagation import output_data_sizes, propagate_data_sizes
from src.measurement.execution_latency import sampled_execution_latency
from src.evaluator import evaluate_deployment
from src.path_utils import enumerate_source_to_sink_paths, path_edges
from src.schemas import (
    DeploymentAssignment,
    Endpoint,
    OptimizationResult,
    Query,
    Scenario,
    SolverConfig,
    WorkflowDAG,
)

logger = logging.getLogger(__name__)

SolveFn = Callable[
    [
        WorkflowDAG,
        list[Endpoint],
        list[Query],
        list[Scenario],
        str,
        SolverConfig,
    ],
    OptimizationResult,
]

QueryScenarioData = tuple[Query, Scenario, dict[str, float], dict[str, float]]


def _node_candidates(
    workflow: WorkflowDAG,
    endpoints: list[Endpoint],
    quality_level: str,
    config: SolverConfig,
) -> dict[str, list[Endpoint]]:
    filtered = filter_endpoints(endpoints, quality_level, config.ablation)
    ops_map = endpoints_by_operation(filtered)
    out: dict[str, list[Endpoint]] = {}
    for node in workflow.compute_nodes():
        cands = ops_map.get(node, [])
        if not cands:
            raise ValueError(f"No candidates for {node} at quality {quality_level}")
        out[node] = cands
    return out


def _topological_compute_nodes(workflow: WorkflowDAG) -> list[str]:
    compute = set(workflow.compute_nodes())
    indegree = {n: len([p for p in workflow.predecessors(n) if p in compute]) for n in compute}
    ready = [n for n in compute if indegree[n] == 0]
    order: list[str] = []
    while ready:
        n = ready.pop(0)
        order.append(n)
        for succ in workflow.successors(n):
            if succ not in compute:
                continue
            indegree[succ] -= 1
            if indegree[succ] == 0:
                ready.append(succ)
    if len(order) != len(compute):
        raise ValueError("Workflow compute nodes are not acyclic")
    return order


def _placeholder_endpoint(candidates: list[Endpoint]) -> Endpoint:
    """Lowest estimated per-query cost endpoint (tie-breaker for unassigned nodes in Greedy)."""

    def key(ep: Endpoint) -> tuple[float, str]:
        est = ep.fixed_cost + ep.cost_per_mb * 100.0
        return (est, ep.endpoint_id)

    return min(candidates, key=key)


def _full_assignment(
    workflow: WorkflowDAG,
    node_candidates: dict[str, list[Endpoint]],
    partial: dict[str, Endpoint],
) -> dict[str, Endpoint]:
    """Complete assignment using placeholders for nodes not yet chosen."""
    full = dict(partial)
    for node in workflow.compute_nodes():
        if node not in full:
            full[node] = _placeholder_endpoint(node_candidates[node])
    return full


def _metrics_for_assignment(
    workflow: WorkflowDAG,
    assignment: dict[str, Endpoint],
    endpoints: list[Endpoint],
    queries: list[Query],
    scenarios: list[Scenario],
    quality_level: str,
    config: SolverConfig,
) -> dict:
    return evaluate_deployment(
        workflow=workflow,
        assignment=assignment,
        endpoints=endpoints,
        queries=queries,
        scenarios=scenarios,
        quality_level=quality_level,
        config=config,
    )


def _assignment_key(assignment: dict[str, Endpoint], workflow: WorkflowDAG) -> tuple[tuple[str, str], ...]:
    return tuple(
        sorted(
            (node, ep.endpoint_id)
            for node, ep in assignment.items()
            if not workflow.is_virtual(node)
        )
    )


class _BaselineEvaluationCache:
    """Exact evaluator with reusable query-scenario data for baseline searches."""

    def __init__(
        self,
        workflow: WorkflowDAG,
        endpoints: list[Endpoint],
        queries: list[Query],
        scenarios: list[Scenario],
        quality_level: str,
        config: SolverConfig,
    ) -> None:
        self.workflow = workflow
        self.endpoint_map = {e.endpoint_id: e for e in endpoints}
        self.virtual_map = get_virtual_endpoint_map(endpoints)
        self.network_index = {
            (l.src_endpoint_id, l.dst_endpoint_id): l for l in load_network_links()
        }
        self.quality_level = quality_level
        self.config = config
        self.paths = enumerate_source_to_sink_paths(workflow)
        self.path_edges = [path_edges(path) for path in self.paths]
        self.qs_data = _build_query_scenario_data(workflow, queries, scenarios, quality_level)
        self._scalar_cache: dict[tuple[tuple[str, str], ...], dict] = {}
        self._avg_node_cost_cache: dict[tuple[str, str], float] = {}
        self._avg_edge_cost_cache: dict[tuple[str, str, str], float] = {}

    def evaluate(
        self,
        assignment: dict[str, Endpoint],
        *,
        include_per_qs: bool = False,
    ) -> dict:
        key = _assignment_key(assignment, self.workflow)
        if not include_per_qs and key in self._scalar_cache:
            return self._scalar_cache[key]

        assign_full = dict(assignment)
        assign_full.setdefault("ClientSource", self.virtual_map["virtual_ClientSource"])
        assign_full.setdefault("ClientSink", self.virtual_map["virtual_ClientSink"])
        compute_assignment = {
            node: ep for node, ep in assign_full.items() if not self.workflow.is_virtual(node)
        }

        per_qs: list[dict] = []
        costs: list[float] = []
        latencies: list[float] = []
        slas: list[float] = []
        violations = 0
        total = 0

        for query, scenario, input_sizes, output_sizes in self.qs_data:
            cost = self._total_cost(compute_assignment, query, scenario, input_sizes, output_sizes)
            latency = self._critical_path_latency(
                assign_full,
                query,
                scenario,
                input_sizes,
                output_sizes,
            )
            violated = latency > query.sla_sec
            if violated:
                violations += 1
            total += 1
            costs.append(cost)
            latencies.append(latency)
            slas.append(query.sla_sec)
            if include_per_qs:
                per_qs.append(
                    {
                        "query_id": query.query_id,
                        "scenario_id": scenario.scenario_id,
                        "cost": cost,
                        "latency": latency,
                        "sla_sec": query.sla_sec,
                        "violated": violated,
                    }
                )

        n = max(total, 1)
        lat_arr = np.array(latencies) if latencies else np.array([0.0])
        cvar = (
            compute_cvar_value(latencies, slas, self.config.eta)
            if self.config.ablation.enable_cvar
            else 0.0
        )
        metrics = {
            "expected_cost": float(np.mean(costs)) if costs else 0.0,
            "avg_latency": float(np.mean(lat_arr)),
            "p95_latency": float(np.percentile(lat_arr, 95)) if len(lat_arr) else 0.0,
            "p99_latency": float(np.percentile(lat_arr, 99)) if len(lat_arr) else 0.0,
            "violation_rate": violations / n,
            "cvar_value": cvar,
            "per_qs": per_qs,
        }
        if not include_per_qs:
            self._scalar_cache[key] = metrics
        return metrics

    def expected_cost(self, assignment: dict[str, Endpoint]) -> float:
        cost = 0.0
        compute_assignment = {
            node: ep for node, ep in assignment.items() if not self.workflow.is_virtual(node)
        }
        for node, ep in compute_assignment.items():
            cost += self._average_node_cost(node, ep)

        for edge in self.workflow.edges:
            src, dst = edge.src, edge.dst
            if not self.config.ablation.enable_client_upload_download:
                if src == "ClientSource" or dst == "ClientSink":
                    continue
            src_ep = self._resolve_endpoint(src, compute_assignment)
            dst_ep = self._resolve_endpoint(dst, compute_assignment)
            if src_ep is None or dst_ep is None:
                continue
            cost += self._average_edge_cost(src, src_ep, dst_ep)
        return cost

    def _average_node_cost(self, node: str, endpoint: Endpoint) -> float:
        key = (node, endpoint.endpoint_id)
        cached = self._avg_node_cost_cache.get(key)
        if cached is not None:
            return cached
        costs = [
            execution_cost(endpoint, input_sizes.get(node, 0.0), output_sizes.get(node, 0.0))
            + storage_cost(
                endpoint,
                input_sizes.get(node, 0.0),
                output_sizes.get(node, 0.0),
                self.config.ablation.enable_storage_cost,
            )
            for _, _, input_sizes, output_sizes in self.qs_data
        ]
        value = float(np.mean(costs)) if costs else 0.0
        self._avg_node_cost_cache[key] = value
        return value

    def _average_edge_cost(
        self,
        src_node: str,
        src_ep: Endpoint,
        dst_ep: Endpoint,
    ) -> float:
        key = (src_node, src_ep.endpoint_id, dst_ep.endpoint_id)
        cached = self._avg_edge_cost_cache.get(key)
        if cached is not None:
            return cached
        link = self.network_index.get((src_ep.endpoint_id, dst_ep.endpoint_id))
        if link is None:
            value = 0.0
        else:
            costs = [
                network_transfer_cost(
                    link,
                    output_sizes.get(src_node, 0.0),
                    self.config.ablation.enable_network_cost,
                )
                for _, _, _, output_sizes in self.qs_data
            ]
            value = float(np.mean(costs)) if costs else 0.0
        self._avg_edge_cost_cache[key] = value
        return value

    def evaluate_feasible_cost(
        self,
        assignment: dict[str, Endpoint],
        max_violations: int,
    ) -> dict | None:
        assign_full = dict(assignment)
        assign_full.setdefault("ClientSource", self.virtual_map["virtual_ClientSource"])
        assign_full.setdefault("ClientSink", self.virtual_map["virtual_ClientSink"])
        compute_assignment = {
            node: ep for node, ep in assign_full.items() if not self.workflow.is_virtual(node)
        }

        costs: list[float] = []
        violations = 0
        total = 0

        for query, scenario, input_sizes, output_sizes in self.qs_data:
            latency = self._critical_path_latency(
                assign_full,
                query,
                scenario,
                input_sizes,
                output_sizes,
            )
            if latency > query.sla_sec:
                violations += 1
                if violations > max_violations:
                    return None
            total += 1
            costs.append(
                self._total_cost(
                    compute_assignment,
                    query,
                    scenario,
                    input_sizes,
                    output_sizes,
                )
            )

        n = max(total, 1)
        return {
            "expected_cost": float(np.mean(costs)) if costs else 0.0,
            "violation_rate": violations / n,
        }

    def _total_cost(
        self,
        assignment: dict[str, Endpoint],
        query: Query,
        scenario: Scenario,
        input_sizes: dict[str, float],
        output_sizes: dict[str, float],
    ) -> float:
        cost = 0.0
        for node, ep in assignment.items():
            inp = input_sizes.get(node, 0.0)
            out = output_sizes.get(node, 0.0)
            cost += execution_cost(ep, inp, out)
            cost += storage_cost(ep, inp, out, self.config.ablation.enable_storage_cost)

        for edge in self.workflow.edges:
            src, dst = edge.src, edge.dst
            if not self.config.ablation.enable_client_upload_download:
                if src == "ClientSource" or dst == "ClientSink":
                    continue
            src_ep = self._resolve_endpoint(src, assignment)
            dst_ep = self._resolve_endpoint(dst, assignment)
            if src_ep is None or dst_ep is None:
                continue
            link = self.network_index.get((src_ep.endpoint_id, dst_ep.endpoint_id))
            if link is None:
                continue
            cost += network_transfer_cost(
                link,
                output_sizes.get(src, 0.0),
                self.config.ablation.enable_network_cost,
            )
        return cost

    def _critical_path_latency(
        self,
        assignment: dict[str, Endpoint],
        query: Query,
        scenario: Scenario,
        input_sizes: dict[str, float],
        output_sizes: dict[str, float],
    ) -> float:
        latencies = [
            self._path_latency(path, edges, assignment, query, scenario, input_sizes, output_sizes)
            for path, edges in zip(self.paths, self.path_edges)
        ]
        return max(latencies) if latencies else 0.0

    def _path_latency(
        self,
        path: list[str],
        edges: list[tuple[str, str]],
        assignment: dict[str, Endpoint],
        query: Query,
        scenario: Scenario,
        input_sizes: dict[str, float],
        output_sizes: dict[str, float],
    ) -> float:
        total = 0.0
        for node in path:
            if self.workflow.is_virtual(node):
                continue
            ep = assignment.get(node)
            if ep is None:
                continue
            sampled = sampled_execution_latency(ep, query, scenario)
            if sampled is not None:
                total += sampled
            else:
                mult = scenario.exec_latency_multiplier.get(ep.endpoint_id, scenario.exec_stress)
                total += execution_latency(
                    ep,
                    input_sizes.get(node, 0.0),
                    output_sizes.get(node, 0.0),
                    mult,
                )

        for src, dst in edges:
            if not self.config.ablation.enable_client_upload_download:
                if src == "ClientSource" or dst == "ClientSink":
                    continue
            src_ep = self._resolve_endpoint(src, assignment)
            dst_ep = self._resolve_endpoint(dst, assignment)
            if src_ep is None or dst_ep is None:
                continue
            link = self.network_index.get((src_ep.endpoint_id, dst_ep.endpoint_id))
            if link is None:
                continue
            pair_key = f"{src_ep.endpoint_id}->{dst_ep.endpoint_id}"
            bw_mult = scenario.bandwidth_multiplier.get(pair_key, scenario.bw_stress)
            rtt_mult = scenario.rtt_multiplier.get(pair_key, scenario.rtt_stress)
            total += network_latency(
                link,
                output_sizes.get(src, 0.0),
                bw_mult,
                rtt_mult,
                self.config.ablation.enable_network_latency,
            )
        return total

    def _resolve_endpoint(
        self,
        node: str,
        assignment: dict[str, Endpoint],
    ) -> Endpoint | None:
        if node in assignment:
            return assignment[node]
        if self.workflow.is_virtual(node):
            if node == "ClientSource":
                return self.endpoint_map.get("client_source")
            if node == "ClientSink":
                return self.endpoint_map.get("client_sink")
            return self.endpoint_map.get(f"virtual_{node}")
        return None


def _build_query_scenario_data(
    workflow: WorkflowDAG,
    queries: list[Query],
    scenarios: list[Scenario],
    quality_level: str,
) -> list[QueryScenarioData]:
    scenario_by_q: dict[str, list[Scenario]] = {}
    for scenario in scenarios:
        scenario_by_q.setdefault(scenario.query_id, []).append(scenario)

    data: list[QueryScenarioData] = []
    for query in queries:
        if query.quality_level != quality_level:
            continue
        for scenario in scenario_by_q.get(query.query_id, []):
            input_sizes = propagate_data_sizes(workflow, query, scenario)
            output_sizes = output_data_sizes(input_sizes, scenario, workflow)
            data.append((query, scenario, input_sizes, output_sizes))
    return data


def _resolve_selected_endpoint(
    node: str,
    assignment: dict[str, Endpoint],
    virtual_map: dict[str, Endpoint],
    workflow: WorkflowDAG,
) -> Endpoint | None:
    if node in assignment:
        return assignment[node]
    if workflow.is_virtual(node):
        return virtual_map.get(f"virtual_{node}")
    return None


def _expected_incremental_cost_latency(
    *,
    workflow: WorkflowDAG,
    node: str,
    endpoint: Endpoint,
    partial: dict[str, Endpoint],
    virtual_map: dict[str, Endpoint],
    network_index: dict[tuple[str, str], object],
    qs_data: list[QueryScenarioData],
    config: SolverConfig,
) -> tuple[float, float]:
    total_cost = 0.0
    total_latency = 0.0
    n = 0

    for query, scenario, input_sizes, output_sizes in qs_data:
        inp = input_sizes.get(node, 0.0)
        out = output_sizes.get(node, 0.0)

        inc_cost = execution_cost(endpoint, inp, out)
        inc_cost += storage_cost(
            endpoint,
            inp,
            out,
            config.ablation.enable_storage_cost,
        )
        sampled = sampled_execution_latency(endpoint, query, scenario)
        if sampled is not None:
            inc_latency = sampled
        else:
            exec_mult = scenario.exec_latency_multiplier.get(
                endpoint.endpoint_id,
                scenario.exec_stress,
            )
            inc_latency = execution_latency(endpoint, inp, out, exec_mult)

        for pred in workflow.predecessors(node):
            pred_ep = _resolve_selected_endpoint(pred, partial, virtual_map, workflow)
            if pred_ep is None:
                continue
            link = network_index.get((pred_ep.endpoint_id, endpoint.endpoint_id))
            if link is None:
                continue
            pred_out = output_sizes.get(pred, 0.0)
            inc_cost += network_transfer_cost(
                link,
                pred_out,
                config.ablation.enable_network_cost,
            )
            pair_key = f"{pred_ep.endpoint_id}->{endpoint.endpoint_id}"
            bw_mult = scenario.bandwidth_multiplier.get(pair_key, scenario.bw_stress)
            rtt_mult = scenario.rtt_multiplier.get(pair_key, scenario.rtt_stress)
            inc_latency += network_latency(
                link,
                pred_out,
                bw_mult,
                rtt_mult,
                config.ablation.enable_network_latency,
            )

        total_cost += inc_cost
        total_latency += inc_latency
        n += 1

    denom = max(n, 1)
    return total_cost / denom, total_latency / denom


def _rank_violation_cost(violation_rate: float, expected_cost: float) -> tuple[float, float]:
    return (violation_rate, expected_cost)


def _to_result(
    *,
    workflow: WorkflowDAG,
    assignment: dict[str, Endpoint],
    quality_level: str,
    method: str,
    metrics: dict,
    runtime_sec: float,
) -> OptimizationResult:
    deployments = [
        DeploymentAssignment(
            logical_node=node,
            endpoint_id=ep.endpoint_id,
            provider=ep.provider,
            region=ep.region,
            model_name=ep.model_name,
        )
        for node, ep in assignment.items()
        if not workflow.is_virtual(node)
    ]
    return OptimizationResult(
        workflow=workflow.name,
        quality_level=quality_level,
        method=method,
        assignments=deployments,
        objective_value=float(metrics["violation_rate"]),
        expected_cost=float(metrics["expected_cost"]),
        avg_latency=float(metrics["avg_latency"]),
        p95_latency=float(metrics["p95_latency"]),
        p99_latency=float(metrics["p99_latency"]),
        violation_rate=float(metrics["violation_rate"]),
        cvar_value=float(metrics["cvar_value"]),
        solver_runtime_sec=runtime_sec,
        status="optimal",
        per_query_scenario_metrics=metrics.get("per_qs", []),
    )


def solve_logical_optimal(
    workflow: WorkflowDAG,
    endpoints: list[Endpoint],
    queries: list[Query],
    scenarios: list[Scenario],
    quality_level: str,
    config: SolverConfig,
) -> OptimizationResult:
    """
    LO: per logical node, pick the endpoint with highest μ_k (ignores cost/latency coupling).
    """
    t0 = time.perf_counter()
    node_candidates = _node_candidates(workflow, endpoints, quality_level, config)
    metrics_cache = _BaselineEvaluationCache(
        workflow, endpoints, queries, scenarios, quality_level, config
    )
    assignment: dict[str, Endpoint] = {}
    for node, cands in node_candidates.items():
        assignment[node] = max(cands, key=lambda ep: (endpoint_capability_mu(ep), ep.endpoint_id))

    metrics = metrics_cache.evaluate(assignment, include_per_qs=True)
    return _to_result(
        workflow=workflow,
        assignment=assignment,
        quality_level=quality_level,
        method="logical_optimal",
        metrics=metrics,
        runtime_sec=time.perf_counter() - t0,
    )


def _providers_covering_all_nodes(node_candidates: dict[str, list[Endpoint]]) -> list[str]:
    providers = {ep.provider for cands in node_candidates.values() for ep in cands}
    return sorted(
        provider
        for provider in providers
        if all(any(ep.provider == provider for ep in cands) for cands in node_candidates.values())
    )


def solve_single_cloud(
    workflow: WorkflowDAG,
    endpoints: list[Endpoint],
    queries: list[Query],
    scenarios: list[Scenario],
    quality_level: str,
    config: SolverConfig,
) -> OptimizationResult:
    """
    SC: build one provider-specific plan per cloud by choosing the lowest
    expected-cost endpoint for each logical operator. Pick the cheapest plan
    among providers satisfying SVR <= eta; otherwise pick the lowest-SVR plan.
    """
    t0 = time.perf_counter()
    node_candidates = _node_candidates(workflow, endpoints, quality_level, config)
    metrics_cache = _BaselineEvaluationCache(
        workflow, endpoints, queries, scenarios, quality_level, config
    )
    providers = _providers_covering_all_nodes(node_candidates)

    best_assign: dict[str, Endpoint] | None = None
    best_rank: tuple[float, float, float, str] | None = None
    best_metrics: dict | None = None

    for prov in providers:
        assignment: dict[str, Endpoint] = {}
        for node in _topological_compute_nodes(workflow):
            cands = [ep for ep in node_candidates[node] if ep.provider == prov]
            assignment[node] = min(
                cands,
                key=lambda ep: (
                    metrics_cache._average_node_cost(node, ep),
                    ep.endpoint_id,
                ),
            )

        metrics = metrics_cache.evaluate(assignment)
        violation = float(metrics["violation_rate"])
        cost = float(metrics["expected_cost"])
        if violation <= config.eta:
            rank = (0.0, cost, violation, prov)
        else:
            rank = (1.0, violation, cost, prov)
        if best_rank is None or rank < best_rank:
            best_rank = rank
            best_assign = assignment
            best_metrics = metrics

    if best_assign is None or best_metrics is None:
        raise RuntimeError(
            "Single-Cloud: no provider covers all logical operators"
        )

    logger.info(
        "Single-Cloud selected provider=%s violation=%.4f cost=%.4f",
        best_rank[3],
        best_metrics["violation_rate"],
        best_metrics["expected_cost"],
    )
    best_metrics = metrics_cache.evaluate(best_assign, include_per_qs=True)
    return _to_result(
        workflow=workflow,
        assignment=best_assign,
        quality_level=quality_level,
        method="single_cloud",
        metrics=best_metrics,
        runtime_sec=time.perf_counter() - t0,
    )


def solve_greedy(
    workflow: WorkflowDAG,
    endpoints: list[Endpoint],
    queries: list[Query],
    scenarios: list[Scenario],
    quality_level: str,
    config: SolverConfig,
) -> OptimizationResult:
    """
    Greedy: traverse the workflow DAG in topological order. For each logical node,
    choose the endpoint with the lowest expected incremental execution/storage and
    already-selected predecessor transfer cost. Ties break by lower expected
    incremental latency. Decisions are irrevocable and local.
    """
    t0 = time.perf_counter()
    node_candidates = _node_candidates(workflow, endpoints, quality_level, config)
    metrics_cache = _BaselineEvaluationCache(
        workflow, endpoints, queries, scenarios, quality_level, config
    )
    partial: dict[str, Endpoint] = {}

    for node in _topological_compute_nodes(workflow):
        best_ep: Endpoint | None = None
        best_rank: tuple[float, float, str] | None = None
        for ep in node_candidates[node]:
            inc_cost, inc_latency = _expected_incremental_cost_latency(
                workflow=workflow,
                node=node,
                endpoint=ep,
                partial=partial,
                virtual_map=metrics_cache.virtual_map,
                network_index=metrics_cache.network_index,
                qs_data=metrics_cache.qs_data,
                config=config,
            )
            rank = (inc_cost, inc_latency, ep.endpoint_id)
            if best_rank is None or rank < best_rank:
                best_rank = rank
                best_ep = ep
        if best_ep is None:
            raise RuntimeError(f"Greedy: no candidate for {node}")
        partial[node] = best_ep

    metrics = metrics_cache.evaluate(partial, include_per_qs=True)
    return _to_result(
        workflow=workflow,
        assignment=partial,
        quality_level=quality_level,
        method="greedy",
        metrics=metrics,
        runtime_sec=time.perf_counter() - t0,
    )


def solve_murakkab_profile(
    workflow: WorkflowDAG,
    endpoints: list[Endpoint],
    queries: list[Query],
    scenarios: list[Scenario],
    quality_level: str,
    config: SolverConfig,
) -> OptimizationResult:
    """
    Murakkab-style profile-guided optimizer.

    This approximates Murakkab's profile-guided SLO-aware configuration:
    it starts from the cheapest profiled deployment, then greedily reconfigures
    one logical executor at a time using only calibration samples until the
    empirical latency SLO target is met or no improving reconfiguration remains.
    """
    t0 = time.perf_counter()
    node_candidates = _node_candidates(workflow, endpoints, quality_level, config)
    metrics_cache = _BaselineEvaluationCache(
        workflow, endpoints, queries, scenarios, quality_level, config
    )

    def cheapest_assignment() -> dict[str, Endpoint]:
        return {
            node: min(cands, key=lambda ep: (ep.fixed_cost + ep.cost_per_mb * 100.0, ep.endpoint_id))
            for node, cands in node_candidates.items()
        }

    assignment = cheapest_assignment()
    metrics = metrics_cache.evaluate(assignment)
    target_violation = config.eta
    seen: set[tuple[tuple[str, str], ...]] = set()

    for _ in range(64):
        key = tuple(sorted((node, ep.endpoint_id) for node, ep in assignment.items()))
        if key in seen:
            break
        seen.add(key)
        if metrics["violation_rate"] <= target_violation:
            break

        best_trial: dict[str, Endpoint] | None = None
        best_metrics: dict | None = None
        best_rank: tuple[float, float, float] | None = None
        current_violation = float(metrics["violation_rate"])
        current_cost = float(metrics["expected_cost"])

        for node, cands in node_candidates.items():
            current_ep = assignment[node]
            for ep in cands:
                if ep.endpoint_id == current_ep.endpoint_id:
                    continue
                trial = dict(assignment)
                trial[node] = ep
                trial_metrics = metrics_cache.evaluate(trial)
                violation = float(trial_metrics["violation_rate"])
                cost = float(trial_metrics["expected_cost"])
                improvement = current_violation - violation
                if improvement <= 1e-12:
                    continue
                # Prefer feasible profiles; otherwise maximize violation reduction with low cost growth.
                feasible_penalty = 0.0 if violation <= target_violation else 1.0
                cost_growth = max(cost - current_cost, 0.0)
                rank = (feasible_penalty, violation, cost_growth / max(improvement, 1e-9))
                if best_rank is None or rank < best_rank:
                    best_rank = rank
                    best_trial = trial
                    best_metrics = trial_metrics

        if best_trial is None or best_metrics is None:
            break
        assignment = best_trial
        metrics = best_metrics

    metrics = metrics_cache.evaluate(assignment, include_per_qs=True)
    return _to_result(
        workflow=workflow,
        assignment=assignment,
        quality_level=quality_level,
        method="murakkab_profile",
        metrics=metrics,
        runtime_sec=time.perf_counter() - t0,
    )


BASELINE_SOLVERS: dict[str, SolveFn] = {
    "single_cloud": solve_single_cloud,
    "greedy": solve_greedy,
    "murakkab_profile": solve_murakkab_profile,
}


def solve_baseline(
    method: str,
    workflow: WorkflowDAG,
    endpoints: list[Endpoint],
    queries: list[Query],
    scenarios: list[Scenario],
    quality_level: str,
    config: SolverConfig,
) -> OptimizationResult:
    try:
        solver = BASELINE_SOLVERS[method]
    except KeyError as e:
        raise ValueError(
            f"Unknown baseline {method!r}; choose from {sorted(BASELINE_SOLVERS)}"
        ) from e
    return solver(workflow, endpoints, queries, scenarios, quality_level, config)
