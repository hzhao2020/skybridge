"""Baseline deployment policies: SC, Greedy, DPGM, and MTGP."""

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
from src.data_propagation import edge_transfer_size, output_data_sizes, propagate_data_sizes
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
    compute_order = workflow.compute_nodes()
    compute = set(compute_order)
    indegree = {
        n: len([p for p in workflow.predecessors(n) if p in compute])
        for n in compute_order
    }
    ready = [n for n in compute_order if indegree[n] == 0]
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
            cost += self._average_edge_cost(src, dst, src_ep, dst_ep)
        return cost

    def _average_node_cost(self, node: str, endpoint: Endpoint) -> float:
        key = (node, endpoint.endpoint_id)
        cached = self._avg_node_cost_cache.get(key)
        if cached is not None:
            return cached
        costs = [
            execution_cost(
                endpoint,
                input_sizes.get(node, 0.0),
                output_sizes.get(node, 0.0),
                query,
            )
            + storage_cost(
                endpoint,
                input_sizes.get(node, 0.0),
                output_sizes.get(node, 0.0),
                self.config.ablation.enable_storage_cost,
            )
            for query, _, input_sizes, output_sizes in self.qs_data
        ]
        value = float(np.mean(costs)) if costs else 0.0
        self._avg_node_cost_cache[key] = value
        return value

    def _average_edge_cost(
        self,
        src_node: str,
        dst_node: str,
        src_ep: Endpoint,
        dst_ep: Endpoint,
    ) -> float:
        key = (src_node, dst_node, src_ep.endpoint_id, dst_ep.endpoint_id)
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
                    edge_transfer_size(src_node, dst_node, output_sizes, query),
                    self.config.ablation.enable_network_cost,
                )
                for query, _, _, output_sizes in self.qs_data
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
            cost += execution_cost(ep, inp, out, query)
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
            transferred = edge_transfer_size(src, dst, output_sizes, query)
            cost += network_transfer_cost(
                link,
                transferred,
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
            transferred = edge_transfer_size(src, dst, output_sizes, query)
            total += network_latency(
                link,
                transferred,
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
            output_sizes = output_data_sizes(input_sizes, scenario, workflow, query)
            data.append((query, scenario, input_sizes, output_sizes))
    return data


def _expected_node_profile_cost_latency(
    *,
    node: str,
    endpoint: Endpoint,
    qs_data: list[QueryScenarioData],
    config: SolverConfig,
) -> tuple[float, float]:
    total_cost = 0.0
    total_latency = 0.0
    n = 0

    for query, scenario, input_sizes, output_sizes in qs_data:
        inp = input_sizes.get(node, 0.0)
        out = output_sizes.get(node, 0.0)

        inc_cost = execution_cost(endpoint, inp, out, query)
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
    choose the physical candidate endpoint with the lowest expected node cost
    from calibration profiles, including execution and storage costs. Ties break
    by lower expected node latency. Decisions are irrevocable and local, and the
    global deployment plan is not jointly optimized.
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
            node_cost, node_latency = _expected_node_profile_cost_latency(
                node=node,
                endpoint=ep,
                qs_data=metrics_cache.qs_data,
                config=config,
            )
            rank = (node_cost, node_latency, ep.endpoint_id)
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


def solve_dpgm(
    workflow: WorkflowDAG,
    endpoints: list[Endpoint],
    queries: list[Query],
    scenarios: list[Scenario],
    quality_level: str,
    config: SolverConfig,
) -> OptimizationResult:
    """
    Deterministic Profile-Guided MILP (DPGM).

    DPGM follows Murakkab's profile-guided optimization principle and adapts it
    to multi-cloud endpoint selection. It minimizes deterministic profile cost
    subject to profiled critical-path latency constraints. If the hard profile
    constraints are infeasible, it returns the minimum-slack profiled solution.
    Unlike SkyFlow's full MILP, it does not model per-scenario tail risk/CVaR.
    """
    import gurobipy as gp
    from gurobipy import GRB

    t0 = time.perf_counter()
    node_candidates = _node_candidates(workflow, endpoints, quality_level, config)
    metrics_cache = _BaselineEvaluationCache(
        workflow, endpoints, queries, scenarios, quality_level, config
    )
    profiles = _build_dpgm_query_profiles(
        workflow,
        node_candidates,
        metrics_cache,
        config,
    )
    artifacts = _build_dpgm_profile_milp(
        workflow=workflow,
        node_candidates=node_candidates,
        virtual_map=metrics_cache.virtual_map,
        network_index=metrics_cache.network_index,
        query_profiles=profiles,
        config=config,
        gp=gp,
        GRB=GRB,
        allow_latency_slack=False,
    )
    artifacts["model"].optimize()
    used_slack = False
    if artifacts["model"].Status in (GRB.INFEASIBLE, GRB.INF_OR_UNBD):
        logger.warning(
            "DPGM profile MILP infeasible for %s %s; resolving with latency slack",
            workflow.name,
            quality_level,
        )
        artifacts = _build_dpgm_profile_milp(
            workflow=workflow,
            node_candidates=node_candidates,
            virtual_map=metrics_cache.virtual_map,
            network_index=metrics_cache.network_index,
            query_profiles=profiles,
            config=config,
            gp=gp,
            GRB=GRB,
            allow_latency_slack=True,
        )
        artifacts["model"].optimize()
        used_slack = True

    model = artifacts["model"]
    if model.SolCount <= 0:
        raise RuntimeError(f"DPGM profile MILP failed to find a solution (status={model.Status})")

    assignment = _extract_dpgm_assignment(node_candidates, artifacts["x"])
    metrics = metrics_cache.evaluate(assignment, include_per_qs=True)
    result = _to_result(
        workflow=workflow,
        assignment=assignment,
        quality_level=quality_level,
        method="dpgm",
        metrics=metrics,
        runtime_sec=time.perf_counter() - t0,
    )
    result.objective_value = float(model.ObjVal)
    status_name = _dpgm_gurobi_status_name(model.Status, GRB)
    result.status = f"profile_slack_{status_name}" if used_slack else status_name
    result.active_scenario_count = len(profiles)
    return result


def solve_murakkab_profile(
    workflow: WorkflowDAG,
    endpoints: list[Endpoint],
    queries: list[Query],
    scenarios: list[Scenario],
    quality_level: str,
    config: SolverConfig,
) -> OptimizationResult:
    """Backward-compatible alias for the DPGM baseline."""
    return solve_dpgm(workflow, endpoints, queries, scenarios, quality_level, config)


def _build_dpgm_query_profiles(
    workflow: WorkflowDAG,
    node_candidates: dict[str, list[Endpoint]],
    metrics_cache: _BaselineEvaluationCache,
    config: SolverConfig,
) -> list[dict]:
    rows = metrics_cache.qs_data
    if not rows:
        return []

    template_query = rows[0][0]
    profile_query = Query(
        query_id="aggregate_profile",
        workflow=template_query.workflow,
        quality_level=template_query.quality_level,
        video_size_mb=float(np.mean([row[0].video_size_mb for row in rows])),
        video_duration_sec=float(np.mean([row[0].video_duration_sec for row in rows])),
        fps=float(np.mean([row[0].fps for row in rows])),
        sla_sec=float(np.mean([row[0].sla_sec for row in rows])),
    )

    node_cost: dict[tuple[str, str], float] = {}
    node_latency: dict[tuple[str, str], float] = {}
    edge_cost: dict[tuple[str, str, str, str], float] = {}
    edge_latency: dict[tuple[str, str, str, str], float] = {}

    for node, cands in node_candidates.items():
        for ep in cands:
            costs = []
            latencies = []
            for q, scenario, input_sizes, output_sizes in rows:
                inp = input_sizes.get(node, 0.0)
                out = output_sizes.get(node, 0.0)
                costs.append(
                    execution_cost(ep, inp, out, q)
                    + storage_cost(ep, inp, out, config.ablation.enable_storage_cost)
                )
                sampled = sampled_execution_latency(ep, q, scenario)
                if sampled is not None:
                    latencies.append(sampled)
                else:
                    mult = scenario.exec_latency_multiplier.get(ep.endpoint_id, scenario.exec_stress)
                    latencies.append(execution_latency(ep, inp, out, mult))
            node_cost[node, ep.endpoint_id] = float(np.mean(costs)) if costs else 0.0
            node_latency[node, ep.endpoint_id] = float(np.mean(latencies)) if latencies else 0.0

    for edge in workflow.edges:
        src, dst = edge.src, edge.dst
        if not config.ablation.enable_client_upload_download:
            if src == "ClientSource" or dst == "ClientSink":
                continue
        src_eps = _dpgm_candidates_for_node(
            src,
            node_candidates,
            metrics_cache.virtual_map,
            workflow,
        )
        dst_eps = _dpgm_candidates_for_node(
            dst,
            node_candidates,
            metrics_cache.virtual_map,
            workflow,
        )
        for src_ep in src_eps:
            for dst_ep in dst_eps:
                link = metrics_cache.network_index.get((src_ep.endpoint_id, dst_ep.endpoint_id))
                costs = []
                latencies = []
                for q, scenario, _, output_sizes in rows:
                    transferred = edge_transfer_size(src, dst, output_sizes, q)
                    if link is None:
                        costs.append(0.0)
                        latencies.append(0.0)
                        continue
                    costs.append(
                        network_transfer_cost(
                            link,
                            transferred,
                            config.ablation.enable_network_cost,
                        )
                    )
                    pair_key = f"{src_ep.endpoint_id}->{dst_ep.endpoint_id}"
                    bw_mult = scenario.bandwidth_multiplier.get(pair_key, scenario.bw_stress)
                    rtt_mult = scenario.rtt_multiplier.get(pair_key, scenario.rtt_stress)
                    latencies.append(
                        network_latency(
                            link,
                            transferred,
                            bw_mult,
                            rtt_mult,
                            config.ablation.enable_network_latency,
                        )
                    )
                edge_cost[src, dst, src_ep.endpoint_id, dst_ep.endpoint_id] = (
                    float(np.mean(costs)) if costs else 0.0
                )
                edge_latency[src, dst, src_ep.endpoint_id, dst_ep.endpoint_id] = (
                    float(np.mean(latencies)) if latencies else 0.0
                )

    return [
        {
            "query": profile_query,
            "node_cost": node_cost,
            "node_latency": node_latency,
            "edge_cost": edge_cost,
            "edge_latency": edge_latency,
        }
    ]


def _build_dpgm_profile_milp(
    *,
    workflow: WorkflowDAG,
    node_candidates: dict[str, list[Endpoint]],
    virtual_map: dict[str, Endpoint],
    network_index: dict[tuple[str, str], object],
    query_profiles: list[dict],
    config: SolverConfig,
    gp,
    GRB,
    allow_latency_slack: bool,
) -> dict:
    model = gp.Model("dpgm")
    model.Params.OutputFlag = 0
    if config.gurobi_time_limit_sec > 0:
        model.Params.TimeLimit = config.gurobi_time_limit_sec
    model.Params.MIPGap = config.gurobi_mip_gap
    model.Params.MIPGapAbs = 1e-12
    model.Params.Seed = config.random_seed
    model.Params.Threads = 1

    x: dict[tuple[str, str], object] = {}
    for node, cands in node_candidates.items():
        for ep in cands:
            x[node, ep.endpoint_id] = model.addVar(vtype=GRB.BINARY, name=f"x_{node}_{ep.endpoint_id}")

    y: dict[tuple[str, str, str, str], object] = {}
    for edge in workflow.edges:
        src_eps = _dpgm_candidates_for_node(edge.src, node_candidates, virtual_map, workflow)
        dst_eps = _dpgm_candidates_for_node(edge.dst, node_candidates, virtual_map, workflow)
        for src_ep in src_eps:
            for dst_ep in dst_eps:
                key = (edge.src, edge.dst, src_ep.endpoint_id, dst_ep.endpoint_id)
                y[key] = model.addVar(lb=0.0, ub=1.0, name=f"y_{edge.src}_{edge.dst}_{src_ep.endpoint_id}_{dst_ep.endpoint_id}")

    for node, cands in node_candidates.items():
        model.addConstr(
            gp.quicksum(x[node, ep.endpoint_id] for ep in cands) == 1,
            name=f"assign_{node}",
        )

    for key, var in y.items():
        src, dst, src_eid, dst_eid = key
        src_x = _dpgm_endpoint_var(src, src_eid, x, workflow)
        dst_x = _dpgm_endpoint_var(dst, dst_eid, x, workflow)
        if isinstance(src_x, float):
            model.addConstr(var <= dst_x, name=f"mc1_{src}_{dst}_{src_eid}_{dst_eid}")
            model.addConstr(var >= dst_x - 1 + src_x, name=f"mc3_{src}_{dst}_{src_eid}_{dst_eid}")
        elif isinstance(dst_x, float):
            model.addConstr(var <= src_x, name=f"mc1b_{src}_{dst}_{src_eid}_{dst_eid}")
            model.addConstr(var >= src_x - 1 + dst_x, name=f"mc3b_{src}_{dst}_{src_eid}_{dst_eid}")
        else:
            model.addConstr(var <= src_x, name=f"mc1_{src}_{dst}_{src_eid}_{dst_eid}")
            model.addConstr(var <= dst_x, name=f"mc2_{src}_{dst}_{src_eid}_{dst_eid}")
            model.addConstr(var >= src_x + dst_x - 1, name=f"mc3_{src}_{dst}_{src_eid}_{dst_eid}")

    n_profiles = max(len(query_profiles), 1)
    cost_expr = gp.LinExpr()
    for profile in query_profiles:
        for node, cands in node_candidates.items():
            for ep in cands:
                cost_expr += profile["node_cost"][node, ep.endpoint_id] * x[node, ep.endpoint_id]
        for key, var in y.items():
            cost_expr += profile["edge_cost"].get(key, 0.0) * var
    cost_expr = cost_expr / n_profiles

    slack_vars = []
    paths = enumerate_source_to_sink_paths(workflow)
    edges_by_path = [path_edges(path) for path in paths]
    for profile in query_profiles:
        query = profile["query"]
        for idx, path in enumerate(paths):
            latency_expr = gp.LinExpr()
            for node in path:
                if workflow.is_virtual(node):
                    continue
                for ep in node_candidates[node]:
                    latency_expr += profile["node_latency"][node, ep.endpoint_id] * x[node, ep.endpoint_id]
            for edge in edges_by_path[idx]:
                src, dst = edge
                for src_ep in _dpgm_candidates_for_node(src, node_candidates, virtual_map, workflow):
                    for dst_ep in _dpgm_candidates_for_node(dst, node_candidates, virtual_map, workflow):
                        key = (src, dst, src_ep.endpoint_id, dst_ep.endpoint_id)
                        var = y.get(key)
                        if var is not None:
                            latency_expr += profile["edge_latency"].get(key, 0.0) * var
            if allow_latency_slack:
                slack = model.addVar(lb=0.0, name=f"slack_{query.query_id}_{idx}")
                slack_vars.append(slack)
                model.addConstr(latency_expr <= query.sla_sec + slack, name=f"profile_slo_{query.query_id}_{idx}")
            else:
                model.addConstr(latency_expr <= query.sla_sec, name=f"profile_slo_{query.query_id}_{idx}")

    if allow_latency_slack:
        max_cost = _dpgm_profile_cost_scale(query_profiles)
        model.setObjective(
            max(1.0, max_cost) * 1_000_000.0 * gp.quicksum(slack_vars) + cost_expr,
            GRB.MINIMIZE,
        )
    else:
        model.setObjective(cost_expr, GRB.MINIMIZE)
    return {"model": model, "x": x}


def _dpgm_profile_cost_scale(query_profiles: list[dict]) -> float:
    values: list[float] = []
    for profile in query_profiles:
        values.extend(float(v) for v in profile["node_cost"].values())
        values.extend(float(v) for v in profile["edge_cost"].values())
    return max(values) if values else 1.0


def _dpgm_gurobi_status_name(status: int, GRB) -> str:
    names = {
        GRB.OPTIMAL: "optimal",
        GRB.TIME_LIMIT: "time_limit",
        GRB.SUBOPTIMAL: "suboptimal",
        GRB.USER_OBJ_LIMIT: "user_obj_limit",
    }
    return names.get(status, f"status_{status}")


def _extract_dpgm_assignment(
    node_candidates: dict[str, list[Endpoint]],
    x: dict[tuple[str, str], object],
) -> dict[str, Endpoint]:
    assignment: dict[str, Endpoint] = {}
    for node, cands in node_candidates.items():
        selected = max(cands, key=lambda ep: (float(x[node, ep.endpoint_id].X), ep.endpoint_id))
        assignment[node] = selected
    return assignment


def _dpgm_candidates_for_node(
    node: str,
    node_candidates: dict[str, list[Endpoint]],
    virtual_map: dict[str, Endpoint],
    workflow: WorkflowDAG,
) -> list[Endpoint]:
    if workflow.is_virtual(node):
        return [virtual_map[f"virtual_{node}"]]
    return node_candidates[node]


def _dpgm_endpoint_var(
    node: str,
    endpoint_id: str,
    x: dict[tuple[str, str], object],
    workflow: WorkflowDAG,
):
    if workflow.is_virtual(node):
        return 1.0
    return x[node, endpoint_id]


from src.mtgp_baseline import solve_mtgp  # noqa: E402

CANONICAL_BASELINE_METHODS: tuple[str, ...] = (
    "single_cloud",
    "greedy",
    "dpgm",
    "mtgp",
)

BASELINE_SOLVERS: dict[str, SolveFn] = {
    "single_cloud": solve_single_cloud,
    "greedy": solve_greedy,
    "dpgm": solve_dpgm,
    "mtgp": solve_mtgp,
    # Backward-compatible aliases for older scripts/results.
    "murakkab_profile": solve_dpgm,
    "mtgp_3d": solve_mtgp,
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
