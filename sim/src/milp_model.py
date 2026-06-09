"""Shared Gurobi MILP model construction for SkyFlow."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import gurobipy as gp
from gurobipy import GRB

from src.cost_latency import (
    endpoints_by_operation,
    execution_cost,
    execution_latency,
    filter_endpoints,
)
from src.data_propagation import edge_transfer_size, output_data_sizes, propagate_data_sizes
from src.measurement.execution_latency import sampled_execution_latency
from src.path_utils import enumerate_source_to_sink_paths, path_edges
from src.schemas import (
    AblationConfig,
    Endpoint,
    NetworkLink,
    Query,
    Scenario,
    SolverConfig,
    WorkflowDAG,
)

logger = logging.getLogger(__name__)


@dataclass
class MilpArtifacts:
    model: gp.Model
    x: dict[tuple[str, str], gp.Var]
    y: dict[tuple[str, str, str, str], gp.Var]
    alpha: gp.Var
    z: dict[tuple[str, str], gp.Var]
    node_candidates: dict[str, list[Endpoint]]
    endpoint_by_id: dict[str, Endpoint]
    network_index: dict[tuple[str, str], NetworkLink]
    paths: list[list[str]]
    qs_pairs: list[tuple[Query, Scenario]]
    active_keys: set[tuple[str, str]]
    workflow: WorkflowDAG
    config: SolverConfig
    virtual_assignment: dict[str, Endpoint] = field(default_factory=dict)
    primary_objective: gp.LinExpr | None = None
    secondary_objective: gp.LinExpr | None = None


def build_milp(
    workflow: WorkflowDAG,
    all_endpoints: list[Endpoint],
    queries: list[Query],
    scenarios: list[Scenario],
    quality_level: str,
    config: SolverConfig,
    active_scenario_keys: set[tuple[str, str]] | None = None,
) -> MilpArtifacts:
    """Build SkyFlow MILP with optional restricted latency constraints."""
    ablation = config.ablation
    endpoints = filter_endpoints(all_endpoints, quality_level, ablation)
    endpoint_by_id = {e.endpoint_id: e for e in all_endpoints}
    virtual_assignment = {
        "ClientSource": endpoint_by_id["client_source"],
        "ClientSink": endpoint_by_id["client_sink"],
    }

    node_candidates: dict[str, list[Endpoint]] = {}
    ops_map = endpoints_by_operation(endpoints)
    for node in workflow.compute_nodes():
        op = node
        cands = ops_map.get(op, [])
        if not cands:
            raise ValueError(f"No candidates for node {node} at quality {quality_level}")
        node_candidates[node] = cands

    scenario_by_q: dict[str, list[Scenario]] = {}
    for s in scenarios:
        scenario_by_q.setdefault(s.query_id, []).append(s)

    qs_pairs: list[tuple[Query, Scenario]] = []
    for q in queries:
        for s in scenario_by_q.get(q.query_id, []):
            qs_pairs.append((q, s))

    if active_scenario_keys is None:
        active_keys = {(q.query_id, s.scenario_id) for q, s in qs_pairs}
    else:
        active_keys = active_scenario_keys
    active_qs_pairs = [
        (q, s) for q, s in qs_pairs if (q.query_id, s.scenario_id) in active_keys
    ]

    network_index = {(l.src_endpoint_id, l.dst_endpoint_id): l for l in _load_network_for_build()}
    paths = enumerate_source_to_sink_paths(workflow)

    model = gp.Model("skyflow")
    model.Params.OutputFlag = 0
    if config.gurobi_time_limit_sec > 0:
        model.Params.TimeLimit = config.gurobi_time_limit_sec
    model.Params.MIPGap = config.gurobi_mip_gap
    model.Params.MIPGapAbs = 1e-12
    model.Params.Seed = config.random_seed
    model.Params.Threads = 1

    x: dict[tuple[str, str], gp.Var] = {}
    for node, cands in node_candidates.items():
        for ep in cands:
            x[node, ep.endpoint_id] = model.addVar(vtype=GRB.BINARY, name=f"x_{node}_{ep.endpoint_id}")

    y: dict[tuple[str, str, str, str], gp.Var] = {}
    for edge in workflow.edges:
        src, dst = edge.src, edge.dst
        src_cands = _candidates_for_node(src, node_candidates, virtual_assignment, workflow)
        dst_cands = _candidates_for_node(dst, node_candidates, virtual_assignment, workflow)
        for ep_s in src_cands:
            for ep_d in dst_cands:
                key = (src, dst, ep_s.endpoint_id, ep_d.endpoint_id)
                y[key] = model.addVar(lb=0, ub=1, name=f"y_{src}_{dst}_{ep_s.endpoint_id}_{ep_d.endpoint_id}")

    for node, cands in node_candidates.items():
        model.addConstr(
            gp.quicksum(x[node, ep.endpoint_id] for ep in cands) == 1,
            name=f"assign_{node}",
        )

    for key, var in y.items():
        src, dst, eid_s, eid_d = key
        if src in workflow.compute_nodes() and dst in workflow.compute_nodes():
            xs = x[src, eid_s]
            xd = x[dst, eid_d]
        elif workflow.is_virtual(src):
            xs = 1.0
            xd = x[dst, eid_d] if dst in node_candidates else 1.0
        elif workflow.is_virtual(dst):
            xs = x[src, eid_s] if src in node_candidates else 1.0
            xd = 1.0
        else:
            xs = x.get((src, eid_s), 1.0)
            xd = x.get((dst, eid_d), 1.0)

        if isinstance(xs, float):
            model.addConstr(var <= xd, name=f"mc1_{src}_{dst}_{eid_s}_{eid_d}")
            model.addConstr(var >= xd - 1 + xs, name=f"mc3_{src}_{dst}_{eid_s}_{eid_d}")
        elif isinstance(xd, float):
            model.addConstr(var <= xs, name=f"mc1b_{src}_{dst}_{eid_s}_{eid_d}")
            model.addConstr(var >= xs - 1 + xd, name=f"mc3b_{src}_{dst}_{eid_s}_{eid_d}")
        else:
            model.addConstr(var <= xs, name=f"mc1_{src}_{dst}_{eid_s}_{eid_d}")
            model.addConstr(var <= xd, name=f"mc2_{src}_{dst}_{eid_s}_{eid_d}")
            model.addConstr(var >= xs + xd - 1, name=f"mc3_{src}_{dst}_{eid_s}_{eid_d}")

    alpha = model.addVar(lb=-GRB.INFINITY, name="alpha")
    z: dict[tuple[str, str], gp.Var] = {}
    for q, s in active_qs_pairs:
        z[q.query_id, s.scenario_id] = model.addVar(lb=0, name=f"z_{q.query_id}_{s.scenario_id}")

    cost_expr = _build_cost_expression(
        model, workflow, x, y, node_candidates, virtual_assignment,
        network_index, qs_pairs, ablation,
    )
    n_qs = max(len(qs_pairs), 1)
    objective = cost_expr / n_qs
    if config.latency_tiebreaker_weight > 0:
        objective += config.latency_tiebreaker_weight * _build_latency_tiebreaker_expression(
            workflow,
            x,
            y,
            node_candidates,
            virtual_assignment,
            network_index,
            active_qs_pairs,
            paths,
            ablation,
        )
    secondary_objective = None
    if config.endpoint_tiebreaker_weight > 0:
        secondary_objective = _build_endpoint_tiebreaker_expression(x, node_candidates)
    model.setObjective(objective, GRB.MINIMIZE)

    if ablation.enable_cvar:
        eta = config.eta
        model.addConstr(
            alpha + (1.0 / (eta * n_qs)) * gp.quicksum(z.values()) <= 0,
            name="cvar_constraint",
        )

        for q, s in active_qs_pairs:
            key = (q.query_id, s.scenario_id)
            input_sizes = propagate_data_sizes(workflow, q, s)
            output_sizes = output_data_sizes(input_sizes, s, workflow)
            for path in paths:
                path_expr = _path_latency_expr(
                    path, workflow, x, y, node_candidates, virtual_assignment,
                    network_index, input_sizes, output_sizes, q, s, ablation,
                )
                model.addConstr(
                    z[key] >= path_expr - q.sla_sec - alpha,
                    name=f"lat_excess_{key}_{hash(tuple(path)) % 10000}",
                )

    return MilpArtifacts(
        model=model,
        x=x,
        y=y,
        alpha=alpha,
        z=z,
        node_candidates=node_candidates,
        endpoint_by_id=endpoint_by_id,
        network_index=network_index,
        paths=paths,
        qs_pairs=qs_pairs,
        active_keys=active_keys,
        workflow=workflow,
        config=config,
        virtual_assignment=virtual_assignment,
        primary_objective=objective,
        secondary_objective=secondary_objective,
    )


def _load_network_for_build() -> list[NetworkLink]:
    from src.data_loader import load_network_links
    return load_network_links()


def _candidates_for_node(
    node: str,
    node_candidates: dict[str, list[Endpoint]],
    virtual_assignment: dict[str, Endpoint],
    workflow: WorkflowDAG,
) -> list[Endpoint]:
    if workflow.is_virtual(node):
        return [virtual_assignment[node]]
    return node_candidates[node]


def _build_cost_expression(
    model: gp.Model,
    workflow: WorkflowDAG,
    x: dict,
    y: dict,
    node_candidates: dict[str, list[Endpoint]],
    virtual_assignment: dict[str, Endpoint],
    network_index: dict,
    qs_pairs: list[tuple[Query, Scenario]],
    ablation: AblationConfig,
) -> gp.LinExpr:
    x_coeffs: dict[tuple[str, str], float] = {key: 0.0 for key in x}
    y_coeffs: dict[tuple[str, str, str, str], float] = {key: 0.0 for key in y}
    for q, s in qs_pairs:
        input_sizes = propagate_data_sizes(workflow, q, s)
        output_sizes = output_data_sizes(input_sizes, s, workflow)

        for node, cands in node_candidates.items():
            inp = input_sizes.get(node, 0.0)
            out = output_sizes.get(node, 0.0)
            for ep in cands:
                exec_c = execution_cost(ep, inp, out, q)
                stor_c = 0.0
                if ablation.enable_storage_cost:
                    stor_c = ep.storage_cost_per_mb * (inp + out)
                x_coeffs[node, ep.endpoint_id] += exec_c + stor_c

        for edge in workflow.edges:
            src, dst = edge.src, edge.dst
            if not ablation.enable_client_upload_download:
                if src == "ClientSource" or dst == "ClientSink":
                    continue
            if not ablation.enable_network_cost:
                continue
            src_cands = _candidates_for_node(src, node_candidates, virtual_assignment, workflow)
            dst_cands = _candidates_for_node(dst, node_candidates, virtual_assignment, workflow)
            out_mb = edge_transfer_size(src, dst, output_sizes, q)
            for ep_s in src_cands:
                for ep_d in dst_cands:
                    link = network_index.get((ep_s.endpoint_id, ep_d.endpoint_id))
                    if link is None:
                        continue
                    net_c = (out_mb / 1024.0) * link.egress_cost_per_gb
                    key = (src, dst, ep_s.endpoint_id, ep_d.endpoint_id)
                    if key in y:
                        y_coeffs[key] += net_c

    total = gp.LinExpr(0)
    for key, coeff in x_coeffs.items():
        if coeff:
            total += coeff * x[key]
    for key, coeff in y_coeffs.items():
        if coeff:
            total += coeff * y[key]

    return total


def _build_latency_tiebreaker_expression(
    workflow: WorkflowDAG,
    x: dict,
    y: dict,
    node_candidates: dict[str, list[Endpoint]],
    virtual_assignment: dict[str, Endpoint],
    network_index: dict,
    qs_pairs: list[tuple[Query, Scenario]],
    paths: list[list[str]],
    ablation: AblationConfig,
) -> gp.LinExpr:
    total = gp.LinExpr(0)
    denom = max(len(qs_pairs) * max(len(paths), 1), 1)
    for q, s in qs_pairs:
        input_sizes = propagate_data_sizes(workflow, q, s)
        output_sizes = output_data_sizes(input_sizes, s, workflow)
        for path in paths:
            total += _path_latency_expr(
                path,
                workflow,
                x,
                y,
                node_candidates,
                virtual_assignment,
                network_index,
                input_sizes,
                output_sizes,
                q,
                s,
                ablation,
            )
    return total / denom


def _build_endpoint_tiebreaker_expression(
    x: dict,
    node_candidates: dict[str, list[Endpoint]],
) -> gp.LinExpr:
    expr = gp.LinExpr(0)
    denom = max(sum(len(cands) for cands in node_candidates.values()), 1)
    for node in sorted(node_candidates):
        for rank, ep in enumerate(sorted(node_candidates[node], key=lambda e: e.endpoint_id), start=1):
            expr += (rank / denom) * x[node, ep.endpoint_id]
    return expr


def _path_latency_expr(
    path: list[str],
    workflow: WorkflowDAG,
    x: dict,
    y: dict,
    node_candidates: dict[str, list[Endpoint]],
    virtual_assignment: dict[str, Endpoint],
    network_index: dict,
    input_sizes: dict[str, float],
    output_sizes: dict[str, float],
    query: Query,
    scenario: Scenario,
    ablation: AblationConfig,
) -> gp.LinExpr:
    expr = gp.LinExpr(0)
    for node in path:
        if workflow.is_virtual(node):
            continue
        inp = input_sizes.get(node, 0.0)
        out = output_sizes.get(node, 0.0)
        for ep in node_candidates[node]:
            sampled = sampled_execution_latency(ep, query, scenario)
            if sampled is not None:
                lat = sampled
            else:
                mult = scenario.exec_latency_multiplier.get(ep.endpoint_id, scenario.exec_stress)
                lat = execution_latency(ep, inp, out, mult)
            expr += lat * x[node, ep.endpoint_id]

    for src, dst in path_edges(path):
        if not ablation.enable_client_upload_download:
            if src == "ClientSource" or dst == "ClientSink":
                continue
        if not ablation.enable_network_latency:
            continue
        src_cands = _candidates_for_node(src, node_candidates, virtual_assignment, workflow)
        dst_cands = _candidates_for_node(dst, node_candidates, virtual_assignment, workflow)
        out_mb = edge_transfer_size(src, dst, output_sizes, query)
        for ep_s in src_cands:
            for ep_d in dst_cands:
                link = network_index.get((ep_s.endpoint_id, ep_d.endpoint_id))
                if link is None:
                    continue
                bw_key = f"{ep_s.endpoint_id}->{ep_d.endpoint_id}"
                bw_mult = scenario.bandwidth_multiplier.get(bw_key, scenario.bw_stress)
                rtt_mult = scenario.rtt_multiplier.get(bw_key, scenario.rtt_stress)
                bw = max(link.bandwidth_mb_per_sec * bw_mult, 1e-6)
                rtt = link.rtt_sec * rtt_mult
                net_lat = out_mb / bw + rtt / 2.0
                key = (src, dst, ep_s.endpoint_id, ep_d.endpoint_id)
                if key in y:
                    expr += net_lat * y[key]

    return expr


def solve_model(artifacts: MilpArtifacts) -> tuple[dict, dict, float, dict, str, float, float]:
    """Solve and extract incumbent solution."""
    t0 = time.perf_counter()
    artifacts.model.optimize()
    status = _gurobi_status(artifacts.model.Status)

    if artifacts.model.SolCount == 0:
        runtime = time.perf_counter() - t0
        logger.error("Gurobi found no feasible solution (status=%s)", status)
        return {}, {}, 0.0, {}, status, runtime, float("inf")

    if artifacts.secondary_objective is not None and artifacts.primary_objective is not None:
        primary_opt = artifacts.primary_objective.getValue()
        artifacts.model.addConstr(
            artifacts.primary_objective <= primary_opt + 1e-9,
            name="primary_objective_lock",
        )
        artifacts.model.setObjective(artifacts.secondary_objective, GRB.MINIMIZE)
        artifacts.model.optimize()
        status = _gurobi_status(artifacts.model.Status)
        if artifacts.model.SolCount == 0:
            runtime = time.perf_counter() - t0
            logger.error("Gurobi found no lexicographic solution (status=%s)", status)
            return {}, {}, 0.0, {}, status, runtime, float("inf")

    runtime = time.perf_counter() - t0

    x_sol: dict[tuple[str, str], float] = {}
    for key, var in artifacts.x.items():
        if var.X > 0.5:
            x_sol[key] = 1.0

    y_sol: dict[tuple[str, str, str, str], float] = {}
    for key, var in artifacts.y.items():
        y_sol[key] = var.X

    alpha_val = artifacts.alpha.X if artifacts.alpha else 0.0
    z_sol = {k: v.X for k, v in artifacts.z.items()}
    obj = (
        artifacts.primary_objective.getValue()
        if artifacts.primary_objective is not None
        else artifacts.model.ObjVal
    )

    return x_sol, y_sol, alpha_val, z_sol, status, runtime, obj


def extract_deployment(
    artifacts: MilpArtifacts,
    x_sol: dict[tuple[str, str], float],
) -> dict[str, Endpoint]:
    assignment: dict[str, Endpoint] = {}
    for (node, eid), val in x_sol.items():
        if val > 0.5:
            assignment[node] = artifacts.endpoint_by_id[eid]
    assignment["ClientSource"] = artifacts.virtual_assignment["ClientSource"]
    assignment["ClientSink"] = artifacts.virtual_assignment["ClientSink"]
    return assignment


def _gurobi_status(status: int) -> str:
    mapping = {
        GRB.OPTIMAL: "OPTIMAL",
        GRB.INFEASIBLE: "INFEASIBLE",
        GRB.INF_OR_UNBD: "INF_OR_UNBD",
        GRB.UNBOUNDED: "UNBOUNDED",
        GRB.TIME_LIMIT: "TIME_LIMIT",
        GRB.SUBOPTIMAL: "SUBOPTIMAL",
    }
    return mapping.get(status, f"STATUS_{status}")
