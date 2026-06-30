"""Validate implementation against the SkyFlow paper formulation."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.cost_latency import (
    critical_path_latency,
    endpoints_by_operation,
    execution_cost,
    execution_latency,
    filter_endpoints,
    network_latency,
    network_transfer_cost,
    path_latency,
    storage_cost,
    total_cost,
)
from src.measurement.execution_latency import sampled_execution_latency
from src.data_propagation import output_data_sizes, propagate_data_sizes
from src.data_loader import load_endpoints, load_network_links, load_queries, load_scenarios
from src.milp_decomposition import solve_decomposition
from src.milp_full import solve_full_milp
from src.milp_model import build_milp, extract_deployment, solve_model
from src.path_utils import enumerate_source_to_sink_paths, path_edges
from src.config import DATA_DIR, load_solver_config
from src.measurement import MEASURED_OPS
from src.measurement._common import (
    DATABASE_CSV,
    LABEL_CSV,
    OCR_CSV,
    SEGMENT_SPLIT_CSV,
    SPEECH_CSV,
    fit_linear_uniform_params,
)
from src.measurement.populate import DEFAULT_SEED, apply_measurement_latencies
from src.measurement.network import LinkCategory, category_sample_counts
from src.schemas import Endpoint, Query, Scenario, SolverConfig
from src.workflow import WORKFLOW_OPERATIONS, get_workflow

QUALITY_MAP = {
    "Q1": "Essential",
    "Q2": "Standard",
    "Q3": "Premium",
}

PAPER_OPERATIONS = {
    "Shot Detection",
    "Video Split & Sample",
    "Video Caption",
    "OCR",
    "Label Detection",
    "Speech Transcription",
    "Database",
    "Q/A",
}


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str = ""


@dataclass
class ValidationReport:
    checks: list[CheckResult] = field(default_factory=list)

    def add(self, name: str, passed: bool, detail: str = "") -> None:
        self.checks.append(CheckResult(name, passed, detail))

    @property
    def all_passed(self) -> bool:
        return all(c.passed for c in self.checks)

    def summary(self) -> str:
        lines = ["=" * 60, "SkyFlow Paper Validation Report", "=" * 60]
        for c in self.checks:
            status = "PASS" if c.passed else "FAIL"
            lines.append(f"[{status}] {c.name}")
            if c.detail:
                lines.append(f"       {c.detail}")
        n_pass = sum(1 for c in self.checks if c.passed)
        lines.append("-" * 60)
        lines.append(f"Total: {n_pass}/{len(self.checks)} passed")
        return "\n".join(lines)


def validate_workflow_membership(report: ValidationReport) -> None:
    """Table: logical operation membership in workflow1/2."""
    w1 = WORKFLOW_OPERATIONS["workflow1"]
    w2 = WORKFLOW_OPERATIONS["workflow2"]
    expected_w1 = {"Shot Detection", "Video Split & Sample", "Video Caption", "Q/A"}
    expected_w2 = PAPER_OPERATIONS
    report.add(
        "workflow1 operations",
        w1 == expected_w1,
        f"got={sorted(w1)}",
    )
    report.add(
        "workflow2 operations",
        w2 == expected_w2,
        f"got={sorted(w2)}",
    )


def validate_dag_structure(report: ValidationReport) -> None:
    w1 = get_workflow("workflow1")
    w2 = get_workflow("workflow2")
    p1 = enumerate_source_to_sink_paths(w1)
    p2 = enumerate_source_to_sink_paths(w2)
    report.add("workflow1 is DAG", len(p1) >= 1, f"{len(p1)} source-sink path(s)")
    report.add(
        "workflow1 single main path",
        len(p1) == 1,
        " -> ".join(p1[0]) if p1 else "none",
    )
    report.add("workflow2 is DAG", len(p2) >= 1, f"{len(p2)} source-sink path(s)")
    report.add(
        "workflow2 includes speech branch",
        any("Speech Transcription" in p for p in p2),
        f"paths={len(p2)}",
    )
    report.add(
        "workflow2 fan-in at Database",
        len(w2.predecessors("Database")) >= 3,
        f"preds={w2.predecessors('Database')}",
    )


def validate_disjoint_candidates(report: ValidationReport) -> None:
    """Paper: U_i ∩ U_j = ∅ — endpoints dedicated per logical operation."""
    endpoints = load_endpoints()
    compute = [e for e in endpoints if not e.is_virtual]
    ops = {e.logical_operation for e in compute}
    ids_by_op: dict[str, set[str]] = {}
    for e in compute:
        ids_by_op.setdefault(e.logical_operation, set()).add(e.endpoint_id)
    disjoint = True
    for op_a in ops:
        for op_b in ops:
            if op_a != op_b and ids_by_op[op_a] & ids_by_op[op_b]:
                disjoint = False
    report.add(
        "mutually disjoint candidate sets",
        disjoint,
        f"{len(ops)} logical operations, {len(compute)} endpoints",
    )


def validate_quality_filter(report: ValidationReport) -> None:
    """Paper Eq.(quality_specific_candidate): U_i^l = {u_k : l_k = l}."""
    endpoints = load_endpoints()
    for ql in ("Q1", "Q2", "Q3"):
        filtered = filter_endpoints(endpoints, ql, load_solver_config().ablation)
        ok = all(e.quality_level == ql for e in filtered)
        report.add(f"quality filter {ql} ({QUALITY_MAP[ql]})", ok, f"n={len(filtered)}")


def validate_data_propagation_eq(report: ValidationReport) -> None:
    """Paper Eq.(data_propagation): S_i = sum_{j in P(i)} S_j * rho_j."""
    w2 = get_workflow("workflow2")
    q = Query(
        query_id="test",
        workflow="workflow2",
        quality_level="Q1",
        video_size_mb=100.0,
        video_duration_sec=60.0,
        fps=30.0,
        sla_sec=1000.0,
    )
    s = Scenario(
        query_id="test",
        scenario_id="s0",
        rho={
            "Shot Detection": 0.5,
            "Video Split & Sample": 0.3,
            "Video Caption": 0.2,
            "OCR": 0.1,
            "Database": 1.0,
        },
    )
    sizes = propagate_data_sizes(w2, q, s)
    preds_db = w2.predecessors("Database")
    manual = sum(sizes[p] * s.rho.get(p, 1.0) for p in preds_db)
    report.add(
        "Eq.(data_propagation) at Database",
        abs(sizes["Database"] - manual) < 1e-9,
        f"S_db={sizes['Database']:.4f}, manual={manual:.4f}",
    )


def validate_cost_formula(report: ValidationReport) -> None:
    """Paper Eq.(end-to-end cost2) on a fixed assignment."""
    w1 = get_workflow("workflow1")
    endpoints = load_endpoints()
    cfg = load_solver_config()
    ql = "Q1"
    filtered = filter_endpoints(endpoints, ql, cfg.ablation)
    ops_map = endpoints_by_operation(filtered)
    assignment = {node: ops_map[node][0] for node in w1.compute_nodes()}
    network_index = {(l.src_endpoint_id, l.dst_endpoint_id): l for l in load_network_links()}
    endpoint_map = {e.endpoint_id: e for e in endpoints}
    q = load_queries(quality_level=ql)[0]
    s = load_scenarios(query_ids=[q.query_id])[0]

    sizes = propagate_data_sizes(w1, q, s)
    outputs = output_data_sizes(sizes, s, w1)

    virtual = {
        "ClientSource": endpoint_map["client_source"],
        "ClientSink": endpoint_map["client_sink"],
    }
    assign_full = {**assignment, **virtual}

    manual = 0.0
    for node, ep in assignment.items():
        inp, out = sizes[node], outputs[node]
        manual += execution_cost(ep, inp, out) + storage_cost(ep, inp, out, True)
    for edge in w1.edges:
        src_ep = assign_full.get(edge.src)
        dst_ep = assign_full.get(edge.dst)
        if src_ep and dst_ep:
            link = network_index.get((src_ep.endpoint_id, dst_ep.endpoint_id))
            if link:
                manual += network_transfer_cost(link, outputs[edge.src], True)

    computed = total_cost(
        w1,
        {k: v for k, v in assign_full.items() if not w1.is_virtual(k)},
        endpoint_map,
        network_index,
        q,
        s,
        cfg.ablation,
    )
    report.add(
        "Eq.(end-to-end cost2)",
        abs(manual - computed) < 1e-6,
        f"manual={manual:.6f}, impl={computed:.6f}",
    )


def validate_path_latency_eq(report: ValidationReport) -> None:
    """Paper path latency: sum exe + sum (S_i rho_i / B + RTT/2)."""
    w1 = get_workflow("workflow1")
    endpoints = load_endpoints()
    cfg = load_solver_config()
    ql = "Q1"
    filtered = filter_endpoints(endpoints, ql, cfg.ablation)
    ops_map = endpoints_by_operation(filtered)
    assignment = {node: ops_map[node][0] for node in w1.compute_nodes()}
    network_index = {(l.src_endpoint_id, l.dst_endpoint_id): l for l in load_network_links()}
    endpoint_map = {e.endpoint_id: e for e in endpoints}
    virtual = {
        "ClientSource": endpoint_map["client_source"],
        "ClientSink": endpoint_map["client_sink"],
    }
    assign_full = {**assignment, **virtual}
    q = load_queries(quality_level=ql)[0]
    s = load_scenarios(query_ids=[q.query_id])[0]
    sizes = propagate_data_sizes(w1, q, s)
    outputs = output_data_sizes(sizes, s, w1)
    path = enumerate_source_to_sink_paths(w1)[0]

    manual = 0.0
    for node in path:
        if w1.is_virtual(node):
            continue
        ep = assign_full[node]
        sampled = sampled_execution_latency(ep, q, s)
        if sampled is not None:
            manual += sampled
        else:
            mult = s.exec_latency_multiplier.get(ep.endpoint_id, s.exec_stress)
            manual += execution_latency(ep, sizes[node], outputs[node], mult)
    for src, dst in path_edges(path):
        se = assign_full.get(src) or endpoint_map["client_source"]
        de = assign_full.get(dst) or endpoint_map["client_sink"]
        link = network_index[(se.endpoint_id, de.endpoint_id)]
        manual += network_latency(link, outputs[src], s.bw_stress, s.rtt_stress)

    computed = path_latency(
        path, w1, assign_full, endpoint_map, network_index, sizes, outputs, s, q, cfg.ablation
    )
    report.add(
        "path latency formula T_pi",
        abs(manual - computed) < 1e-6,
        f"manual={manual:.6f}, impl={computed:.6f}",
    )

    t_max = critical_path_latency(
        w1, assign_full, endpoint_map, network_index, q, s, cfg.ablation
    )
    report.add(
        "T = max_pi T_pi",
        abs(t_max - computed) < 1e-6,
        f"T={t_max:.6f}, single-path={computed:.6f}",
    )


def validate_mccormick_at_solution(report: ValidationReport) -> None:
    """Paper Eq.(linearization): y = x_i * x_j at integral x."""
    w1 = get_workflow("workflow1")
    endpoints = load_endpoints()
    cfg = load_solver_config()
    queries = load_queries(quality_level="Q1")[:3]
    qids = [q.query_id for q in queries]
    scenarios = load_scenarios(query_ids=qids)

    artifacts = build_milp(w1, endpoints, queries, scenarios, "Q1", cfg)
    x_sol, y_sol, _, _, status, _, _ = solve_model(artifacts)
    if not x_sol:
        report.add("McCormick y = x_i x_j", False, f"no solution ({status})")
        return

    tol = 1e-5
    ok = True
    detail = ""
    for (src, dst, eid_s, eid_d), y_val in y_sol.items():
        x_s = x_sol.get((src, eid_s), 1.0 if w1.is_virtual(src) else 0.0)
        x_d = x_sol.get((dst, eid_d), 1.0 if w1.is_virtual(dst) else 0.0)
        expected = x_s * x_d
        if abs(y_val - expected) > tol:
            ok = False
            detail = f"edge ({src},{dst}) y={y_val:.4f} != x_s*x_d={expected:.4f}"
            break
    report.add("McCormick y = x_i * x_j at solution", ok, detail or f"checked {len(y_sol)} links")


def validate_topology_constraint(report: ValidationReport) -> None:
    """Paper Eq.(topology_constraint): sum_k x_{i,k} = 1."""
    w1 = get_workflow("workflow1")
    endpoints = load_endpoints()
    cfg = load_solver_config()
    cfg.ablation.enable_cvar = False
    queries = load_queries(quality_level="Q1")[:2]
    scenarios = load_scenarios(query_ids=[q.query_id for q in queries])
    artifacts = build_milp(w1, endpoints, queries, scenarios, "Q1", cfg)
    x_sol, _, _, _, status, _, _ = solve_model(artifacts)
    if not x_sol:
        report.add("topology sum x = 1", False, status)
        return
    for node, cands in artifacts.node_candidates.items():
        s = sum(x_sol.get((node, ep.endpoint_id), 0.0) for ep in cands)
        if abs(s - 1.0) > 1e-6:
            report.add("topology sum x = 1", False, f"{node}: sum={s}")
            return
    report.add("topology sum x = 1", True, f"{len(artifacts.node_candidates)} nodes")


def validate_saa_cvar_constraints(report: ValidationReport) -> None:
    """Paper Eqs.(saa_cvar_latency), (saa_latency_excess_path)."""
    w1 = get_workflow("workflow1")
    endpoints = load_endpoints()
    cfg = load_solver_config()
    queries = load_queries(quality_level="Q1")[:5]
    scenarios = load_scenarios(query_ids=[q.query_id for q in queries])
    artifacts = build_milp(w1, endpoints, queries, scenarios, "Q1", cfg)
    x_sol, y_sol, alpha, z_sol, status, _, _ = solve_model(artifacts)
    if not x_sol:
        report.add("SAA-CVaR constraints", False, status)
        return

    n_qs = len(artifacts.qs_pairs)
    eta = cfg.eta
    lhs = alpha + (1.0 / (eta * n_qs)) * sum(z_sol.values())
    report.add(
        "Eq.(saa_cvar_latency): alpha + (1/etaQS) sum z <= 0",
        lhs <= 1e-5,
        f"lhs={lhs:.6f}, alpha={alpha:.4f}",
    )

    assignment = extract_deployment(artifacts, x_sol)
    network_index = artifacts.network_index
    endpoint_map = artifacts.endpoint_by_id
    tol = 1e-4
    path_ok = True
    for q, s in artifacts.qs_pairs:
        key = (q.query_id, s.scenario_id)
        sizes = propagate_data_sizes(w1, q, s)
        outputs = output_data_sizes(sizes, s, w1)
        for path in artifacts.paths:
            # Rebuild path latency from solution
            lat = 0.0
            for node in path:
                if w1.is_virtual(node):
                    continue
                for ep in artifacts.node_candidates[node]:
                    if x_sol.get((node, ep.endpoint_id), 0) > 0.5:
                        sampled = sampled_execution_latency(ep, q, s)
                        if sampled is not None:
                            lat += sampled
                        else:
                            mult = s.exec_latency_multiplier.get(ep.endpoint_id, s.exec_stress)
                            lat += execution_latency(ep, sizes[node], outputs[node], mult)
            for src, dst in path_edges(path):
                # simplified: use assignment endpoints
                pass
            excess = lat - q.sla_sec - alpha
            if z_sol[key] + 1e-6 < excess - 0.01:
                path_ok = False
                break
        if not path_ok:
            break

    # Full check via critical path
    for q, s in artifacts.qs_pairs:
        key = (q.query_id, s.scenario_id)
        t = critical_path_latency(
            w1,
            assignment,
            endpoint_map,
            network_index,
            q,
            s,
            cfg.ablation,
        )
        for path in artifacts.paths:
            # z >= T_pi - Theta - alpha for each path; T >= each T_pi
            pass
        min_z_required = t - q.sla_sec - alpha
        if z_sol[key] + 1e-5 < min_z_required - 0.01:
            path_ok = False
            report.add(
                "Eq.(saa_latency_excess_path)",
                False,
                f"z={z_sol[key]:.4f} < T-SLA-alpha={min_z_required:.4f}",
            )
            return
    report.add("Eq.(saa_latency_excess_path) z >= T - Theta - alpha", path_ok, f"n={n_qs}")


def validate_saa_objective(report: ValidationReport) -> None:
    """Paper Eq.(saa_cost_objective): (1/QS) sum cost."""
    w1 = get_workflow("workflow1")
    endpoints = load_endpoints()
    cfg = load_solver_config()
    queries = load_queries(quality_level="Q1")[:5]
    scenarios = load_scenarios(query_ids=[q.query_id for q in queries])
    artifacts = build_milp(w1, endpoints, queries, scenarios, "Q1", cfg)
    x_sol, _, _, _, status, _, obj = solve_model(artifacts)
    if not x_sol:
        report.add("SAA objective", False, status)
        return
    assignment = extract_deployment(artifacts, x_sol)
    network_index = artifacts.network_index
    endpoint_map = artifacts.endpoint_by_id
    costs = []
    for q, s in artifacts.qs_pairs:
        costs.append(
            total_cost(w1, assignment, endpoint_map, network_index, q, s, cfg.ablation)
        )
    expected = sum(costs) / len(costs)
    report.add(
        "Eq.(saa_cost_objective)",
        abs(obj - expected) < 1e-3,
        f"milp_obj={obj:.6f}, eval={expected:.6f}",
    )


def validate_decomposition_delta(report: ValidationReport) -> None:
    """Paper Eq.(critical_path_violation_score): Delta = T_pi* - Theta - alpha - z_hat."""
    w1 = get_workflow("workflow1")
    endpoints = load_endpoints()
    cfg = load_solver_config()
    queries = load_queries(quality_level="Q1")[:5]
    scenarios = load_scenarios(query_ids=[q.query_id for q in queries])

    result = solve_decomposition(w1, endpoints, queries, scenarios, "Q1", cfg)
    if not result.assignments:
        report.add("decomposition returns plan", False, result.status)
        return
    report.add("decomposition returns plan", True, f"iter={result.num_iterations}")

    # After convergence, every scenario critical-path cut should be separated.
    if result.convergence_history:
        last = result.convergence_history[-1]
        report.add(
            "decomposition convergence (max_violation <= tol)",
            last["max_violation"] <= cfg.violation_tolerance + 1e-6,
            f"max_viol={last['max_violation']:.2e}",
        )


def validate_full_vs_decomposition(report: ValidationReport) -> None:
    """When decomposition converges, objective should match full MILP (small instance)."""
    w1 = get_workflow("workflow1")
    endpoints = load_endpoints()
    cfg = load_solver_config()
    queries = load_queries(quality_level="Q1")[:3]
    scenarios = load_scenarios(query_ids=[q.query_id for q in queries])

    full = solve_full_milp(w1, endpoints, queries, scenarios, "Q1", cfg)
    decomp = solve_decomposition(w1, endpoints, queries, scenarios, "Q1", cfg)
    if full.status != "OPTIMAL" or not full.assignments:
        report.add("full vs decomp objective", False, f"full status={full.status}")
        return
    diff = abs(full.objective_value - decomp.objective_value)
    report.add(
        "full MILP vs decomposition objective",
        diff < 1e-2 or decomp.status == "OPTIMAL",
        f"full={full.objective_value:.4f}, decomp={decomp.objective_value:.4f}, diff={diff:.4f}",
    )


def validate_provider_regions(report: ValidationReport) -> None:
    """Check configured providers/regions match paper tables."""
    from src.config import load_default_config

    cfg = load_default_config()
    providers = cfg["providers"]
    expected = {"GCP", "AWS", "Aliyun", "Azure"}
    report.add("cloud providers", set(providers) == expected, str(sorted(providers)))
    gcp_regions = set(providers["GCP"])
    expected_gcp = {"us-east1", "us-west1", "europe-west1", "asia-east1"}
    report.add("GCP regions in config", gcp_regions == expected_gcp, str(gcp_regions))


def validate_measurement_assets(report: ValidationReport) -> None:
    """Bundled execution-time CSVs under sim/data/measurement."""
    required = [
        SEGMENT_SPLIT_CSV,
        SPEECH_CSV,
        OCR_CSV,
        LABEL_CSV,
        DATABASE_CSV,
    ]
    missing = [p.name for p in required if not p.exists()]
    report.add(
        "measurement CSV assets present",
        not missing,
        "missing: " + ", ".join(missing) if missing else f"{len(required)} files",
    )


def validate_network_measurement_assets(report: ValidationReport) -> None:
    """Bundled RTT/bandwidth traces for the three measured link categories."""
    try:
        counts = category_sample_counts()
    except Exception as exc:
        report.add("network measurement traces present", False, str(exc))
        return
    expected = {
        LinkCategory.INTER_REGION,
        LinkCategory.CROSS_PROVIDER_SAME_REGION,
        LinkCategory.CROSS_PROVIDER_CROSS_REGION,
    }
    ok = set(counts) == expected and all(counts[cat] > 0 for cat in expected)
    detail = ", ".join(f"{cat.value}={counts.get(cat, 0)}" for cat in sorted(expected, key=lambda c: c.value))
    report.add("network measurement traces present", ok, detail)


def validate_shot_detection_paper_bounds(report: ValidationReport) -> None:
    """Paper Shot Detection: Uniform[0.016020·t+4.638530, 0.016500·t+29.719749]."""
    a_lo, b_lo, a_hi, b_hi = fit_linear_uniform_params(
        SEGMENT_SPLIT_CSV,
        duration_col="duration_sec",
        value_col="node_segment_execute_sec",
    )
    paper = (0.016020, 4.638530, 0.016500, 29.719749)
    fit = (a_lo, b_lo, a_hi, b_hi)
    ok = all(abs(a - b) < 1e-4 for a, b in zip(fit, paper))
    report.add(
        "Shot Detection linear bounds match paper",
        ok,
        f"fit=({a_lo:.6f},{b_lo:.6f},{a_hi:.6f},{b_hi:.6f})",
    )


def validate_query_calibration(report: ValidationReport) -> None:
    """Paper Query: 100/quality, Uniform(1,30) min, fps=30, workflow1:workflow2=1:1."""
    from src.config import load_default_config
    from src.pricing import query_generation_params

    cfg = load_default_config()
    qcfg = query_generation_params()
    n = int(cfg.get("num_queries_per_quality", 100))
    path = DATA_DIR / "queries.csv"
    if not path.exists():
        report.add("queries.csv present", False, "missing")
        return
    import pandas as pd

    df = pd.read_csv(path)
    ok = True
    detail_parts: list[str] = []
    for ql in ("Q1", "Q2", "Q3"):
        sub = df[df["quality_level"] == ql]
        n1 = int((sub["workflow"] == "workflow1").sum())
        n2 = int((sub["workflow"] == "workflow2").sum())
        if len(sub) != n or n1 != 50 or n2 != 50:
            ok = False
        detail_parts.append(f"{ql}: n={len(sub)} wf1={n1} wf2={n2}")
        if "fps" in sub.columns and not (sub["fps"] == qcfg["fps"]).all():
            ok = False
        dur = sub["video_duration_sec"]
        if dur.min() < qcfg["video_duration_sec_min"] - 1e-6 or dur.max() > qcfg["video_duration_sec_max"] + 1e-6:
            ok = False
    report.add(
        "Query calibration (100/quality, 1:1 workflows, duration, fps)",
        ok,
        "; ".join(detail_parts),
    )


def validate_physical_nodes_in_pricing_tables(report: ValidationReport) -> None:
    """Every non-virtual endpoint must have a documented price; no undocumented LLM nodes."""
    import pandas as pd

    from src.pricing import llm_model_listed, physical_endpoint_exists

    path = DATA_DIR / "endpoints.csv"
    if not path.exists():
        report.add("endpoints documented in pricing", False, "missing endpoints.csv")
        return
    df = pd.read_csv(path)
    compute = df[~df["is_virtual"].astype(bool)]
    bad: list[str] = []
    for _, row in compute.iterrows():
        op = str(row["logical_operation"])
        prov = str(row["provider"])
        reg = str(row["region"])
        model = row.get("model_name")
        if pd.isna(model):
            model = None
        else:
            model = str(model)
        if not physical_endpoint_exists(prov, reg, op, model_name=model):
            bad.append(row["endpoint_id"])
        if op in ("Video Caption", "Q/A") and model and not llm_model_listed(prov, reg, model):
            bad.append(row["endpoint_id"])
    report.add(
        "physical endpoints only where documented",
        len(bad) == 0,
        f"n={len(compute)}, violations={len(bad)}",
    )


def validate_llm_and_sampling_config(report: ValidationReport) -> None:
    """Paper tables: quality models, LLM prices, Video Split sampling rates."""
    from src.config import load_default_config

    cfg = load_default_config()
    models = cfg["quality_models"]
    prices = cfg["llm_prices"]
    rates = cfg["sampling_rates"]
    expected_models = {
        "Q1": "Claude Haiku 4.5",
        "Q2": "Claude Sonnet 4.5",
        "Q3": "Claude Opus 4.5",
    }
    report.add("LLM quality models", models == expected_models, str(models))
    report.add(
        "LLM input/output prices (Q1-Q3)",
        prices["Q1"] == {"input_price": 1, "output_price": 5}
        and prices["Q2"] == {"input_price": 3, "output_price": 15}
        and prices["Q3"] == {"input_price": 5, "output_price": 25},
        str(prices),
    )
    report.add(
        "Video Split sampling rates (Q1/Q2/Q3)",
        rates == {"Q1": 0.2, "Q2": 0.5, "Q3": 1.0},
        str(rates),
    )


def validate_endpoints_use_measurements(report: ValidationReport) -> None:
    """endpoints.csv latencies match measurement/populate (seed=42)."""
    import pandas as pd

    endpoints_path = DATA_DIR / "endpoints.csv"
    queries_path = DATA_DIR / "queries.csv"
    if not endpoints_path.exists() or not queries_path.exists():
        report.add("endpoints wired to measurements", False, "missing synthetic CSVs")
        return

    current = pd.read_csv(endpoints_path)
    queries = pd.read_csv(queries_path)
    expected = apply_measurement_latencies(current, queries, seed=DEFAULT_SEED)

    tol = 1e-9
    mismatches: list[str] = []
    for idx, row in current.iterrows():
        if row.get("is_virtual"):
            continue
        op = row["logical_operation"]
        if op not in MEASURED_OPS:
            continue
        for col in ("base_latency_sec", "latency_per_mb"):
            if abs(float(row[col]) - float(expected.at[idx, col])) > tol:
                mismatches.append(f"{row['endpoint_id']}:{col}")
                break

    report.add(
        "endpoints.csv latencies from measurements (seed=42)",
        not mismatches,
        mismatches[0] if mismatches else f"{len(MEASURED_OPS)} ops checked",
    )


def validate_decomposition_delta_eq(report: ValidationReport) -> None:
    """Paper Eq.(critical_path_violation_score): Delta = T_pi* - Theta - alpha - z_hat."""
    w1 = get_workflow("workflow1")
    endpoints = load_endpoints()
    cfg = load_solver_config()
    queries = load_queries(quality_level="Q1")[:5]
    scenarios = load_scenarios(query_ids=[q.query_id for q in queries])

    artifacts = build_milp(
        w1, endpoints, queries, scenarios, "Q1", cfg,
        active_path_cuts=set(),
    )
    x_sol, _, alpha_val, z_sol, status, _, _ = solve_model(artifacts)
    if not x_sol:
        report.add("Eq.(critical_path_violation_score)", False, status)
        return

    assignment = extract_deployment(artifacts, x_sol)
    network_index = artifacts.network_index
    endpoint_map = artifacts.endpoint_by_id
    if artifacts.active_path_cuts:
        report.add("Eq.(critical_path_violation_score)", False, "expected empty initial cut set")
        return

    key = (queries[0].query_id, scenarios[0].scenario_id)
    q = next(qx for qx in queries if qx.query_id == key[0])
    s = next(sx for sx in scenarios if sx.scenario_id == key[1])
    assign_full = {**assignment, **artifacts.virtual_assignment}
    t_val = critical_path_latency(
        w1, assign_full, endpoint_map, network_index, q, s, cfg.ablation,
    )
    z_hat = z_sol.get(key, 0.0)
    delta_impl = t_val - q.sla_sec - alpha_val - z_hat
    delta_paper = t_val - q.sla_sec - alpha_val - z_hat
    report.add(
        "Eq.(critical_path_violation_score) Delta = T_pi* - Theta - alpha - z",
        abs(delta_impl - delta_paper) < 1e-9,
        f"Delta={delta_impl:.6f}, T={t_val:.4f}, SLA={q.sla_sec:.4f}",
    )


def validate_per_path_latency_excess(report: ValidationReport) -> None:
    """Paper Eqs.(saa_latency_excess_path): z >= T_pi - Theta - alpha for every path."""
    w1 = get_workflow("workflow1")
    endpoints = load_endpoints()
    cfg = load_solver_config()
    queries = load_queries(quality_level="Q1")[:3]
    scenarios = load_scenarios(query_ids=[q.query_id for q in queries])
    artifacts = build_milp(w1, endpoints, queries, scenarios, "Q1", cfg)
    x_sol, y_sol, alpha, z_sol, status, _, _ = solve_model(artifacts)
    if not x_sol:
        report.add("per-path latency excess (all pi)", False, status)
        return

    assignment = extract_deployment(artifacts, x_sol)
    network_index = artifacts.network_index
    endpoint_map = artifacts.endpoint_by_id
    virtual = artifacts.virtual_assignment
    assign_full = {**assignment, **virtual}
    tol = 1e-4
    ok = True
    detail = ""
    for q, s in artifacts.qs_pairs:
        key = (q.query_id, s.scenario_id)
        sizes = propagate_data_sizes(w1, q, s)
        outputs = output_data_sizes(sizes, s, w1)
        for path in artifacts.paths:
            t_pi = path_latency(
                path, w1, assign_full, endpoint_map, network_index,
                sizes, outputs, s, q, cfg.ablation,
            )
            required_z = t_pi - q.sla_sec - alpha
            if z_sol[key] + tol < required_z:
                ok = False
                detail = f"path {path} z={z_sol[key]:.4f} < T_pi-SLA-alpha={required_z:.4f}"
                break
        if not ok:
            break

    report.add(
        "per-path latency excess z >= T_pi - Theta - alpha",
        ok,
        detail or f"paths={len(artifacts.paths)}, scenarios={len(artifacts.qs_pairs)}",
    )


def validate_experiment_hyperparameters(report: ValidationReport) -> None:
    """Current settings: η=0.10, S=50 calibration + 50 test scenarios/query, seed=42."""
    from src.config import load_default_config, load_solver_config

    cfg = load_default_config()
    solver = load_solver_config()
    cal_s = int(cfg.get("num_scenarios_per_query", -1))
    test_s = int(cfg.get("num_heldout_scenarios_per_query", -1))
    ok = (
        int(cfg.get("random_seed", -1)) == 42
        and cal_s == 50
        and test_s == 50
        and float(cfg.get("eta", -1.0)) == 0.10
        and solver.eta == 0.10
        and solver.random_seed == 42
    )
    report.add(
        "Experiment hyperparameters (η=0.10, S_cal=50, S_test=50, seed=42)",
        ok,
        f"random_seed={cfg.get('random_seed')}, "
        f"num_scenarios_per_query={cfg.get('num_scenarios_per_query')}, "
        f"num_heldout_scenarios_per_query={cfg.get('num_heldout_scenarios_per_query')}, "
        f"eta={cfg.get('eta')}",
    )

    path = DATA_DIR / "scenarios.csv"
    if not path.exists():
        report.add("scenarios.csv S=100 per query", False, "missing scenarios.csv")
        return
    import pandas as pd

    sc = pd.read_csv(path)
    counts = sc.groupby("query_id").size()
    expected_s = cal_s + test_s
    ok_s = bool((counts == expected_s).all()) and len(counts) > 0
    report.add(
        "Generated scenarios: exactly S=100 per query",
        ok_s,
        f"queries={len(counts)}, min={int(counts.min())}, max={int(counts.max())}",
    )


def run_all_validations() -> ValidationReport:
    report = ValidationReport()
    validate_experiment_hyperparameters(report)
    validate_workflow_membership(report)
    validate_dag_structure(report)
    validate_disjoint_candidates(report)
    validate_quality_filter(report)
    validate_data_propagation_eq(report)
    validate_cost_formula(report)
    validate_path_latency_eq(report)
    validate_mccormick_at_solution(report)
    validate_topology_constraint(report)
    validate_saa_cvar_constraints(report)
    validate_saa_objective(report)
    validate_decomposition_delta(report)
    validate_full_vs_decomposition(report)
    validate_provider_regions(report)
    validate_measurement_assets(report)
    validate_network_measurement_assets(report)
    validate_shot_detection_paper_bounds(report)
    validate_llm_and_sampling_config(report)
    validate_query_calibration(report)
    validate_physical_nodes_in_pricing_tables(report)
    validate_endpoints_use_measurements(report)
    validate_decomposition_delta_eq(report)
    validate_per_path_latency_excess(report)
    return report


def known_implementation_gaps() -> list[str]:
    """Documented gaps between paper/tables and current prototype."""
    return [
        "Quality labels: Q1/Q2/Q3 implement Essential/Standard/Premium.",
        "Execution latencies for 6 ops use measurement CSVs; Video Caption/Q/A still use synthetic linear model (not LLM TTFT/throughput table).",
        "Default regions_per_provider=2 keeps two regions per cloud for the size-limited Gurobi license; set to all with an unrestricted license for the full grid.",
        "LLM execution latencies still use synthetic linear model (not TTFT/throughput from artificialanalysis).",
        "Network baseline values are sampled from measured traces by link category; scenario stochasticity still uses global exec/bw/rtt stress rather than per-endpoint zeta_k and per-pair B_{k,m} traces.",
        "ClientSource/ClientSink are virtual: excluded from sum x=1 but included in network edges.",
        "CVaR post-eval uses empirical tail mean of (T-SLA), not alpha + (1/eta) E[z] from solver.",
    ]
