"""
Chance-constrained deployment optimization (CVaR + SAA MILP, soft slack ε,
Scenario-Adaptive decomposition, locality-aware greedy warm-start).

Pipeline: segment → split → caption → query.

Dependencies: ``pip install pulp``

Import resolution: the directory that **contains** ``sim_env/`` (usually
``simulation/``) must be on ``sys.path``. Run scripts from that folder, or add
it to ``PYTHONPATH``, or install the project as a package.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import NamedTuple

from sim_env import config as cfg
from sim_env.cost import (
    ProviderRegion,
    egress_cost_usd,
    llm_token_cost_usd,
    split_cost_usd,
    storage_cost_usd,
    video_service_cost_usd,
)
from sim_env.execution_latency import llm_decode_duration_sec
from sim_env.network import reset_link_counters, sample_link
from sim_env.utility import PhysicalNode, QueryProfile, physical_node_utility

import utils as wf_utils

try:
    import pulp as pl
except ImportError as e:  # pragma: no cover
    raise ImportError("Sky MILP requires PuLP: pip install pulp") from e


OPS_ORDER = ("segment", "split", "caption", "query")


def _endpoint(n: PhysicalNode) -> ProviderRegion:
    return (n.provider, n.region)


def enumerate_candidates(max_per_op: int | None = None) -> tuple[tuple[PhysicalNode, ...], ...]:
    """Candidate sets U_i from ``WORKFLOW_OPERATIONS``; optionally cap list length per op."""
    layers: list[list[PhysicalNode]] = [[], [], [], []]
    for i, op in enumerate(OPS_ORDER):
        spec = cfg.WORKFLOW_OPERATIONS[op]
        for _prov, pdata in spec.items():
            regions = pdata["supported_regions"]
            models = pdata.get("models")
            if models is None:
                for rg in regions:
                    layers[i].append(PhysicalNode(op, _prov, rg, None))
                    if max_per_op is not None and len(layers[i]) >= max_per_op:
                        break
            else:
                for rg in regions:
                    for m in models:
                        layers[i].append(PhysicalNode(op, _prov, rg, m))
                        if max_per_op is not None and len(layers[i]) >= max_per_op:
                            break
                    if max_per_op is not None and len(layers[i]) >= max_per_op:
                        break
                if max_per_op is not None and len(layers[i]) >= max_per_op:
                    break
        if not layers[i]:
            raise RuntimeError(f"No candidates for {op!r}")
        if max_per_op is not None:
            layers[i] = layers[i][:max_per_op]
    return tuple(tuple(x) for x in layers)


def _segment_minutes(s0_gb: float) -> float:
    return cfg.video_duration_sec_from_megabytes(float(s0_gb) * 1000.0) / 60.0


def coef_local_cost_latency(
    i: int,
    node: PhysicalNode,
    s_in: list[float],
    rho: tuple[float, float, float, float],
    seg_minutes: float,
    rng_seed: int | None,
) -> tuple[float, float]:
    """Node-local exe + storage (USD), and execution-only latency (seconds)."""
    rho_i = float(rho[i])
    stor = storage_cost_usd(
        node.provider, node.region, float(s_in[i]) * (1.0 + rho_i), days=1.0
    )
    op = OPS_ORDER[i]
    rsq = rng_seed

    if op == "segment":
        exe = video_service_cost_usd(node.provider, node.region, "segment", seg_minutes)
        t_exe = wf_utils._segment_exec_seconds(seg_minutes, None)
        return exe + stor, t_exe

    if op == "split":
        exe = split_cost_usd(node.provider, node.region, minutes=1.0)
        t_exe = wf_utils._split_exec_seconds(seg_minutes, None)
        return exe + stor, t_exe

    cap_pair, q_pair = wf_utils._resolve_llm_tokens(
        seg_minutes,
        caption_tokens=None,
        query_tokens=None,
        rng_seed=rsq,
    )
    cin, cout = cap_pair
    qin, qout = q_pair

    if op == "caption":
        mm = node.model or ""
        exe = llm_token_cost_usd(node.provider, node.region, mm, cin, cout)
        t_exe = llm_decode_duration_sec(mm, cout)
        return exe + stor, t_exe

    mm = node.model or ""
    exe = llm_token_cost_usd(node.provider, node.region, mm, qin, qout)
    t_exe = llm_decode_duration_sec(mm, qout)
    return exe + stor, t_exe


def _edge_pair_cost_latency(
    src: PhysicalNode,
    dst: PhysicalNode,
    xfer_gb: float,
) -> tuple[float, float]:
    ep_s, ep_d = _endpoint(src), _endpoint(dst)
    cents = egress_cost_usd(ep_s, ep_d, xfer_gb)
    reset_link_counters([(ep_s, ep_d)])
    sm = sample_link(ep_s, ep_d)
    lat = wf_utils._transfer_seconds(xfer_gb, sm) + 0.5 * (sm.rtt_ms / 1000.0)
    return cents, lat


def prepare_coefficients(
    cands: tuple[tuple[PhysicalNode, ...], tuple[PhysicalNode, ...], tuple[PhysicalNode, ...], tuple[PhysicalNode, ...]],
    queries: list,
    scenarios: list,
) -> tuple[
    list[list[list[float]]],
    list[list[list[list[float]]]],
    list[list[list[float]]],
    list[list[list[list[float]]]],
]:
    """Per scenario ω: matrices a_cos[i][k], b_cos[e][ka][kb]; same for latency."""
    a_cost_arr: list[list[list[float]]] = []
    b_cost_arr: list[list[list[list[float]]]] = []
    a_lat_arr: list[list[list[float]]] = []
    b_lat_arr: list[list[list[list[float]]]] = []

    for om in scenarios:
        qprof = queries[om.q_idx]
        s_src = qprof.s_src_gb
        rho = om.rho
        seed = om.rng_seed
        sn, xfer = wf_utils._propagate_sizes_gb(s_src, rho)
        seg_min = _segment_minutes(sn[0])

        ac: list[list[float]] = []
        al: list[list[float]] = []
        for i in range(4):
            row_c = []
            row_l = []
            for ki, node in enumerate(cands[i]):
                c_ij, lat_ij = coef_local_cost_latency(
                    i, node, sn, rho, seg_min, seed + ki
                )
                row_c.append(c_ij)
                row_l.append(lat_ij)
            ac.append(row_c)
            al.append(row_l)

        bc_e = []
        bl_e = []
        for ei in range(3):
            i0, i1 = ei, ei + 1
            xg = xfer[ei]
            mat_c = []
            mat_l = []
            for ka, na in enumerate(cands[i0]):
                rc = []
                rl = []
                for kb, nb in enumerate(cands[i1]):
                    nc, nl = _edge_pair_cost_latency(na, nb, xg)
                    rc.append(nc)
                    rl.append(nl)
                mat_c.append(rc)
                mat_l.append(rl)
            bc_e.append(mat_c)
            bl_e.append(mat_l)

        a_cost_arr.append(ac)
        b_cost_arr.append(bc_e)
        a_lat_arr.append(al)
        b_lat_arr.append(bl_e)

    return a_cost_arr, b_cost_arr, a_lat_arr, b_lat_arr


@dataclass
class JointScenario:
    omega_id: int
    q_idx: int
    s_idx: int
    rng_seed: int
    rho: tuple[float, float, float, float]


def build_joint_scenarios(
    queries: list[QueryProfile],
    s_per_query: int,
    rng: random.Random | None,
) -> list[JointScenario]:
    r = rng or random.Random()
    out: list[JointScenario] = []
    oid = 0
    for q_idx in range(len(queries)):
        for _si in range(s_per_query):
            rho = tuple(cfg.sample_data_conversion_ratio(op, r) for op in OPS_ORDER)
            out.append(JointScenario(oid, q_idx, _si, r.randint(1, 2**30), rho))
            oid += 1
    return out


class MilpSolution(NamedTuple):
    """Incumbent from one restricted MILP solve."""

    x_choice: tuple[int, int, int, int]
    nodes: tuple[PhysicalNode, PhysicalNode, PhysicalNode, PhysicalNode]
    objective_value: float | None
    alpha_c: float | None
    alpha_t: float | None
    eps_c: float | None
    eps_t: float | None
    pulp_status: str


def _safe_set_initial(var: pl.LpVariable, val: float) -> None:
    """PuLP CBC may ignore MIP starts, but setting values is still the correct API."""
    setter = getattr(var, "setInitialValue", None)
    if callable(setter):
        setter(val)


def _apply_warm_start(
    x: list[list[pl.LpVariable]],
    y_edge: list[list[list[pl.LpVariable]]],
    dims: tuple[int, int, int, int],
    warm: tuple[int, int, int, int],
) -> None:
    for i in range(4):
        for k in range(dims[i]):
            _safe_set_initial(x[i][k], 1.0 if k == warm[i] else 0.0)
    for ei in range(3):
        ka_star, kb_star = warm[ei], warm[ei + 1]
        for ka in range(dims[ei]):
            for kb in range(dims[ei + 1]):
                v = 1.0 if ka == ka_star and kb == kb_star else 0.0
                _safe_set_initial(y_edge[ei][ka][kb], v)


def _solve_milp(
    cands: tuple[tuple[PhysicalNode, ...], ...],
    queries: list[QueryProfile],
    scenarios_all: list[JointScenario],
    omega_active: list[int],
    *,
    eta_c: float,
    eta_t: float,
    lamb_c: float,
    lamb_t: float,
    weights: tuple[float, float, float, float],
    time_limit_sec: int | None = None,
    warm_start: tuple[int, int, int, int] | None = None,
) -> MilpSolution:
    dims = tuple(len(cands[i]) for i in range(4))
    prob = pl.LpProblem("sky_milp", pl.LpMaximize)

    x: list[list[pl.LpVariable]] = [
        [pl.LpVariable(f"x_{i}_{k}", cat=pl.LpBinary) for k in range(dims[i])]
        for i in range(4)
    ]
    y_edge: list[list[list[pl.LpVariable]]] = []
    for ei in range(3):
        mat = [
            [
                pl.LpVariable(f"y_{ei}_{ka}_{kb}", lowBound=0, upBound=1)
                for kb in range(dims[ei + 1])
            ]
            for ka in range(dims[ei])
        ]
        y_edge.append(mat)

    for i in range(4):
        prob += pl.lpSum(x[i][kk] for kk in range(dims[i])) == 1

    for ei in range(3):
        for ka in range(dims[ei]):
            for kb in range(dims[ei + 1]):
                yv = y_edge[ei][ka][kb]
                prob += yv <= x[ei][ka]
                prob += yv <= x[ei + 1][kb]
                prob += yv >= x[ei][ka] + x[ei + 1][kb] - 1

    mu_w = weights
    util_terms: list = []
    for i in range(4):
        for k in range(dims[i]):
            util_terms.append(
                mu_w[i] * physical_node_utility(cands[i][k]) * x[i][k]
            )

    alpha_c_v = pl.LpVariable("alpha_C", lowBound=None, cat=pl.LpContinuous)
    alpha_t_v = pl.LpVariable("alpha_T", lowBound=None, cat=pl.LpContinuous)
    eps_c = pl.LpVariable("eps_C", lowBound=0)
    eps_t = pl.LpVariable("eps_T", lowBound=0)

    z_cost = {oid: pl.LpVariable(f"zC_{oid}", lowBound=0) for oid in omega_active}
    z_lat = {oid: pl.LpVariable(f"zT_{oid}", lowBound=0) for oid in omega_active}

    a_cost_arr, b_cost_arr, a_lat_arr, b_lat_arr = prepare_coefficients(
        cands, queries, scenarios_all
    )

    for oid in omega_active:
        qprof = queries[scenarios_all[oid].q_idx]
        om_a, om_b = a_cost_arr[oid], b_cost_arr[oid]
        om_la, om_lb = a_lat_arr[oid], b_lat_arr[oid]

        lc_terms: list = []
        for i in range(4):
            for k in range(dims[i]):
                lc_terms.append(om_a[i][k] * x[i][k])
        for ei in range(3):
            for ka in range(dims[ei]):
                for kb in range(dims[ei + 1]):
                    lc_terms.append(om_b[ei][ka][kb] * y_edge[ei][ka][kb])
        lt_terms: list = []
        for i in range(4):
            for k in range(dims[i]):
                lt_terms.append(om_la[i][k] * x[i][k])
        for ei in range(3):
            for ka in range(dims[ei]):
                for kb in range(dims[ei + 1]):
                    lt_terms.append(om_lb[ei][ka][kb] * y_edge[ei][ka][kb])

        prob += z_cost[oid] >= pl.lpSum(lc_terms) - qprof.theta_cost - alpha_c_v
        prob += z_lat[oid] >= pl.lpSum(lt_terms) - qprof.theta_latency_sec - alpha_t_v

    # SAA denominator: paper uses η · Q · S; inactive scenarios contribute z=0 in the sum.
    n_saa_total = max(len(scenarios_all), 1)
    den_c = float(eta_c) * float(n_saa_total)
    den_t = float(eta_t) * float(n_saa_total)
    inv_c = 1.0 / den_c if den_c > 0 else 0.0
    inv_t = 1.0 / den_t if den_t > 0 else 0.0

    prob += alpha_c_v + inv_c * pl.lpSum(z_cost[oid] for oid in omega_active) <= eps_c
    prob += alpha_t_v + inv_t * pl.lpSum(z_lat[oid] for oid in omega_active) <= eps_t

    prob += pl.lpSum(util_terms) - lamb_c * eps_c - lamb_t * eps_t

    if warm_start is not None:
        _apply_warm_start(x, y_edge, dims, warm_start)

    solver = pl.PULP_CBC_CMD(msg=0, timeLimit=time_limit_sec if time_limit_sec else None)
    prob.solve(solver)

    status_str = str(pl.LpStatus[prob.status])

    picks: list[int] = []
    for i in range(4):
        vals = [(k, float(pl.value(v) or 0.0)) for k, v in enumerate(x[i])]
        ks = sorted(vals, key=lambda t: t[1], reverse=True)[0][0]
        picks.append(int(ks))

    chosen = tuple(cands[i][picks[i]] for i in range(4))

    def gv(v: pl.LpVariable) -> float | None:
        vv = pl.value(v)
        return float(vv) if vv is not None else None

    obj_val = pl.value(prob.objective)

    return MilpSolution(
        x_choice=tuple(picks),
        nodes=chosen,
        objective_value=float(obj_val) if obj_val is not None else None,
        alpha_c=gv(alpha_c_v),
        alpha_t=gv(alpha_t_v),
        eps_c=gv(eps_c),
        eps_t=gv(eps_t),
        pulp_status=status_str,
    )


def compute_linear_aggregate(
    picks: tuple[int, int, int, int],
    om_a: list[list[float]],
    om_b: list[list[list[float]]],
) -> float:
    """Sum a[i][pick_i] + sum_e b[e][pick_e][pick_{e+1}] for sanity checks."""
    s = om_a[0][picks[0]] + om_a[1][picks[1]] + om_a[2][picks[2]] + om_a[3][picks[3]]
    for ei in range(3):
        s += om_b[ei][picks[ei]][picks[ei + 1]]
    return s


class DecompositionResult(NamedTuple):
    solution: MilpSolution
    active_indices: list[int]
    iterations: int


def scenario_adaptive_decomposition(
    *,
    queries: list[QueryProfile],
    scenarios: list[JointScenario],
    cands: tuple[tuple[PhysicalNode, ...], tuple[PhysicalNode, ...], tuple[PhysicalNode, ...], tuple[PhysicalNode, ...]],
    eta_c: float,
    eta_t: float,
    lamb_c: float,
    lamb_t: float,
    weights: tuple[float, float, float, float],
    batch_k: int,
    seed_init_ratio: float = 0.15,
    time_limit_sec: int | None = None,
    rng: random.Random | None = None,
) -> DecompositionResult:
    """
    Joint Scenario-Adaptive Decomposition (Algorithm in paper).

    Starts from a random subset of joint scenarios ω=(q,s), solves restricted MILPs,
    measures violation Δ on withheld ω, grows active set by top-``batch_k``.
    Terminates when no positive violations remain among withheld scenarios.
    """
    r = rng or random.Random()
    n_tot = len(scenarios)
    n0 = max(1, int(math.ceil(seed_init_ratio * n_tot)))

    omega_space = list(range(n_tot))
    r.shuffle(omega_space)
    active = sorted(omega_space[: min(n0, n_tot)])
    active_set = set(active)

    iters = 0
    sol: MilpSolution | None = None

    while True:
        warm = locality_greedy_warm_start_indices(
            cands,
            omega_active=sorted(active_set),
            scenarios=scenarios,
            queries=queries,
            weights=weights,
        )
        sol = _solve_milp(
            cands,
            queries,
            scenarios,
            sorted(active_set),
            eta_c=eta_c,
            eta_t=eta_t,
            lamb_c=lamb_c,
            lamb_t=lamb_t,
            weights=weights,
            time_limit_sec=time_limit_sec,
            warm_start=warm,
        )
        picks = sol.x_choice
        a_cost_arr, b_cost_arr, a_lat_arr, b_lat_arr = prepare_coefficients(
            cands, queries, scenarios
        )

        violators: list[tuple[float, int]] = []
        for oid in omega_space:
            if oid in active_set:
                continue
            om = scenarios[oid]
            qprof = queries[om.q_idx]
            wc = compute_linear_aggregate(
                picks,
                a_cost_arr[oid],
                b_cost_arr[oid],
            )
            wt = compute_linear_aggregate(
                picks,
                a_lat_arr[oid],
                b_lat_arr[oid],
            )
            v_c = max(0.0, wc - qprof.theta_cost - (sol.alpha_c or 0.0))
            v_t = max(0.0, wt - qprof.theta_latency_sec - (sol.alpha_t or 0.0))
            delta = v_c + v_t
            if delta > 1e-9:
                violators.append((delta, oid))

        if not violators:
            return DecompositionResult(sol, sorted(active_set), iters)

        violators.sort(reverse=True)
        adding = violators[: max(1, batch_k)]
        changed = False
        for _, oid in adding:
            if oid not in active_set:
                active_set.add(oid)
                changed = True
        iters += 1
        if not changed:
            return DecompositionResult(sol, sorted(active_set), iters)


def prefix_aggregate_ct(
    nodes_prefix: tuple[PhysicalNode, ...],
    src_gb: float,
    rho: tuple[float, float, float, float],
    seed: int,
) -> tuple[float, float]:
    """Cumulative cost and execution+network latency through a prefix of the chain."""
    if not nodes_prefix:
        return 0.0, 0.0
    sn, xfer = wf_utils._propagate_sizes_gb(src_gb, rho)
    seg_min = _segment_minutes(sn[0])
    c_tot = 0.0
    t_tot = 0.0
    for i in range(len(nodes_prefix)):
        cc, lt = coef_local_cost_latency(
            i, nodes_prefix[i], sn, rho, seg_min, seed + i
        )
        c_tot += cc
        t_tot += lt
    for ei in range(len(nodes_prefix) - 1):
        ec, et = _edge_pair_cost_latency(
            nodes_prefix[ei], nodes_prefix[ei + 1], xfer[ei]
        )
        c_tot += ec
        t_tot += et
    return c_tot, t_tot


def locality_greedy_warm_start_indices(
    cands: tuple[tuple[PhysicalNode, ...], ...],
    *,
    omega_active: list[int],
    scenarios: list[JointScenario],
    queries: list[QueryProfile],
    weights: tuple[float, float, float, float],
) -> tuple[int, int, int, int]:
    """
    Greedy heuristic: sequential choice maximizing

        (w_i · μ_{i,k}) / ( Ĉ_prefix / Θ̄_C + \hat{T}_prefix / Θ̄_T ),

    with Ĉ, \hat{T} the prefix monetary cost and latency under mean ρ over Ω_act.
    """
    oa = omega_active or [0]

    mean_rho = [0.0, 0.0, 0.0, 0.0]
    for oid in oa:
        rho = scenarios[oid].rho
        for j in range(4):
            mean_rho[j] += rho[j]
    denom_n = float(len(oa))
    for j in range(4):
        mean_rho[j] /= denom_n

    rho_tup = (mean_rho[0], mean_rho[1], mean_rho[2], mean_rho[3])

    theta_c_bar = (
        max(sum(q.theta_cost for q in queries) / max(len(queries), 1), 1e-9)
    )
    theta_t_bar = (
        max(sum(q.theta_latency_sec for q in queries) / max(len(queries), 1), 1e-9)
    )

    ref = oa[len(oa) // 2]
    src = queries[scenarios[ref].q_idx].s_src_gb
    seed_ref = scenarios[ref].rng_seed

    partial: list[PhysicalNode] = []
    picks: list[int] = []

    for i in range(4):
        best_ki = 0
        best_score = -1e300
        for ki, cand in enumerate(cands[i]):
            part = tuple(partial + [cand])
            c_hat, t_hat = prefix_aggregate_ct(part, src, rho_tup, seed_ref)
            denom = c_hat / theta_c_bar + t_hat / theta_t_bar
            mu = physical_node_utility(cand)
            num = weights[i] * mu
            score = num / denom if denom > 1e-15 else num
            if score > best_score:
                best_score = score
                best_ki = ki
        picks.append(best_ki)
        partial.append(cands[i][best_ki])

    return (picks[0], picks[1], picks[2], picks[3])


def run_sky_deployment(
    *,
    queries: list[QueryProfile],
    s_per_query: int,
    max_per_op: int | None = None,
    eta_c: float = 0.1,
    eta_t: float = 0.1,
    lamb_c: float = 1.0,
    lamb_t: float = 1.0,
    weights: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
    batch_k: int = 8,
    decomposition: bool = True,
    rng_seed: int = 0,
) -> DecompositionResult | MilpSolution:
    """
    Convenience: build candidates + joint scenarios, optionally run decomposition loop.
    """
    r = random.Random(rng_seed)
    cands = enumerate_candidates(max_per_op=max_per_op)
    scen = build_joint_scenarios(queries, s_per_query, r)
    if decomposition:
        return scenario_adaptive_decomposition(
            queries=queries,
            scenarios=scen,
            cands=cands,
            eta_c=eta_c,
            eta_t=eta_t,
            lamb_c=lamb_c,
            lamb_t=lamb_t,
            weights=weights,
            batch_k=batch_k,
            rng=r,
        )
    omega_all = list(range(len(scen)))
    warm = locality_greedy_warm_start_indices(
        cands,
        omega_active=omega_all,
        scenarios=scen,
        queries=queries,
        weights=weights,
    )
    return _solve_milp(
        cands,
        queries,
        scen,
        omega_all,
        eta_c=eta_c,
        eta_t=eta_t,
        lamb_c=lamb_c,
        lamb_t=lamb_t,
        weights=weights,
        warm_start=warm,
    )


if __name__ == "__main__":  # pragma: no cover
    import time

    # 100 个校准请求 + 每请求 SAA 场景数；联立场景数 = len(queries) * s_per_query
    NUM_QUERIES = 100
    S_PER_QUERY = 20
    QUERY_SEED = 42
    RUN_RNG_SEED = 0

    # 限制每层候选数以控制 MILP 规模；None 为全量（可能极慢）
    MAX_PER_OP = 5
    BATCH_K = 12
    ETA_C = 0.1
    ETA_T = 0.1

    t0 = time.perf_counter()
    queries = wf_utils.generate_realistic_queries(NUM_QUERIES, seed=QUERY_SEED)
    print(
        f"Loaded {len(queries)} queries (joint scenarios ≤ {len(queries) * S_PER_QUERY})."
    )

    WEIGHTS_KPI = (1.0, 1.0, 1.0, 1.0)
    rep = run_sky_deployment(
        queries=queries,
        s_per_query=S_PER_QUERY,
        max_per_op=MAX_PER_OP,
        eta_c=ETA_C,
        eta_t=ETA_T,
        batch_k=BATCH_K,
        rng_seed=RUN_RNG_SEED,
        weights=WEIGHTS_KPI,
    )
    elapsed = time.perf_counter() - t0

    if isinstance(rep, DecompositionResult):
        s = rep.solution
        print(
            "decomposition iters",
            rep.iterations,
            "active_scenarios",
            len(rep.active_indices),
        )
        algo_tag = "Sky (CVaR–SAA MILP)"
    else:
        s = rep
        algo_tag = "Sky (full MILP)"

    print("solver_status", s.pulp_status, "solver_objective", s.objective_value)
    print("deployment", s.nodes)
    print(f"elapsed_sec {elapsed:.2f}")

    from evaluation import evaluate_deployment_empirical, print_metrics_report

    EVAL_SAMPLES_PER_QUERY = 40
    EVAL_MC_SEED = 12345 + RUN_RNG_SEED
    m = evaluate_deployment_empirical(
        s.nodes,
        queries,
        weights=WEIGHTS_KPI,
        samples_per_query=EVAL_SAMPLES_PER_QUERY,
        eval_seed=EVAL_MC_SEED,
    )
    print_metrics_report(
        algorithm_label=f"{algo_tag} | MILP_weights={WEIGHTS_KPI} | MonteCarlo_draws_per_query={EVAL_SAMPLES_PER_QUERY}",
        metrics=m,
    )
