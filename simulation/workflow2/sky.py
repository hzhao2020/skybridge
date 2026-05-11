"""
Workflow 2 — CVaR–SAA MILP + Scenario-Adaptive Decomposition + locality warm-start。

对应论文 Eq.(final MILP) 与 Algorithm (Joint Scenario-Adaptive Decomposition)，
逻辑拓扑为 **互斥单路径** ``path_id ∈ {caption, ocr, label, speech}`` 的线性 DAG；
MILP / CVaR 中的端到端时延为该路径上的延迟 **求和**（不做岔路 max）。
完整 DAG 上并行岔路与 speech 支路的 **max** 聚合仅在评估展示（``workflow_display_latency_max_fork``）中使用。

依赖：``pip install gurobipy``（有效 Gurobi license）。

从包含 ``simulation/`` 的目录运行::

    python -m workflow2.run_all_algorithms --path caption --num-queries 20
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Literal, NamedTuple, Sequence

from sim_env.cost import (
    database_instance_cost_usd,
    database_storage_cost_usd,
    egress_cost_usd,
    llm_token_cost_usd,
    split_cost_usd,
    storage_cost_usd,
    video_service_cost_usd,
)
from sim_env.execution_latency import (
    llm_decode_duration_sec,
    sample_database_query_execute_sec,
    sample_label_detection_execute_sec,
    sample_ocr_execute_sec,
    sample_segment_execute_sec,
    sample_speech_transcription_execute_sec,
    sample_split_execute_sec,
)
from sim_env.network import reset_link_counters, sample_link
from sim_env.utility import PhysicalNode, QueryProfile

from workflow1 import utils as wf1_utils

from . import utils as wf2_utils
from .candidates import enumerate_candidates_wf2
from .utils import (
    WF2PathId,
    WF2PhysicalNode,
    path_logical_ops,
    propagate_path_sizes,
    wf2_node_utility,
)

try:
    import gurobipy as gp
    from gurobipy import GRB
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "Sky MILP requires gurobipy: pip install gurobipy (Gurobi license required)"
    ) from e


def _segment_minutes_src(s_src_gb: float) -> float:
    return wf2_utils._segment_video_minutes_from_source(s_src_gb)


def _carrier_seg(node: WF2PhysicalNode) -> PhysicalNode:
    return PhysicalNode("segment", node.provider, node.region)


def coef_local_cost_latency_wf2(
    logical_op: str,
    node: WF2PhysicalNode,
    s_in: list[float],
    rho: tuple[float, ...],
    idx: int,
    seg_min_source: float,
    *,
    vi_exe_sec: float,
    cap_pair: tuple[float, float],
    q_pair: tuple[float, float],
    llm_latency_rng: random.Random | None,
) -> tuple[float, float]:
    """节点本地 exe+storage 费用与 **执行** 时延（不含出边网络）。"""
    rho_i = float(rho[idx])
    gb_local = float(s_in[idx]) * (1.0 + rho_i)
    if logical_op == "database":
        stor = database_storage_cost_usd(node.provider, node.region, gb_local, days=1.0)
    else:
        stor = storage_cost_usd(node.provider, node.region, gb_local, hours=1.0)
    p, r = node.provider, node.region
    cin, cout = cap_pair
    qin, qout = q_pair

    if logical_op == "video_segment":
        exe = video_service_cost_usd(p, r, "segment", seg_min_source)
        return exe + stor, vi_exe_sec
    if logical_op == "shot_detection":
        exe = split_cost_usd(p, r, minutes=1.0)
        return exe + stor, vi_exe_sec
    if logical_op == "video_caption":
        mm = node.model or ""
        exe = llm_token_cost_usd(p, r, mm, cin, cout)
        return exe + stor, llm_decode_duration_sec(mm, cout, rng=llm_latency_rng)
    if logical_op == "ocr":
        exe = video_service_cost_usd(p, r, "ocr", seg_min_source)
        return exe + stor, vi_exe_sec
    if logical_op == "label_detection":
        exe = video_service_cost_usd(p, r, "label_detection", seg_min_source)
        return exe + stor, vi_exe_sec
    if logical_op == "speech_transcription":
        exe = video_service_cost_usd(p, r, "speech_transcription", seg_min_source)
        return exe + stor, vi_exe_sec
    if logical_op == "database":
        exe_inst = database_instance_cost_usd(p, r, days=1.0)
        return exe_inst + stor, wf2_utils.WF2_PLACEHOLDER_DB_LATENCY_SEC
    if logical_op == "qa":
        return wf2_utils.WF2_PLACEHOLDER_QA_FIXED_COST_USD + stor, wf2_utils.WF2_PLACEHOLDER_QA_LATENCY_SEC
    if logical_op == "answer":
        mm = node.model or ""
        exe = llm_token_cost_usd(
            p, r, mm, qin, qout
        )
        return exe + stor, llm_decode_duration_sec(mm, qout, rng=llm_latency_rng)
    raise ValueError(f"unknown logical_op: {logical_op!r}")


def _edge_pair_cost_latency_wf2(
    src: WF2PhysicalNode,
    dst: WF2PhysicalNode,
    xfer_gb: float,
    *,
    rng: random.Random | None = None,
) -> tuple[float, float]:
    ep_s = (src.provider, src.region)
    ep_d = (dst.provider, dst.region)
    cents = egress_cost_usd(ep_s, ep_d, xfer_gb)
    reset_link_counters([(ep_s, ep_d)])
    sm = sample_link(ep_s, ep_d, rng=rng)
    lat = wf1_utils._transfer_seconds(xfer_gb, sm) + 0.5 * (sm.rtt_ms / 1000.0)
    return cents, lat


def _sample_vi_exe_wf2(
    logical_op: str,
    node: WF2PhysicalNode,
    dur_sec: float,
    *,
    seed: int,
    layer_idx: int,
    cand_idx: int,
) -> float:
    dr = wf1_utils.det_rng(seed, "wf2_sky_vi", layer_idx, cand_idx, node.provider, node.region)
    car = _carrier_seg(node)
    scope = str(seed)
    if logical_op == "video_segment":
        return sample_segment_execute_sec(
            dur_sec,
            rng=dr,
            node=car,
            execution_scale_scope=scope,
            execution_scale_seed=seed,
        )
    if logical_op == "shot_detection":
        return sample_split_execute_sec(
            dur_sec,
            rng=dr,
            node=PhysicalNode("split", node.provider, node.region),
            execution_scale_scope=scope,
            execution_scale_seed=seed,
        )
    if logical_op == "ocr":
        return sample_ocr_execute_sec(
            dur_sec,
            rng=dr,
            node=car,
            execution_scale_scope=scope,
            execution_scale_seed=seed,
        )
    if logical_op == "label_detection":
        return sample_label_detection_execute_sec(
            dur_sec,
            rng=dr,
            node=car,
            execution_scale_scope=scope,
            execution_scale_seed=seed,
        )
    if logical_op == "speech_transcription":
        return sample_speech_transcription_execute_sec(
            dur_sec,
            rng=dr,
            node=car,
            execution_scale_scope=scope,
            execution_scale_seed=seed,
        )
    if logical_op == "database":
        dr = wf1_utils.det_rng(
            seed,
            "wf2_sky_db_lat",
            layer_idx,
            cand_idx,
            node.provider,
            node.region,
        )
        return sample_database_query_execute_sec(
            rng=dr,
            node=car,
            execution_scale_scope=scope,
            execution_scale_seed=seed,
        )
    return 0.0


def prepare_coefficients_wf2(
    path_id: WF2PathId,
    cands: tuple[tuple[WF2PhysicalNode, ...], ...],
    queries: list[QueryProfile],
    scenarios: list["JointScenarioWf2"],
) -> tuple[
    list[list[list[float]]],
    list[list[list[list[float]]]],
    list[list[list[float]]],
    list[list[list[list[float]]]],
]:
    """每个 joint scenario ω：``a[i][k]``、``b[e][ka][kb]``（费用与时延）。"""
    ops = path_logical_ops(path_id)
    L = len(ops)

    a_cost_arr: list[list[list[float]]] = []
    b_cost_arr: list[list[list[list[float]]]] = []
    a_lat_arr: list[list[list[float]]] = []
    b_lat_arr: list[list[list[list[float]]]] = []

    for om in scenarios:
        qprof = queries[om.q_idx]
        s_src = qprof.s_src_gb
        rho = om.rho
        seed = om.rng_seed
        sn, xfer = propagate_path_sizes(s_src, rho)
        seg_min = _segment_minutes_src(s_src)
        dur_sec = max(seg_min * 60.0, 1e-6)
        cap_pair, q_pair = wf2_utils.wf2_llm_token_bundle(path_id, s_src, rho)

        ac: list[list[float]] = []
        al: list[list[float]] = []
        for i in range(L):
            op = ops[i]
            row_c = []
            row_l = []
            for ki, node in enumerate(cands[i]):
                ve = _sample_vi_exe_wf2(op, node, dur_sec, seed=seed, layer_idx=i, cand_idx=ki)
                llm_latency_rng = None
                if op in ("video_caption", "answer"):
                    llm_latency_rng = wf1_utils.det_rng(
                        seed,
                        "wf2_sky_llm_jit",
                        i,
                        ki,
                        node.provider,
                        node.region,
                        node.model,
                    )
                c_ij, lat_ij = coef_local_cost_latency_wf2(
                    op,
                    node,
                    sn,
                    rho,
                    i,
                    seg_min,
                    vi_exe_sec=ve,
                    cap_pair=cap_pair,
                    q_pair=q_pair,
                    llm_latency_rng=llm_latency_rng,
                )
                if i == 0 and op == "video_segment":
                    uc, ul = wf1_utils.local_edge_cost_latency(
                        (node.provider, node.region),
                        float(sn[0]),
                        direction="upload",
                        rng=wf1_utils.det_rng(
                            seed, "wf2_sky_loc_up", ki, node.provider, node.region
                        ),
                    )
                    c_ij += uc
                    lat_ij += ul
                if i == L - 1 and op == "answer":
                    dc, dl = wf1_utils.local_edge_cost_latency(
                        (node.provider, node.region),
                        float(xfer[i]),
                        direction="download",
                        rng=wf1_utils.det_rng(
                            seed, "wf2_sky_loc_dn", ki, node.provider, node.region
                        ),
                    )
                    c_ij += dc
                    lat_ij += dl
                row_c.append(c_ij)
                row_l.append(lat_ij)
            ac.append(row_c)
            al.append(row_l)

        bc_e: list[list[list[float]]] = []
        bl_e: list[list[list[float]]] = []
        for ei in range(L - 1):
            xg = xfer[ei]
            mat_c: list[list[float]] = []
            mat_l: list[list[float]] = []
            for ka, na in enumerate(cands[ei]):
                rc = []
                rl = []
                for kb, nb in enumerate(cands[ei + 1]):
                    ep_a = (na.provider, na.region)
                    ep_b = (nb.provider, nb.region)
                    nc, nl = _edge_pair_cost_latency_wf2(
                        na,
                        nb,
                        xg,
                        rng=wf1_utils.det_rng(
                            seed,
                            "wf2_sky_edge",
                            ei,
                            ep_a[0],
                            ep_a[1],
                            ep_b[0],
                            ep_b[1],
                        ),
                    )
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


PreparedCoefficientsWf2 = tuple[
    list[list[list[float]]],
    list[list[list[list[float]]]],
    list[list[list[float]]],
    list[list[list[list[float]]]],
]


@dataclass
class JointScenarioWf2:
    omega_id: int
    q_idx: int
    s_idx: int
    rng_seed: int
    rho: tuple[float, ...]


def build_joint_scenarios_wf2(
    path_id: WF2PathId,
    queries: list[QueryProfile],
    s_per_query: int,
    rng: random.Random | None,
) -> list[JointScenarioWf2]:
    r = rng or random.Random()
    out: list[JointScenarioWf2] = []
    oid = 0
    for q_idx in range(len(queries)):
        for _si in range(s_per_query):
            rho = wf2_utils.sample_wf2_path_rho(path_id, r, queries[q_idx].s_src_gb)
            out.append(JointScenarioWf2(oid, q_idx, _si, r.randint(1, 2**30), rho))
            oid += 1
    return out


class MilpSolutionWf2(NamedTuple):
    x_choice: tuple[int, ...]
    nodes: tuple[WF2PhysicalNode, ...]
    objective_value: float | None
    alpha_c: float | None
    alpha_t: float | None
    eps_c: float | None
    eps_t: float | None
    pulp_status: str


def _gurobi_status_str(model: gp.Model) -> str:
    s = model.Status
    if s == GRB.OPTIMAL:
        return "Optimal"
    if s == GRB.INFEASIBLE:
        return "Infeasible"
    if s == GRB.UNBOUNDED:
        return "Unbounded"
    if s in (GRB.INF_OR_UNBD, GRB.NUMERIC_ERROR):
        return "Undefined"
    if s == GRB.TIME_LIMIT:
        return "TimeLimit"
    if s == GRB.INTERRUPTED:
        return "Interrupted"
    if s == GRB.SUBOPTIMAL:
        return "Not Solved"
    if s in (GRB.LOADED,):
        return "Not Solved"
    return "Undefined"


def _safe_var_x(v: gp.Var) -> float:
    try:
        x = v.X
        return float(x) if x == x else 0.0
    except (gp.GurobiError, AttributeError, ValueError):
        return 0.0


def _safe_set_initial(var: gp.Var, val: float) -> None:
    try:
        var.Start = float(val)
    except (gp.GurobiError, AttributeError):  # pragma: no cover
        pass


def compute_linear_aggregate_wf2(
    picks: tuple[int, ...],
    om_a: list[list[float]],
    om_b: list[list[list[float]]],
) -> float:
    L = len(picks)
    s = sum(om_a[i][picks[i]] for i in range(L))
    for ei in range(L - 1):
        s += om_b[ei][picks[ei]][picks[ei + 1]]
    return s


def _apply_warm_start_wf2(
    x: list[list[gp.Var]],
    y_edge: list[list[list[gp.Var]]],
    dims: tuple[int, ...],
    warm: tuple[int, ...],
) -> None:
    L = len(dims)
    for i in range(L):
        for k in range(dims[i]):
            _safe_set_initial(x[i][k], 1.0 if k == warm[i] else 0.0)
    for ei in range(L - 1):
        ka_star, kb_star = warm[ei], warm[ei + 1]
        for ka in range(dims[ei]):
            for kb in range(dims[ei + 1]):
                v = 1.0 if ka == ka_star and kb == kb_star else 0.0
                _safe_set_initial(y_edge[ei][ka][kb], v)


def _solve_milp_wf2(
    path_id: WF2PathId,
    cands: tuple[tuple[WF2PhysicalNode, ...], ...],
    queries: list[QueryProfile],
    scenarios_all: list[JointScenarioWf2],
    omega_active: list[int],
    *,
    eta_c: float,
    eta_t: float,
    lamb_c: float,
    lamb_t: float,
    weights: Sequence[float],
    time_limit_sec: int | None = None,
    warm_start: tuple[int, ...] | None = None,
    coef_precomputed: PreparedCoefficientsWf2 | None = None,
) -> MilpSolutionWf2:
    ops = path_logical_ops(path_id)
    L = len(ops)
    if len(weights) != L:
        raise ValueError("weights length must match logical layers")
    dims = tuple(len(cands[i]) for i in range(L))

    m = gp.Model("sky_wf2_milp")
    m.setParam("OutputFlag", 0)
    if time_limit_sec is not None and time_limit_sec > 0:
        m.setParam("TimeLimit", float(time_limit_sec))

    x = [[m.addVar(vtype=GRB.BINARY, name=f"x_{i}_{k}") for k in range(dims[i])] for i in range(L)]
    y_edge: list[list[list[gp.Var]]] = []
    for ei in range(L - 1):
        mat = [
            [
                m.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name=f"y_{ei}_{ka}_{kb}")
                for kb in range(dims[ei + 1])
            ]
            for ka in range(dims[ei])
        ]
        y_edge.append(mat)

    for i in range(L):
        m.addConstr(gp.quicksum(x[i][kk] for kk in range(dims[i])) == 1)

    for ei in range(L - 1):
        for ka in range(dims[ei]):
            for kb in range(dims[ei + 1]):
                yv = y_edge[ei][ka][kb]
                m.addConstr(yv <= x[ei][ka])
                m.addConstr(yv <= x[ei + 1][kb])
                m.addConstr(yv >= x[ei][ka] + x[ei + 1][kb] - 1)

    mu_w = weights
    util_expr = gp.quicksum(
        mu_w[i] * wf2_node_utility(cands[i][k]) * x[i][k] for i in range(L) for k in range(dims[i])
    )

    alpha_c_v = m.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="alpha_C")
    alpha_t_v = m.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="alpha_T")
    eps_c = m.addVar(lb=0.0, name="eps_C")
    eps_t = m.addVar(lb=0.0, name="eps_T")

    z_cost = {oid: m.addVar(lb=0.0, name=f"zC_{oid}") for oid in omega_active}
    z_lat = {oid: m.addVar(lb=0.0, name=f"zT_{oid}") for oid in omega_active}

    if coef_precomputed is not None:
        a_cost_arr, b_cost_arr, a_lat_arr, b_lat_arr = coef_precomputed
    else:
        a_cost_arr, b_cost_arr, a_lat_arr, b_lat_arr = prepare_coefficients_wf2(
            path_id, cands, queries, scenarios_all
        )

    for oid in omega_active:
        qprof = queries[scenarios_all[oid].q_idx]
        om_a, om_b = a_cost_arr[oid], b_cost_arr[oid]
        om_la, om_lb = a_lat_arr[oid], b_lat_arr[oid]

        lc = gp.LinExpr()
        for i in range(L):
            for k in range(dims[i]):
                lc.add(om_a[i][k] * x[i][k])
        for ei in range(L - 1):
            for ka in range(dims[ei]):
                for kb in range(dims[ei + 1]):
                    lc.add(om_b[ei][ka][kb] * y_edge[ei][ka][kb])

        lt = gp.LinExpr()
        for i in range(L):
            for k in range(dims[i]):
                lt.add(om_la[i][k] * x[i][k])
        for ei in range(L - 1):
            for ka in range(dims[ei]):
                for kb in range(dims[ei + 1]):
                    lt.add(om_lb[ei][ka][kb] * y_edge[ei][ka][kb])

        m.addConstr(z_cost[oid] >= lc - qprof.theta_cost - alpha_c_v)
        m.addConstr(z_lat[oid] >= lt - qprof.theta_latency_sec - alpha_t_v)

    n_saa_total = max(len(scenarios_all), 1)
    den_c = float(eta_c) * float(n_saa_total)
    den_t = float(eta_t) * float(n_saa_total)
    inv_c = 1.0 / den_c if den_c > 0 else 0.0
    inv_t = 1.0 / den_t if den_t > 0 else 0.0

    m.addConstr(alpha_c_v + inv_c * gp.quicksum(z_cost[oid] for oid in omega_active) <= eps_c)
    m.addConstr(alpha_t_v + inv_t * gp.quicksum(z_lat[oid] for oid in omega_active) <= eps_t)

    m.setObjective(util_expr - lamb_c * eps_c - lamb_t * eps_t, GRB.MAXIMIZE)

    if warm_start is not None:
        _apply_warm_start_wf2(x, y_edge, dims, warm_start)

    m.optimize()
    status_str = _gurobi_status_str(m)

    picks: list[int] = []
    for i in range(L):
        vals = [(k, _safe_var_x(x[i][k])) for k in range(dims[i])]
        ks = sorted(vals, key=lambda t: t[1], reverse=True)[0][0]
        picks.append(int(ks))

    chosen = tuple(cands[i][picks[i]] for i in range(L))

    def gv(var: gp.Var) -> float | None:
        try:
            v = var.X
            return float(v) if v == v else None
        except (gp.GurobiError, AttributeError, ValueError):
            return None

    obj_val: float | None
    try:
        ov = m.ObjVal
        obj_val = float(ov) if ov == ov else None
    except (gp.GurobiError, AttributeError, ValueError):
        obj_val = None

    m.dispose()

    return MilpSolutionWf2(
        x_choice=tuple(picks),
        nodes=chosen,
        objective_value=obj_val,
        alpha_c=gv(alpha_c_v),
        alpha_t=gv(alpha_t_v),
        eps_c=gv(eps_c),
        eps_t=gv(eps_t),
        pulp_status=status_str,
    )


class DecompositionResultWf2(NamedTuple):
    solution: MilpSolutionWf2
    active_indices: list[int]
    iterations: int


def scenario_adaptive_decomposition_wf2(
    *,
    path_id: WF2PathId,
    queries: list[QueryProfile],
    scenarios: list[JointScenarioWf2],
    cands: tuple[tuple[WF2PhysicalNode, ...], ...],
    eta_c: float,
    eta_t: float,
    lamb_c: float,
    lamb_t: float,
    weights: Sequence[float],
    batch_k: int,
    seed_init_ratio: float = 0.15,
    time_limit_sec: int | None = None,
    rng: random.Random | None = None,
    use_warm_start: bool = True,
) -> DecompositionResultWf2:
    r = rng or random.Random()
    n_tot = len(scenarios)
    n0 = max(1, int(math.ceil(seed_init_ratio * n_tot)))

    omega_space = list(range(n_tot))
    r.shuffle(omega_space)
    active = sorted(omega_space[: min(n0, n_tot)])
    active_set = set(active)

    coef_bundle = prepare_coefficients_wf2(path_id, cands, queries, scenarios)

    iters = 0
    sol: MilpSolutionWf2 | None = None

    while True:
        warm: tuple[int, ...] | None = None
        if use_warm_start:
            warm = locality_greedy_warm_start_indices_wf2(
                path_id,
                cands,
                omega_active=sorted(active_set),
                scenarios=scenarios,
                queries=queries,
                weights=weights,
            )
        sol = _solve_milp_wf2(
            path_id,
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
            coef_precomputed=coef_bundle,
        )
        picks = sol.x_choice
        a_cost_arr, b_cost_arr, a_lat_arr, b_lat_arr = coef_bundle

        violators: list[tuple[float, int]] = []
        for oid in omega_space:
            if oid in active_set:
                continue
            om = scenarios[oid]
            qprof = queries[om.q_idx]
            wc = compute_linear_aggregate_wf2(picks, a_cost_arr[oid], b_cost_arr[oid])
            wt = compute_linear_aggregate_wf2(picks, a_lat_arr[oid], b_lat_arr[oid])
            v_c = max(0.0, wc - qprof.theta_cost - (sol.alpha_c or 0.0))
            v_t = max(0.0, wt - qprof.theta_latency_sec - (sol.alpha_t or 0.0))
            delta = v_c + v_t
            if delta > 1e-9:
                violators.append((delta, oid))

        if not violators:
            return DecompositionResultWf2(sol, sorted(active_set), iters)

        violators.sort(reverse=True)
        adding = violators[: max(1, batch_k)]
        changed = False
        for _, oid in adding:
            if oid not in active_set:
                active_set.add(oid)
                changed = True
        iters += 1
        if not changed:
            return DecompositionResultWf2(sol, sorted(active_set), iters)


def prefix_aggregate_ct_wf2(
    path_id: WF2PathId,
    nodes_prefix: tuple[WF2PhysicalNode, ...],
    src_gb: float,
    rho_prefix: tuple[float, ...],
    seed: int,
) -> tuple[float, float]:
    ops = path_logical_ops(path_id)
    Lp = len(nodes_prefix)
    if len(rho_prefix) != Lp:
        raise ValueError("rho_prefix length must match prefix nodes")

    sn, xfer = propagate_path_sizes(src_gb, rho_prefix)
    seg_min = _segment_minutes_src(src_gb)
    dur_sec = max(seg_min * 60.0, 1e-6)
    cap_pair, q_pair = wf2_utils.wf2_llm_token_bundle(path_id, src_gb, rho_prefix)

    ops_full = path_logical_ops(path_id)

    c_tot = 0.0
    t_tot = 0.0
    if nodes_prefix:
        uc, ul = wf1_utils.local_edge_cost_latency(
            (nodes_prefix[0].provider, nodes_prefix[0].region),
            float(sn[0]),
            direction="upload",
            rng=wf1_utils.det_rng(seed, "wf2_pfx_loc_up"),
        )
        c_tot += uc
        t_tot += ul
    for i in range(Lp):
        op = ops[i]
        node_i = nodes_prefix[i]
        ve = _sample_vi_exe_wf2(op, node_i, dur_sec, seed=seed, layer_idx=i, cand_idx=0)
        llm_latency_rng = None
        if op in ("video_caption", "answer"):
            llm_latency_rng = wf1_utils.det_rng(
                seed, "wf2_pfx_llm", i, node_i.provider, node_i.region, node_i.model
            )
        cc, lt = coef_local_cost_latency_wf2(
            op,
            node_i,
            sn,
            rho_prefix,
            i,
            seg_min,
            vi_exe_sec=ve,
            cap_pair=cap_pair,
            q_pair=q_pair,
            llm_latency_rng=llm_latency_rng,
        )
        c_tot += cc
        t_tot += lt

    for ei in range(Lp - 1):
        ec, et = _edge_pair_cost_latency_wf2(
            nodes_prefix[ei],
            nodes_prefix[ei + 1],
            xfer[ei],
            rng=wf1_utils.det_rng(seed, "wf2_pfx_edge", ei),
        )
        c_tot += ec
        t_tot += et

    if Lp == len(ops_full) and ops_full[-1] == "answer":
        dc, dl = wf1_utils.local_edge_cost_latency(
            (nodes_prefix[-1].provider, nodes_prefix[-1].region),
            float(xfer[Lp - 1]),
            direction="download",
            rng=wf1_utils.det_rng(seed, "wf2_pfx_loc_dn"),
        )
        c_tot += dc
        t_tot += dl

    return c_tot, t_tot


def locality_greedy_warm_start_indices_wf2(
    path_id: WF2PathId,
    cands: tuple[tuple[WF2PhysicalNode, ...], ...],
    *,
    omega_active: list[int],
    scenarios: list[JointScenarioWf2],
    queries: list[QueryProfile],
    weights: Sequence[float],
) -> tuple[int, ...]:
    ops = path_logical_ops(path_id)
    L = len(ops)
    oa = omega_active or [0]

    mean_rho_acc = [0.0] * L
    for oid in oa:
        rho = scenarios[oid].rho
        for j in range(L):
            mean_rho_acc[j] += rho[j]
    denom_n = float(len(oa))
    mean_rho = tuple(mean_rho_acc[j] / denom_n for j in range(L))

    active_q_idx = {scenarios[oid].q_idx for oid in oa}
    theta_c_bar = max(
        sum(queries[qi].theta_cost for qi in active_q_idx) / max(len(active_q_idx), 1),
        1e-9,
    )
    theta_t_bar = max(
        sum(queries[qi].theta_latency_sec for qi in active_q_idx) / max(len(active_q_idx), 1),
        1e-9,
    )

    ref = oa[len(oa) // 2]
    src = queries[scenarios[ref].q_idx].s_src_gb
    seed_ref = scenarios[ref].rng_seed

    partial: list[WF2PhysicalNode] = []
    picks: list[int] = []

    for i in range(L):
        best_ki = 0
        best_score = -1e300
        for ki, cand in enumerate(cands[i]):
            part = tuple(partial + [cand])
            rho_p = mean_rho[: len(part)]
            c_hat, t_hat = prefix_aggregate_ct_wf2(path_id, part, src, rho_p, seed_ref)
            denom = c_hat / theta_c_bar + t_hat / theta_t_bar
            mu = wf2_node_utility(cand)
            num = weights[i] * mu
            score = num / denom if denom > 1e-15 else num
            if score > best_score:
                best_score = score
                best_ki = ki
        picks.append(best_ki)
        partial.append(cands[i][best_ki])

    return tuple(picks)


def run_sky_deployment_wf2(
    path_id: WF2PathId,
    *,
    queries: list[QueryProfile],
    s_per_query: int,
    eta_c: float = 0.1,
    eta_t: float = 0.1,
    lamb_c: float = 1.0,
    lamb_t: float = 1.0,
    weights: Sequence[float] | None = None,
    batch_k: int = 8,
    decomposition: bool = True,
    use_warm_start: bool = True,
    rng_seed: int = 0,
) -> DecompositionResultWf2 | MilpSolutionWf2:
    w = weights if weights is not None else wf2_utils.default_weights_for_path(path_id)
    r = random.Random(rng_seed)
    cands = enumerate_candidates_wf2(path_id)
    scen = build_joint_scenarios_wf2(path_id, queries, s_per_query, r)

    if decomposition:
        return scenario_adaptive_decomposition_wf2(
            path_id=path_id,
            queries=queries,
            scenarios=scen,
            cands=cands,
            eta_c=eta_c,
            eta_t=eta_t,
            lamb_c=lamb_c,
            lamb_t=lamb_t,
            weights=w,
            batch_k=batch_k,
            rng=r,
            use_warm_start=use_warm_start,
        )

    omega_all = list(range(len(scen)))
    warm: tuple[int, ...] | None = None
    if use_warm_start:
        warm = locality_greedy_warm_start_indices_wf2(
            path_id,
            cands,
            omega_active=omega_all,
            scenarios=scen,
            queries=queries,
            weights=w,
        )
    coef_bundle = prepare_coefficients_wf2(path_id, cands, queries, scen)
    return _solve_milp_wf2(
        path_id,
        cands,
        queries,
        scen,
        omega_all,
        eta_c=eta_c,
        eta_t=eta_t,
        lamb_c=lamb_c,
        lamb_t=lamb_t,
        weights=w,
        warm_start=warm,
        coef_precomputed=coef_bundle,
    )


def sky_ablation_settings_wf2(
    variant: Literal["full", "no_warm_start", "direct_milp"],
) -> tuple[bool, bool]:
    if variant == "full":
        return (True, True)
    if variant == "no_warm_start":
        return (True, False)
    if variant == "direct_milp":
        return (False, True)
    raise ValueError(f"unknown variant: {variant!r}")
