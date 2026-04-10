"""
部署层机会约束问题的 CVaR–SAA–MILP 实现（对应论文 *Section IV-A: CVaR-based MILP Reformulation*）。

本文件的实现**完全基于**你定义的 simulation 环境：
- ``distribution.py``：场景采样（价格/网络/LLM/token/ratio 等）
- ``nodes.py``：节点/链路的 cost/latency 口径
- ``workflow.py``：单次 end-to-end 计算的口径（作为 ground truth）
- ``config.py`` + ``config.yaml``：拓扑与参数范围

MILP 需要在“固定场景”下把 cost/latency 写成对 ``(x,y)`` 的线性函数。这里的做法是：
在每个 SAA 场景里调用一次 ``distribution.sample()`` 固定随机参数，并据此生成
``ScenarioCoeffs``（线性系数矩阵），然后交给 PuLP 求解。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence
import random
import time

import pulp

# --- simulation environment ---
from distribution import Query, sample_query_with_budget
from nodes import (
    CaptionCloudNode,
    CaptionNonCloudNode,
    QueryCloudNode,
    QueryNonCloudNode,
    SegmentNode,
    SplitNode,
)
from workflow import Workflow

# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PhysicalEndpoint:
    """物理端点 ``u_k``：名称、质量 ``µ_k``（用于效用式 15）。"""

    name: str
    mu: float


@dataclass(slots=True)
class LogicalNode:
    """逻辑节点 ``v_i^L`` 及其可选端点 ``U_i``。"""

    name: str
    weight: float
    """效用权重 ``w_i``（式 15）。"""
    candidates: list[PhysicalEndpoint]


@dataclass(frozen=True, slots=True)
class LogicalEdge:
    """逻辑边 ``(v_i^L, v_j^L)``（``E_L`` 中的有向依赖）。"""

    src: int
    dst: int


@dataclass(slots=True)
class LogicalDAG:
    """逻辑 DAG：节点列表 + 边集（拓扑序不要求调用方给出）。"""

    nodes: list[LogicalNode]
    edges: list[LogicalEdge]

    def num_nodes(self) -> int:
        return len(self.nodes)

    def adj_out(self) -> dict[int, list[int]]:
        g: dict[int, list[int]] = {i: [] for i in range(self.num_nodes())}
        for e in self.edges:
            g[e.src].append(e.dst)
        return g

    def adj_in(self) -> dict[int, list[int]]:
        g: dict[int, list[int]] = {i: [] for i in range(self.num_nodes())}
        for e in self.edges:
            g[e.dst].append(e.src)
        return g

    def source_sink_paths(self) -> list[list[int]]:
        """从所有入度为 0 的源点到出度为 0 的汇点的所有简单路径 ``Π``（式 13–14）。"""
        n = self.num_nodes()
        adj = self.adj_out()
        indeg = [0] * n
        for e in self.edges:
            indeg[e.dst] += 1
        sources = [i for i in range(n) if indeg[i] == 0]
        outdeg = [0] * n
        for e in self.edges:
            outdeg[e.src] += 1
        sinks = [i for i in range(n) if outdeg[i] == 0]

        paths: list[list[int]] = []

        def dfs(cur: int, path: list[int]) -> None:
            path.append(cur)
            if cur in sinks:
                paths.append(list(path))
            else:
                for nxt in adj[cur]:
                    dfs(nxt, path)
            path.pop()

        for s in sources:
            dfs(s, [])

        # 去重（同一汇点多条路径时）
        uniq: set[tuple[int, ...]] = set()
        out: list[list[int]] = []
        for p in paths:
            t = tuple(p)
            if t not in uniq:
                uniq.add(t)
                out.append(p)
        return out


@dataclass(frozen=True, slots=True)
class CalibrationQuery:
    """校准集 ``Q`` 中的单个请求：源数据量 ``S_src^q`` 与预算 ``Θ^q``。"""

    name: str
    s_src: float
    theta_cost: float
    theta_latency: float


@dataclass(frozen=True, slots=True)
class ScenarioCoeffs:
    """
    单个 (query q, SAA 样本 k) 下，对 ``(x,y)`` 线性的代价与延迟系数。

    - ``node_exec_cost[i][k]``：节点 ``i`` 选候选 ``k`` 的执行+存储等代价系数。
    - ``edge_trans_cost[(i,j)][k][m]``：逻辑边 ``(i,j)`` 上选 ``(k,m)`` 的传输代价系数（已含数据量权重）。
    - ``node_exec_lat[i][k]``、``edge_trans_lat[(i,j)][k][m]``：延迟同理（式 11–12 的数值实例）。
    """

    node_exec_cost: list[list[float]]
    edge_trans_cost: dict[tuple[int, int], list[list[float]]]
    node_exec_lat: list[list[float]]
    edge_trans_lat: dict[tuple[int, int], list[list[float]]]


@dataclass(frozen=True, slots=True)
class CvarMilpParams:
    """CVaR–SAA 中的 ``η``（式 21–22 分母）与求解器选项。"""

    eta_cost: float = 0.05
    eta_latency: float = 0.05
    solver_time_limit_sec: int | None = 120
    solver_msg: bool = False


@dataclass(frozen=True, slots=True)
class LagrangianDualParams:
    """
    拉格朗日松弛 + 对偶子梯度参数。

    说明：
    - 将 (17) 的 McCormick 耦合约束松弛到目标函数中；
    - 在每轮中把问题分解为可并行的“节点子问题”和“边子问题”；
    - 用投影子梯度更新对偶变量。
    """

    max_iters: int = 120
    init_step_size: float = 0.8
    step_decay: float = 0.98
    min_step_size: float = 1e-4
    tol: float = 1e-4
    verbose: bool = False


# ---------------------------------------------------------------------------
# 线性目标与约束：给定 (x,y) 下的 C 与 L_π
# ---------------------------------------------------------------------------


def linear_total_cost(
    dag: LogicalDAG,
    coeff: ScenarioCoeffs,
    x_val: list[list[float]],
    y_val: dict[tuple[int, int, int, int], float],
) -> float:
    n = dag.num_nodes()
    c = 0.0
    for i in range(n):
        for k in range(len(dag.nodes[i].candidates)):
            c += coeff.node_exec_cost[i][k] * x_val[i][k]
    for e in dag.edges:
        key = (e.src, e.dst)
        mat = coeff.edge_trans_cost[key]
        for k in range(len(dag.nodes[e.src].candidates)):
            for m in range(len(dag.nodes[e.dst].candidates)):
                c += mat[k][m] * y_val[(e.src, e.dst, k, m)]
    return c


def linear_path_latency(
    dag: LogicalDAG,
    coeff: ScenarioCoeffs,
    path: list[int],
    x_val: list[list[float]],
    y_val: dict[tuple[int, int, int, int], float],
) -> float:
    """路径 ``π`` 上节点与边延迟之和（式 13）。"""
    t = 0.0
    for idx, i in enumerate(path):
        for k in range(len(dag.nodes[i].candidates)):
            t += coeff.node_exec_lat[i][k] * x_val[i][k]
        if idx + 1 < len(path):
            j = path[idx + 1]
            e = (i, j)
            mat = coeff.edge_trans_lat[e]
            for k in range(len(dag.nodes[i].candidates)):
                for m in range(len(dag.nodes[j].candidates)):
                    t += mat[k][m] * y_val[(i, j, k, m)]
    return t


# ---------------------------------------------------------------------------
# MILP 构建与求解
# ---------------------------------------------------------------------------


def _add_mccormick(
    prob: pulp.LpProblem,
    y: pulp.LpVariable,
    x_i: pulp.LpVariable,
    x_j: pulp.LpVariable,
    name: str,
) -> None:
    prob += y <= x_i, f"{name}_y_le_xi"
    prob += y <= x_j, f"{name}_y_le_xj"
    prob += y >= x_i + x_j - 1, f"{name}_y_ge_sum"
    prob += y >= 0, f"{name}_y_ge_0"


def build_cvar_sa_milp(
    dag: LogicalDAG,
    queries: Sequence[CalibrationQuery],
    scenarios: Sequence[Sequence[ScenarioCoeffs]],
    params: CvarMilpParams | None = None,
) -> tuple[pulp.LpProblem, dict[str, object]]:
    """
    构造论文式 (26) 的 MILP：最大化 ``U(x)``（式 15），满足 (16)(17) 与 CVaR–SAA (21)–(25)。

    ``scenarios[q][k]``：查询 ``q`` 的第 ``k`` 个 i.i.d. 场景（共 ``|Q|`` 行、每行长度 ``K``）。
    """
    if params is None:
        params = CvarMilpParams()

    n = dag.num_nodes()
    Q = len(queries)
    if Q == 0:
        raise ValueError("calibration set Q must be non-empty")
    K = len(scenarios[0])
    if K == 0:
        raise ValueError("SAA scenario count K must be >= 1")
    for row in scenarios:
        if len(row) != K:
            raise ValueError("all scenario rows must have the same length K")

    paths = dag.source_sink_paths()
    if not paths:
        raise ValueError("DAG has no source-to-sink path")

    prob = pulp.LpProblem("deployment_cvar_milp", sense=pulp.LpMaximize)

    # x[i][k] ∈ {0,1}
    x: list[list[pulp.LpVariable]] = []
    for i in range(n):
        row = []
        for k, _ in enumerate(dag.nodes[i].candidates):
            row.append(
                pulp.LpVariable(f"x_{i}_{k}", cat=pulp.LpBinary),
            )
        x.append(row)

    # y[(i,j)][k][m]
    y: dict[tuple[int, int], list[list[pulp.LpVariable]]] = {}
    for e in dag.edges:
        ni = len(dag.nodes[e.src].candidates)
        nj = len(dag.nodes[e.dst].candidates)
        mat: list[list[pulp.LpVariable]] = []
        for k in range(ni):
            row = []
            for m in range(nj):
                row.append(
                    pulp.LpVariable(
                        f"y_{e.src}_{e.dst}_{k}_{m}",
                        lowBound=0,
                        upBound=1,
                    )
                )
            mat.append(row)
        y[(e.src, e.dst)] = mat

    # (16) 每个逻辑节点恰选一个端点
    for i in range(n):
        prob += pulp.lpSum(x[i]) == 1, f"pick_one_{i}"

    # (17) McCormick
    for e in dag.edges:
        ni = len(dag.nodes[e.src].candidates)
        nj = len(dag.nodes[e.dst].candidates)
        for k in range(ni):
            for m in range(nj):
                _add_mccormick(
                    prob,
                    y[(e.src, e.dst)][k][m],
                    x[e.src][k],
                    x[e.dst][m],
                    f"mcc_{e.src}_{e.dst}_{k}_{m}",
                )

    # (15) 目标：max sum_i w_i * sum_k mu_{i,k} * x_{i,k}
    obj = []
    for i in range(n):
        node = dag.nodes[i]
        for k, ep in enumerate(node.candidates):
            obj.append(node.weight * ep.mu * x[i][k])
    prob += pulp.lpSum(obj), "utility"

    # CVaR – SAA：α 与 z
    alpha_c = pulp.LpVariable("alpha_C", lowBound=None, upBound=None)
    alpha_t = pulp.LpVariable("alpha_T", lowBound=None, upBound=None)

    denom_c = params.eta_cost * float(Q * K)
    denom_t = params.eta_latency * float(Q * K)

    z_c: dict[tuple[int, int], pulp.LpVariable] = {}
    z_t: dict[tuple[int, int], pulp.LpVariable] = {}

    for q in range(Q):
        theta_c = queries[q].theta_cost
        theta_t = queries[q].theta_latency
        for k in range(K):
            coeff = scenarios[q][k]
            # C(x,y) 线性形式
            c_terms: list[pulp.LpAffineExpression] = []
            for i in range(n):
                for kk in range(len(dag.nodes[i].candidates)):
                    c_terms.append(coeff.node_exec_cost[i][kk] * x[i][kk])
            for e in dag.edges:
                mat = coeff.edge_trans_cost[(e.src, e.dst)]
                for kk in range(len(dag.nodes[e.src].candidates)):
                    for m in range(len(dag.nodes[e.dst].candidates)):
                        c_terms.append(mat[kk][m] * y[(e.src, e.dst)][kk][m])
            c_expr = pulp.lpSum(c_terms)

            zcv = pulp.LpVariable(f"zC_q{q}_k{k}", lowBound=0)
            ztv = pulp.LpVariable(f"zT_q{q}_k{k}", lowBound=0)
            z_c[(q, k)] = zcv
            z_t[(q, k)] = ztv

            prob += zcv >= c_expr - theta_c - alpha_c, f"cvar_cost_slack_{q}_{k}"

            # (24) z^T >= L_π - Θ^T - α^T 对每个路径 π
            for pi_idx, path in enumerate(paths):
                lat_terms: list[pulp.LpAffineExpression] = []
                for idx, i in enumerate(path):
                    for kk in range(len(dag.nodes[i].candidates)):
                        lat_terms.append(coeff.node_exec_lat[i][kk] * x[i][kk])
                    if idx + 1 < len(path):
                        j = path[idx + 1]
                        mat = coeff.edge_trans_lat[(i, j)]
                        for kk in range(len(dag.nodes[i].candidates)):
                            for m in range(len(dag.nodes[j].candidates)):
                                lat_terms.append(mat[kk][m] * y[(i, j)][kk][m])
                l_expr = pulp.lpSum(lat_terms)
                prob += ztv >= l_expr - theta_t - alpha_t, f"cvar_lat_slack_{q}_{k}_p{pi_idx}"

    # (21)(22)
    prob += (
        alpha_c + (1.0 / denom_c) * pulp.lpSum(z_c.values()) <= 0,
        "cvar_cost_agg",
    )
    prob += (
        alpha_t + (1.0 / denom_t) * pulp.lpSum(z_t.values()) <= 0,
        "cvar_lat_agg",
    )

    meta: dict[str, object] = {
        "x": x,
        "y": y,
        "alpha_c": alpha_c,
        "alpha_t": alpha_t,
        "z_c": z_c,
        "z_t": z_t,
        "paths": paths,
    }
    return prob, meta


def solve_deployment(
    dag: LogicalDAG,
    queries: Sequence[CalibrationQuery],
    scenarios: Sequence[Sequence[ScenarioCoeffs]],
    params: CvarMilpParams | None = None,
) -> tuple[pulp.LpStatus, dict[str, object]]:
    """构建并求解 MILP，返回 PuLP 状态与变量取值说明。"""
    prob, meta = build_cvar_sa_milp(dag, queries, scenarios, params)
    if params is None:
        params = CvarMilpParams()
    solver = pulp.PULP_CBC_CMD(
        msg=params.solver_msg,
        timeLimit=params.solver_time_limit_sec,
    )
    prob.solve(solver)
    status = pulp.LpStatus[prob.status]

    x_vars: list[list[pulp.LpVariable]] = meta["x"]  # type: ignore[assignment]
    n = dag.num_nodes()
    pick: list[int | None] = [None] * n
    for i in range(n):
        for k in range(len(dag.nodes[i].candidates)):
            if x_vars[i][k].value() is not None and x_vars[i][k].value() > 0.5:
                pick[i] = k

    meta["pick"] = pick
    meta["objective_value"] = pulp.value(prob.objective)
    return status, meta


def _mean_coeffs(
    dag: LogicalDAG,
    scenarios: Sequence[Sequence[ScenarioCoeffs]],
) -> ScenarioCoeffs:
    """对所有 (q,k) 场景取均值，得到用于拉格朗日子问题的平均线性系数。"""
    n = dag.num_nodes()
    Q = len(scenarios)
    if Q == 0:
        raise ValueError("scenarios must be non-empty")
    K = len(scenarios[0])
    if K == 0:
        raise ValueError("scenario row must be non-empty")
    for row in scenarios:
        if len(row) != K:
            raise ValueError("all scenario rows must have same length")

    denom = float(Q * K)
    node_exec_cost = [[0.0 for _ in dag.nodes[i].candidates] for i in range(n)]
    node_exec_lat = [[0.0 for _ in dag.nodes[i].candidates] for i in range(n)]

    edge_trans_cost: dict[tuple[int, int], list[list[float]]] = {}
    edge_trans_lat: dict[tuple[int, int], list[list[float]]] = {}
    for e in dag.edges:
        ni = len(dag.nodes[e.src].candidates)
        nj = len(dag.nodes[e.dst].candidates)
        edge_trans_cost[(e.src, e.dst)] = [[0.0 for _ in range(nj)] for _ in range(ni)]
        edge_trans_lat[(e.src, e.dst)] = [[0.0 for _ in range(nj)] for _ in range(ni)]

    for q in range(Q):
        for k in range(K):
            coeff = scenarios[q][k]
            for i in range(n):
                for kk in range(len(dag.nodes[i].candidates)):
                    node_exec_cost[i][kk] += coeff.node_exec_cost[i][kk] / denom
                    node_exec_lat[i][kk] += coeff.node_exec_lat[i][kk] / denom
            for e in dag.edges:
                key = (e.src, e.dst)
                mat_c = coeff.edge_trans_cost[key]
                mat_l = coeff.edge_trans_lat[key]
                ni = len(dag.nodes[e.src].candidates)
                nj = len(dag.nodes[e.dst].candidates)
                for kk in range(ni):
                    for mm in range(nj):
                        edge_trans_cost[key][kk][mm] += mat_c[kk][mm] / denom
                        edge_trans_lat[key][kk][mm] += mat_l[kk][mm] / denom

    return ScenarioCoeffs(
        node_exec_cost=node_exec_cost,
        edge_trans_cost=edge_trans_cost,
        node_exec_lat=node_exec_lat,
        edge_trans_lat=edge_trans_lat,
    )


def solve_deployment_lagrangian(
    dag: LogicalDAG,
    queries: Sequence[CalibrationQuery],
    scenarios: Sequence[Sequence[ScenarioCoeffs]],
    *,
    dual_params: LagrangianDualParams | None = None,
    risk_tradeoff_cost: float = 1.0,
    risk_tradeoff_latency: float = 1.0,
) -> tuple[str, dict[str, object]]:
    """
    使用拉格朗日松弛求解“Final ILP 核心结构”：
    - 保留 (16) 每节点单选；
    - 松弛 (17) McCormick 耦合，拆成节点/边独立子问题并行可解；
    - 通过对偶子梯度迭代恢复可行一致性。

    备注：
    - 这里把多场景 CVaR 项折算到“均值 cost/latency 风险惩罚”系数里，作为可分解近似；
    - 适合作为集中式 MILP 的分布式并行 warm-start / 近似解器。
    """
    if dual_params is None:
        dual_params = LagrangianDualParams()
    if not queries:
        raise ValueError("queries must be non-empty")

    n = dag.num_nodes()
    mean_coeff = _mean_coeffs(dag, scenarios)

    theta_cost = sum(q.theta_cost for q in queries) / float(len(queries))
    theta_lat = sum(q.theta_latency for q in queries) / float(len(queries))
    _ = (theta_cost, theta_lat)  # 预算值用于元信息输出，不直接进对偶更新。

    # 对偶变量：
    # mu1[e][k][m] 对应 y - x_i <= 0
    # mu2[e][k][m] 对应 y - x_j <= 0
    # mu3[e][k][m] 对应 x_i + x_j - 1 - y <= 0
    mu1: dict[tuple[int, int], list[list[float]]] = {}
    mu2: dict[tuple[int, int], list[list[float]]] = {}
    mu3: dict[tuple[int, int], list[list[float]]] = {}
    for e in dag.edges:
        ni = len(dag.nodes[e.src].candidates)
        nj = len(dag.nodes[e.dst].candidates)
        key = (e.src, e.dst)
        mu1[key] = [[0.0 for _ in range(nj)] for _ in range(ni)]
        mu2[key] = [[0.0 for _ in range(nj)] for _ in range(ni)]
        mu3[key] = [[0.0 for _ in range(nj)] for _ in range(ni)]

    best_primal_val = float("-inf")
    best_pick: list[int | None] = [None] * n
    best_y: dict[tuple[int, int, int, int], float] = {}
    best_gap = float("inf")
    history: list[dict[str, float]] = []

    step = max(dual_params.min_step_size, dual_params.init_step_size)

    for it in range(dual_params.max_iters):
        # -------------------------
        # 1) 节点子问题（可并行）：每个 i 独立单选
        # -------------------------
        x_pick = [0 for _ in range(n)]
        x_mat = [[0.0 for _ in dag.nodes[i].candidates] for i in range(n)]
        for i in range(n):
            scores = []
            for k, ep in enumerate(dag.nodes[i].candidates):
                sc = dag.nodes[i].weight * ep.mu
                sc -= risk_tradeoff_cost * mean_coeff.node_exec_cost[i][k]
                sc -= risk_tradeoff_latency * mean_coeff.node_exec_lat[i][k]
                # 拉格朗日项贡献
                for e in dag.edges:
                    if e.src == i:
                        key = (e.src, e.dst)
                        for m in range(len(dag.nodes[e.dst].candidates)):
                            sc += mu1[key][k][m] - mu3[key][k][m]
                    if e.dst == i:
                        key = (e.src, e.dst)
                        for kk in range(len(dag.nodes[e.src].candidates)):
                            sc += mu2[key][kk][k] - mu3[key][kk][k]
                scores.append(sc)
            k_star = max(range(len(scores)), key=lambda t: scores[t])
            x_pick[i] = k_star
            x_mat[i][k_star] = 1.0

        # -------------------------
        # 2) 边子问题（可并行）：每条边独立选一个 (k,m)
        # -------------------------
        y_val: dict[tuple[int, int, int, int], float] = {}
        for e in dag.edges:
            key = (e.src, e.dst)
            ni = len(dag.nodes[e.src].candidates)
            nj = len(dag.nodes[e.dst].candidates)
            best_pair = (0, 0)
            best_score = float("-inf")
            for k in range(ni):
                for m in range(nj):
                    sc = 0.0
                    sc -= risk_tradeoff_cost * mean_coeff.edge_trans_cost[key][k][m]
                    sc -= risk_tradeoff_latency * mean_coeff.edge_trans_lat[key][k][m]
                    sc += -mu1[key][k][m] - mu2[key][k][m] + mu3[key][k][m]
                    if sc > best_score:
                        best_score = sc
                        best_pair = (k, m)
            for k in range(ni):
                for m in range(nj):
                    y_val[(e.src, e.dst, k, m)] = 1.0 if (k, m) == best_pair else 0.0

        # -------------------------
        # 3) 评估当前可行修复（由 x 诱导 y）
        # -------------------------
        y_proj: dict[tuple[int, int, int, int], float] = {}
        for e in dag.edges:
            ni = len(dag.nodes[e.src].candidates)
            nj = len(dag.nodes[e.dst].candidates)
            for k in range(ni):
                for m in range(nj):
                    y_proj[(e.src, e.dst, k, m)] = (
                        1.0 if (k == x_pick[e.src] and m == x_pick[e.dst]) else 0.0
                    )

        utility_val = 0.0
        for i in range(n):
            utility_val += dag.nodes[i].weight * dag.nodes[i].candidates[x_pick[i]].mu
        risk_cost_val = linear_total_cost(dag, mean_coeff, x_mat, y_proj)
        # DAG 仅用于单路径/多路径上限估算：这里取最大路径延迟作为全局风险代理。
        path_lat_vals = [
            linear_path_latency(dag, mean_coeff, p, x_mat, y_proj)
            for p in dag.source_sink_paths()
        ]
        risk_lat_val = max(path_lat_vals) if path_lat_vals else 0.0
        primal_val = utility_val - risk_tradeoff_cost * risk_cost_val - risk_tradeoff_latency * risk_lat_val

        # -------------------------
        # 4) 计算子梯度并更新对偶变量
        # -------------------------
        sq_norm = 0.0
        gap_sum = 0.0
        for e in dag.edges:
            key = (e.src, e.dst)
            ni = len(dag.nodes[e.src].candidates)
            nj = len(dag.nodes[e.dst].candidates)
            for k in range(ni):
                for m in range(nj):
                    ykm = y_val[(e.src, e.dst, k, m)]
                    xi = x_mat[e.src][k]
                    xj = x_mat[e.dst][m]

                    g1 = ykm - xi
                    g2 = ykm - xj
                    g3 = xi + xj - 1.0 - ykm

                    gap_sum += abs(g1) + abs(g2) + abs(g3)
                    sq_norm += g1 * g1 + g2 * g2 + g3 * g3

                    mu1[key][k][m] = max(0.0, mu1[key][k][m] + step * g1)
                    mu2[key][k][m] = max(0.0, mu2[key][k][m] + step * g2)
                    mu3[key][k][m] = max(0.0, mu3[key][k][m] + step * g3)

        best_primal_val = max(best_primal_val, primal_val)
        if primal_val >= best_primal_val - 1e-12:
            best_pick = list(x_pick)
            best_y = dict(y_proj)
            best_gap = gap_sum

        history.append(
            {
                "iter": float(it + 1),
                "step": step,
                "primal_val": primal_val,
                "best_primal_val": best_primal_val,
                "relaxed_violation_l1": gap_sum,
                "subgrad_norm_l2": sq_norm**0.5,
            }
        )
        if dual_params.verbose:
            print(
                "[lagrangian]",
                f"iter={it+1:03d}",
                f"step={step:.4g}",
                f"primal={primal_val:.6f}",
                f"best={best_primal_val:.6f}",
                f"viol={gap_sum:.6f}",
            )

        if gap_sum <= dual_params.tol:
            break
        step = max(dual_params.min_step_size, step * dual_params.step_decay)

    meta: dict[str, object] = {
        "method": "lagrangian_dual_decomposition",
        "pick": best_pick,
        "y": best_y,
        "objective_value": best_primal_val,
        "relaxed_violation_l1": best_gap,
        "history": history,
        "risk_tradeoff_cost": risk_tradeoff_cost,
        "risk_tradeoff_latency": risk_tradeoff_latency,
        "avg_theta_cost": theta_cost,
        "avg_theta_latency": theta_lat,
    }
    status = "Converged" if best_gap <= dual_params.tol else "Stopped"
    return status, meta


def print_solution(
    dag: LogicalDAG,
    meta: dict[str, object],
) -> None:
    """人类可读的选中端点与目标值。"""
    pick: list[int | None] = meta["pick"]  # type: ignore[assignment]
    obj = meta.get("objective_value")
    print("Objective U(x):", obj)
    for i, k in enumerate(pick):
        if k is None:
            print(f"  node {i} ({dag.nodes[i].name}): UNSAT")
            continue
        ep = dag.nodes[i].candidates[k]
        print(f"  node {i} ({dag.nodes[i].name}): {ep.name} (mu={ep.mu})")


# ---------------------------------------------------------------------------
# 基于 simulation 环境：从 Workflow 生成 DAG 与线性场景系数
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class WorkflowDag:
    """把 Workflow 的候选节点与 LogicalDAG 对齐，便于生成系数矩阵。"""

    dag: LogicalDAG
    segment_nodes: list[SegmentNode]
    split_nodes: list[SplitNode]
    caption_nodes: list[CaptionCloudNode | CaptionNonCloudNode]
    query_nodes: list[QueryCloudNode | QueryNonCloudNode]


def build_workflow_chain_dag(wf: Workflow) -> WorkflowDag:
    """
    将你的 simulation 环境中四阶段流水线映射成 LogicalDAG：
      segment -> split -> caption -> query

    candidates 来自 wf.nodes.*，mu 取节点 utility；weight 取 config.yaml 的 utility_weight。
    """
    # utility_weight 与 Workflow 内部保持同口径：来自 config.yaml
    from distribution import config as simulation_config  # local import to avoid cycles

    wcfg = simulation_config.get("utility_weight") or {}
    w_seg = float(wcfg.get("segment", 0.0))
    w_spl = float(wcfg.get("split", 0.0))
    w_cap = float(wcfg.get("caption", 0.0))
    w_qry = float(wcfg.get("query", 0.0))

    seg_nodes = list(wf.nodes.segment.values())
    spl_nodes = list(wf.nodes.split.values())
    cap_nodes = list(wf.nodes.caption.values())
    qry_nodes = list(wf.nodes.query.values())

    dag = LogicalDAG(
        nodes=[
            LogicalNode(
                "segment",
                weight=w_seg,
                candidates=[PhysicalEndpoint(n.name, float(n.utility)) for n in seg_nodes],
            ),
            LogicalNode(
                "split",
                weight=w_spl,
                candidates=[PhysicalEndpoint(n.name, float(n.utility)) for n in spl_nodes],
            ),
            LogicalNode(
                "caption",
                weight=w_cap,
                candidates=[PhysicalEndpoint(n.name, float(n.utility)) for n in cap_nodes],
            ),
            LogicalNode(
                "query",
                weight=w_qry,
                candidates=[PhysicalEndpoint(n.name, float(n.utility)) for n in qry_nodes],
            ),
        ],
        edges=[
            LogicalEdge(0, 1),
            LogicalEdge(1, 2),
            LogicalEdge(2, 3),
        ],
    )

    return WorkflowDag(
        dag=dag,
        segment_nodes=seg_nodes,
        split_nodes=spl_nodes,
        caption_nodes=cap_nodes,
        query_nodes=qry_nodes,
    )


def _fit_pairwise_decomposition(
    observations: list[tuple[int, int, int, int, float]],
    n0: int,
    n1: int,
    n2: int,
    n3: int,
    *,
    iters: int = 8,
) -> tuple[float, list[float], list[float], list[float], list[float], list[list[float]], list[list[float]], list[list[float]]]:
    """
    Black-box拟合:
      y ~= c + a0[i0] + a1[i1] + a2[i2] + a3[i3]
              + b01[i0,i1] + b12[i1,i2] + b23[i2,i3]
    使用分块坐标下降（backfitting）做最小二乘近似，不读取任何底层分布参数。
    """
    if not observations:
        raise ValueError("observations must be non-empty")

    c = sum(v for _, _, _, _, v in observations) / float(len(observations))
    a0 = [0.0] * n0
    a1 = [0.0] * n1
    a2 = [0.0] * n2
    a3 = [0.0] * n3
    b01 = [[0.0 for _ in range(n1)] for _ in range(n0)]
    b12 = [[0.0 for _ in range(n2)] for _ in range(n1)]
    b23 = [[0.0 for _ in range(n3)] for _ in range(n2)]

    def pred(i0: int, i1: int, i2: int, i3: int) -> float:
        return c + a0[i0] + a1[i1] + a2[i2] + a3[i3] + b01[i0][i1] + b12[i1][i2] + b23[i2][i3]

    def _center_vec(v: list[float]) -> float:
        m = sum(v) / float(len(v))
        for i in range(len(v)):
            v[i] -= m
        return m

    def _center_mat(mtx: list[list[float]]) -> float:
        s = 0.0
        cnt = 0
        for r in mtx:
            for x in r:
                s += x
                cnt += 1
        m = s / float(cnt)
        for i in range(len(mtx)):
            for j in range(len(mtx[i])):
                mtx[i][j] -= m
        return m

    for _ in range(max(1, iters)):
        # update a0
        sum0 = [0.0] * n0
        cnt0 = [0] * n0
        for i0, i1, i2, i3, y in observations:
            r = y - (pred(i0, i1, i2, i3) - a0[i0])
            sum0[i0] += r
            cnt0[i0] += 1
        for i in range(n0):
            if cnt0[i]:
                a0[i] = sum0[i] / float(cnt0[i])
        c += _center_vec(a0)

        # update a1
        sum1 = [0.0] * n1
        cnt1 = [0] * n1
        for i0, i1, i2, i3, y in observations:
            r = y - (pred(i0, i1, i2, i3) - a1[i1])
            sum1[i1] += r
            cnt1[i1] += 1
        for i in range(n1):
            if cnt1[i]:
                a1[i] = sum1[i] / float(cnt1[i])
        c += _center_vec(a1)

        # update a2
        sum2 = [0.0] * n2
        cnt2 = [0] * n2
        for i0, i1, i2, i3, y in observations:
            r = y - (pred(i0, i1, i2, i3) - a2[i2])
            sum2[i2] += r
            cnt2[i2] += 1
        for i in range(n2):
            if cnt2[i]:
                a2[i] = sum2[i] / float(cnt2[i])
        c += _center_vec(a2)

        # update a3
        sum3 = [0.0] * n3
        cnt3 = [0] * n3
        for i0, i1, i2, i3, y in observations:
            r = y - (pred(i0, i1, i2, i3) - a3[i3])
            sum3[i3] += r
            cnt3[i3] += 1
        for i in range(n3):
            if cnt3[i]:
                a3[i] = sum3[i] / float(cnt3[i])
        c += _center_vec(a3)

        # update b01
        s01 = [[0.0 for _ in range(n1)] for _ in range(n0)]
        c01 = [[0 for _ in range(n1)] for _ in range(n0)]
        for i0, i1, i2, i3, y in observations:
            r = y - (pred(i0, i1, i2, i3) - b01[i0][i1])
            s01[i0][i1] += r
            c01[i0][i1] += 1
        for i in range(n0):
            for j in range(n1):
                if c01[i][j]:
                    b01[i][j] = s01[i][j] / float(c01[i][j])
        c += _center_mat(b01)

        # update b12
        s12 = [[0.0 for _ in range(n2)] for _ in range(n1)]
        c12 = [[0 for _ in range(n2)] for _ in range(n1)]
        for i0, i1, i2, i3, y in observations:
            r = y - (pred(i0, i1, i2, i3) - b12[i1][i2])
            s12[i1][i2] += r
            c12[i1][i2] += 1
        for i in range(n1):
            for j in range(n2):
                if c12[i][j]:
                    b12[i][j] = s12[i][j] / float(c12[i][j])
        c += _center_mat(b12)

        # update b23
        s23 = [[0.0 for _ in range(n3)] for _ in range(n2)]
        c23 = [[0 for _ in range(n3)] for _ in range(n2)]
        for i0, i1, i2, i3, y in observations:
            r = y - (pred(i0, i1, i2, i3) - b23[i2][i3])
            s23[i2][i3] += r
            c23[i2][i3] += 1
        for i in range(n2):
            for j in range(n3):
                if c23[i][j]:
                    b23[i][j] = s23[i][j] / float(c23[i][j])
        c += _center_mat(b23)

    return c, a0, a1, a2, a3, b01, b12, b23


def workflow_scenarios_from_simulation(
    queries: Sequence[CalibrationQuery],
    *,
    K: int,
    deterministic: bool = False,
    samples_per_scenario: int = 1200,
    fit_iters: int = 8,
    rng_seed: int = 7,
    show_progress: bool = True,
) -> tuple[WorkflowDag, list[list[ScenarioCoeffs]]]:
    """
    严格 black-box 版本：
    - 不读取 distribution 参数矩阵、不访问 wf.params 的分布细节；
    - 仅通过 repeated `Workflow.calculate(...)` 采样 cost/latency；
    - 再拟合成 MILP 所需的 node/edge 线性系数。
    """
    if K < 1:
        raise ValueError("K must be >= 1")

    # 先生成一个“结构模板”的 DAG（候选集合大小固定）；后续每个场景用同样的候选顺序。
    wf0 = Workflow()
    wdag0 = build_workflow_chain_dag(wf0)

    # 缓存候选名顺序（跨场景对齐）
    seg_names = [n.name for n in wdag0.segment_nodes]
    spl_names = [n.name for n in wdag0.split_nodes]
    cap_names = [n.name for n in wdag0.caption_nodes]
    qry_names = [n.name for n in wdag0.query_nodes]

    scenarios: list[list[ScenarioCoeffs]] = [[None for _ in range(K)] for _ in range(len(queries))]  # type: ignore[list-item]

    rnd = random.Random(rng_seed)
    total_tasks = max(1, K * len(queries))
    done_tasks = 0
    start_ts = time.time()

    def _render_progress(done: int) -> None:
        if not show_progress:
            return
        now = time.time()
        elapsed = now - start_ts
        ratio = min(max(done / total_tasks, 0.0), 1.0)
        eta = (elapsed / done) * (total_tasks - done) if done > 0 else 0.0
        width = 28
        filled = int(round(width * ratio))
        bar = "#" * filled + "-" * (width - filled)
        print(
            f"\r[algo] 采样拟合进度 [{bar}] {done}/{total_tasks} "
            f"({ratio*100:5.1f}%) | elapsed {elapsed:6.1f}s | eta {eta:6.1f}s",
            end="",
            flush=True,
        )

    for k in range(K):
        # 每个 k 使用一个新的 Workflow 实例（内部自采样环境），算法层看不到分布细节
        wf = Workflow()

        # 对齐候选顺序
        seg_nodes = [wf.nodes.segment[nm] for nm in seg_names]
        spl_nodes = [wf.nodes.split[nm] for nm in spl_names]
        cap_nodes = [wf.nodes.caption[nm] for nm in cap_names]
        qry_nodes = [wf.nodes.query[nm] for nm in qry_names]

        for qi, q in enumerate(queries):
            obs_cost: list[tuple[int, int, int, int, float]] = []
            obs_lat: list[tuple[int, int, int, int, float]] = []
            s_MB = float(q.s_src)

            # 随机抽样 pipeline 并调用 black-box simulator
            for _ in range(max(1, samples_per_scenario)):
                i0 = rnd.randrange(len(seg_nodes))
                i1 = rnd.randrange(len(spl_nodes))
                i2 = rnd.randrange(len(cap_nodes))
                i3 = rnd.randrange(len(qry_nodes))
                rr = wf.calculate(
                    seg_nodes[i0],
                    spl_nodes[i1],
                    cap_nodes[i2],
                    qry_nodes[i3],
                    s_MB,
                    deterministic=deterministic,
                )
                obs_cost.append((i0, i1, i2, i3, float(rr.cost)))
                obs_lat.append((i0, i1, i2, i3, float(rr.latency)))

            c0, a0c, a1c, a2c, a3c, b01c, b12c, b23c = _fit_pairwise_decomposition(
                obs_cost,
                len(seg_nodes),
                len(spl_nodes),
                len(cap_nodes),
                len(qry_nodes),
                iters=fit_iters,
            )
            t0, a0t, a1t, a2t, a3t, b01t, b12t, b23t = _fit_pairwise_decomposition(
                obs_lat,
                len(seg_nodes),
                len(spl_nodes),
                len(cap_nodes),
                len(qry_nodes),
                iters=fit_iters,
            )

            # 将常数项分摊到第一个stage，保证线性表达可被现有MILP直接消费
            a0c[0] += c0
            a0t[0] += t0

            scenarios[qi][k] = ScenarioCoeffs(
                node_exec_cost=[a0c, a1c, a2c, a3c],
                edge_trans_cost={(0, 1): b01c, (1, 2): b12c, (2, 3): b23c},
                node_exec_lat=[a0t, a1t, a2t, a3t],
                edge_trans_lat={(0, 1): b01t, (1, 2): b12t, (2, 3): b23t},
            )
            done_tasks += 1
            _render_progress(done_tasks)

    if show_progress:
        _render_progress(total_tasks)
        print()

    return wdag0, scenarios


def demo_with_simulation_env() -> None:
    """
    用你创建的 simulation 环境生成系数并求解一次 MILP。
    """
    # 用同一口径的 Query 生成器，转成 algo.CalibrationQuery
    qs: list[Query] = sample_query_with_budget(2)
    queries = [
        CalibrationQuery(
            f"q{i+1}",
            s_src=float(q.data_size_MB),
            theta_cost=float(q.cost_budget),
            theta_latency=float(q.latency_budget),
        )
        for i, q in enumerate(qs)
    ]

    wdag, scenarios = workflow_scenarios_from_simulation(queries, K=3, deterministic=False)

    status, meta = solve_deployment(
        wdag.dag,
        queries,
        scenarios,
        CvarMilpParams(eta_cost=0.1, eta_latency=0.1, solver_msg=False),
    )
    print("Status:", status)
    print_solution(wdag.dag, meta)


if __name__ == "__main__":
    demo_with_simulation_env()
