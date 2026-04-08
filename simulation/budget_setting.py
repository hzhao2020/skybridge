"""
Data-driven budget profiling for config.yaml.

Goal
----
Estimate linear budget parameters:
  latency_budget(S) = a_L + b_L * S
  cost_budget(S)    = a_C + b_C * S

using your simulation environment:
  distribution.py + nodes.py + workflow.py + config.py/config.yaml

Method (practical + reproducible)
---------------------------------
1) Pick candidate deployment pipelines (segment/split/caption/query endpoint tuples).
2) For each data size S, compute:
   - Deterministic mean cost/latency for each pipeline (deterministic=True),
     then take the minimum as a lower bound.
3) For each S, take the best deterministic pipeline(s) and run Monte Carlo,
   then extract (1-eta) quantiles (e.g. 95th percentile).
4) Fit a line y ≈ a + b*x via least squares.

Notes
-----
- Exhaustive enumeration of all combinations can be large; this script supports:
  - single-cloud pipelines (12 combos) as a fast baseline
  - a limited random sample of cross-stage combinations for broader coverage
"""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass

from workflow import Workflow


@dataclass(frozen=True, slots=True)
class Pipeline:
    seg_name: str
    spl_name: str
    cap_name: str
    qry_name: str


def _percentile(xs: list[float], q: float) -> float:
    """q in [0,1]. Nearest-rank with linear interpolation."""
    if not xs:
        raise ValueError("empty sample")
    if q <= 0:
        return min(xs)
    if q >= 1:
        return max(xs)
    ys = sorted(xs)
    pos = q * (len(ys) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return ys[lo]
    w = pos - lo
    return ys[lo] * (1.0 - w) + ys[hi] * w


def _fit_line(xs: list[float], ys: list[float]) -> tuple[float, float]:
    """Least squares fit: y ≈ a + b*x. Returns (a,b)."""
    if len(xs) != len(ys) or not xs:
        raise ValueError("xs/ys length mismatch or empty")
    n = float(len(xs))
    sx = sum(xs)
    sy = sum(ys)
    sxx = sum(x * x for x in xs)
    sxy = sum(x * y for x, y in zip(xs, ys))
    denom = n * sxx - sx * sx
    if abs(denom) < 1e-12:
        raise ValueError("degenerate x values for linear fit")
    b = (n * sxy - sx * sy) / denom
    a = (sy - b * sx) / n
    return a, b


def build_candidate_pipelines(
    wf: Workflow,
    *,
    random_samples: int = 2000,
    seed: int = 0,
) -> list[Pipeline]:
    """
    Build a reasonably rich candidate set.
    - Always include all single-cloud pipelines.
    - Add random cross-stage combinations (to discover cheaper/faster mixes).
    """
    rnd = random.Random(seed)

    segs = list(wf.nodes.segment.keys())
    spls = list(wf.nodes.split.keys())
    caps = list(wf.nodes.caption.keys())
    qrys = list(wf.nodes.query.keys())

    out: list[Pipeline] = []

    # single-cloud (provider,region) consistent across 4 stages
    for seg in segs:
        # seg name looks like "p?_r?_segment" -> prefix "p?_r?"
        pr = seg.rsplit("_", 1)[0]
        spl = f"{pr}_split"
        cap = f"{pr}_caption"
        qry = f"{pr}_query"
        if spl in wf.nodes.split and cap in wf.nodes.caption and qry in wf.nodes.query:
            out.append(Pipeline(seg, spl, cap, qry))

    # random mixes
    seen = {(p.seg_name, p.spl_name, p.cap_name, p.qry_name) for p in out}
    for _ in range(max(0, int(random_samples))):
        p = Pipeline(
            rnd.choice(segs),
            rnd.choice(spls),
            rnd.choice(caps),
            rnd.choice(qrys),
        )
        key = (p.seg_name, p.spl_name, p.cap_name, p.qry_name)
        if key in seen:
            continue
        seen.add(key)
        out.append(p)

    return out


def profile(
    *,
    sizes_mb: list[float],
    eta: float,
    mc_runs: int,
    random_pipelines: int,
    seed: int,
) -> None:
    wf = Workflow()
    cands = build_candidate_pipelines(wf, random_samples=random_pipelines, seed=seed)

    q = 1.0 - eta
    print(f"Candidates: {len(cands)} pipelines")
    print(f"Quantile: p{int(round(q*100))} (eta={eta})")
    print("")

    xs: list[float] = []
    cost_qs: list[float] = []
    lat_qs: list[float] = []

    for s in sizes_mb:
        best_mean_cost = (float("inf"), None)  # (value, pipeline)
        best_mean_lat = (float("inf"), None)

        # deterministic screening
        for p in cands:
            seg = wf.nodes.segment[p.seg_name]
            spl = wf.nodes.split[p.spl_name]
            cap = wf.nodes.caption[p.cap_name]
            qry = wf.nodes.query[p.qry_name]
            r = wf.calculate(seg, spl, cap, qry, float(s), deterministic=True)
            if r.cost < best_mean_cost[0]:
                best_mean_cost = (r.cost, p)
            if r.latency < best_mean_lat[0]:
                best_mean_lat = (r.latency, p)

        assert best_mean_cost[1] is not None and best_mean_lat[1] is not None

        # MC on the best pipelines (cost-best and latency-best)
        def run_mc(p: Pipeline) -> tuple[list[float], list[float]]:
            seg = wf.nodes.segment[p.seg_name]
            spl = wf.nodes.split[p.spl_name]
            cap = wf.nodes.caption[p.cap_name]
            qry = wf.nodes.query[p.qry_name]
            cs: list[float] = []
            ts: list[float] = []
            for _ in range(mc_runs):
                rr = wf.calculate(seg, spl, cap, qry, float(s), deterministic=False)
                cs.append(rr.cost)
                ts.append(rr.latency)
            return cs, ts

        cs1, ts1 = run_mc(best_mean_cost[1])
        cs2, ts2 = run_mc(best_mean_lat[1])

        c_q = max(_percentile(cs1, q), _percentile(cs2, q))
        t_q = max(_percentile(ts1, q), _percentile(ts2, q))

        xs.append(float(s))
        cost_qs.append(float(c_q))
        lat_qs.append(float(t_q))

        print(
            f"S={s:>7.2f}MB  "
            f"C_det_min={best_mean_cost[0]:>10.4f}  T_det_min={best_mean_lat[0]:>10.2f}  "
            f"C_p{int(round(q*100))}={c_q:>10.4f}  T_p{int(round(q*100))}={t_q:>10.2f}"
        )

    a_c, b_c = _fit_line(xs, cost_qs)
    a_t, b_t = _fit_line(xs, lat_qs)

    print("\n=== Suggested baseline (fit on quantiles) ===")
    print(f"cost_intercept_usd:    {a_c:.6f}")
    print(f"cost_slope_per_mb:     {b_c:.6f}")
    print(f"latency_intercept_s:   {a_t:.6f}")
    print(f"latency_slope_per_mb:  {b_t:.6f}")

    print("\n=== config.yaml snippet (budget.baseline + slack_factor) ===")
    print("budget:")
    print("  baseline:")
    print(f"    latency_intercept_s: {a_t:.6f}")
    print(f"    latency_slope_per_mb: {b_t:.6f}")
    print(f"    cost_intercept_usd: {a_c:.6f}")
    print(f"    cost_slope_per_mb: {b_c:.6f}")
    print("  slack_factor:")
    print("    latency: 1.20")
    print("    cost: 1.20")


def main() -> None:
    ap = argparse.ArgumentParser(description="Profile feasible budgets from simulation environment.")
    ap.add_argument("--eta", type=float, default=0.05, help="CVaR/quantile tail probability (eta).")
    ap.add_argument("--mc", type=int, default=200, help="Monte Carlo runs per size for quantiles.")
    ap.add_argument("--random-pipelines", type=int, default=2000, help="Random cross-stage pipelines to sample.")
    ap.add_argument("--seed", type=int, default=0, help="Random seed for pipeline sampling.")
    ap.add_argument("--min-mb", type=float, default=5.0)
    ap.add_argument("--max-mb", type=float, default=500.0)
    ap.add_argument("--step-mb", type=float, default=25.0)
    args = ap.parse_args()

    sizes: list[float] = []
    x = float(args.min_mb)
    while x <= float(args.max_mb) + 1e-9:
        sizes.append(x)
        x += float(args.step_mb)

    profile(
        sizes_mb=sizes,
        eta=float(args.eta),
        mc_runs=int(args.mc),
        random_pipelines=int(args.random_pipelines),
        seed=int(args.seed),
    )


if __name__ == "__main__":
    main()

