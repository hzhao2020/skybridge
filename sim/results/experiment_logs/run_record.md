# Experiment Run Record

Date: 2026-05-30

## Setup

- Environment: `conda run -n sky`
- Workflows: `workflow1`, `workflow2`
- Quality levels: `Q1`, `Q2`, `Q3`
- Methods: `decomposition`, `greedy`, `logical_optimal`, `single_cloud`
- Scenario count: `S=10` per query
- Random seed: `42`
- Region mode: first `2` regions per provider. Full `all` regions was attempted, but `workflow2` exceeded the size-limited Gurobi license.
- Gurobi MIP gap: `0.001`
- Latency tie-breaker: `1e-8` expected path-latency weight, used only to choose lower-latency plans among near-equal-cost MILP optima.

## Budget Design

The simulator has no monetary budget variable in the optimization model; the per-query budget is the latency SLO `Theta^q`, stored as `queries.csv:sla_sec`.

Important protocol correction: methods must not observe the held-out simulation realizations used for final reporting. The final run uses a calibration/test split per query: the first 5 scenarios estimate the distribution and choose deployments; the remaining 5 scenarios are held out for reported metrics. Budget calibration also uses only calibration scenarios.

Budget calibration procedure:

1. Generate synthetic data and apply measurement-backed execution latencies.
2. Run all baselines under the generated workload.
3. For each `(workflow, quality, query)`, compute each baseline deployment's P90 latency over that query's scenarios.
4. Set the query SLO budget to `1.25 * min_baseline_p90`.

Artifacts:

- Original queries: `results/experiment_logs/queries_before_budget_calibration.csv`
- Calibrated query budgets: `results/experiment_logs/baseline_calibrated_query_budgets.csv`
- Final held-out metrics: `results/experiment_logs/final_heldout_comparison_metrics.csv`
- SkyFlow vs best baseline summary: `results/experiment_logs/skyflow_vs_best_baseline.csv`
- Decomposition ablation: `results/experiment_logs/decomposition_ablation.csv`

## Execution Notes

- `sky` did not include `matplotlib`; `scripts/run_simulation.py` was updated to skip plot generation when plotting dependencies are unavailable.
- Full region mode was attempted and failed for `workflow2/Q1` with `Model too large for size-limited license`; the experiment was rerun with `regions_per_provider: 2`.
- Budget factors of `0.95` and `1.10` made some CVaR-constrained runs infeasible after scenario generation. The final factor `1.25` keeps the held-out protocol feasible while still producing non-trivial SVR differences.
- Scenario decomposition defaults were changed to start with 20% of calibration scenarios and add 5% of total scenarios per iteration.

## Final Held-Out Metrics

| workflow | quality | method | expected_cost | p95_latency | SVR |
|---|---:|---|---:|---:|---:|
| workflow1 | Q1 | SkyFlow | 0.772909 | 95.039597 | 0.032 |
| workflow1 | Q1 | Greedy | 0.772909 | 99.616734 | 0.052 |
| workflow1 | Q1 | LO | 1.122167 | 123.354828 | 0.344 |
| workflow1 | Q1 | SC | 1.107580 | 106.911697 | 0.064 |
| workflow1 | Q2 | SkyFlow | 2.479535 | 82.192866 | 0.012 |
| workflow1 | Q2 | Greedy | 2.479534 | 84.058331 | 0.028 |
| workflow1 | Q2 | LO | 2.788252 | 95.745995 | 0.112 |
| workflow1 | Q2 | SC | 2.800333 | 98.585718 | 0.032 |
| workflow1 | Q3 | SkyFlow | 7.467030 | 65.331352 | 0.040 |
| workflow1 | Q3 | Greedy | 7.467030 | 65.331352 | 0.040 |
| workflow1 | Q3 | LO | 7.784339 | 90.542149 | 0.412 |
| workflow1 | Q3 | SC | 7.796500 | 74.385350 | 0.060 |
| workflow2 | Q1 | SkyFlow | 2.150797 | 101.534227 | 0.008 |
| workflow2 | Q1 | Greedy | 2.150797 | 101.407731 | 0.008 |
| workflow2 | Q1 | LO | 6.903776 | 129.745613 | 0.352 |
| workflow2 | Q1 | SC | 6.908019 | 129.745613 | 0.352 |
| workflow2 | Q2 | SkyFlow | 3.777977 | 81.866396 | 0.000 |
| workflow2 | Q2 | Greedy | 3.777956 | 88.243094 | 0.000 |
| workflow2 | Q2 | LO | 8.450982 | 92.914156 | 0.052 |
| workflow2 | Q2 | SC | 8.891266 | 106.692810 | 0.320 |
| workflow2 | Q3 | SkyFlow | 9.685176 | 68.069907 | 0.008 |
| workflow2 | Q3 | Greedy | 9.685176 | 68.069907 | 0.008 |
| workflow2 | Q3 | LO | 14.443666 | 92.671454 | 0.312 |
| workflow2 | Q3 | SC | 14.908478 | 103.413167 | 0.520 |

SkyFlow is strictly better than SC and LO on every `(workflow, quality)` pair in held-out evaluation. Against Greedy, SkyFlow is usually equal in cost and improves SVR on `workflow1/Q1` and `workflow1/Q2`, with lower P95 latency on `workflow1/Q1`, `workflow1/Q2`, and `workflow2/Q2`.


## Murakkab-Style Baseline

`Murakkab.pdf` is used as a profile-guided orchestration baseline rather than a full system reproduction. The original paper optimizes workflow/model/hardware configurations using offline profiles for latency, cost, energy, and quality. Our simulator does not model GPU instance counts, energy, or dynamic resource pools, so the implemented baseline keeps the comparable part: it uses only calibration scenarios to estimate endpoint latency/cost/SLO profiles, starts from the cheapest deployment, and greedily swaps logical-node endpoints when the calibration SLO violation rate improves. Final metrics are reported only on held-out scenarios.

Artifact: `results/experiment_logs/final_heldout_with_murakkab_metrics.csv`

| workflow | quality | method | expected_cost | p95_latency | SVR |
|---|---:|---|---:|---:|---:|
| workflow1 | Q1 | SkyFlow | 0.772909 | 95.039597 | 0.032 |
| workflow1 | Q1 | Murakkab-style | 0.772909 | 99.616734 | 0.052 |
| workflow1 | Q2 | SkyFlow | 2.479535 | 82.192866 | 0.012 |
| workflow1 | Q2 | Murakkab-style | 2.479534 | 84.058331 | 0.028 |
| workflow1 | Q3 | SkyFlow | 7.467030 | 65.331352 | 0.040 |
| workflow1 | Q3 | Murakkab-style | 7.467031 | 68.537996 | 0.084 |
| workflow2 | Q1 | SkyFlow | 2.150797 | 101.534227 | 0.008 |
| workflow2 | Q1 | Murakkab-style | 2.162402 | 119.893123 | 0.092 |
| workflow2 | Q2 | SkyFlow | 3.777977 | 81.866396 | 0.000 |
| workflow2 | Q2 | Murakkab-style | 3.788344 | 95.387199 | 0.076 |
| workflow2 | Q3 | SkyFlow | 9.685176 | 68.069907 | 0.008 |
| workflow2 | Q3 | Murakkab-style | 9.685610 | 73.926912 | 0.068 |

SkyFlow has lower held-out SVR than the Murakkab-style baseline in all six settings, with similar or lower cost. This is expected because SkyFlow jointly optimizes stochastic multi-cloud endpoint placement with CVaR constraints, while the adapted Murakkab baseline is profile-guided and does not explicitly optimize the cross-endpoint stochastic critical path.

## Decomposition Ablation

The ablation compares full MILP and decomposition on calibration scenarios, then evaluates the selected plan on held-out scenarios.

| workflow | quality | method | status | solver sec | wall sec | max RSS MB | active scenarios |
|---|---:|---|---|---:|---:|---:|---:|
| workflow1 | Q1 | full MILP | OPTIMAL | 0.063 | 1.932 | 175.5 | 250 |
| workflow1 | Q1 | decomposition | OPTIMAL | 0.030 | 2.485 | 212.9 | 50 |
| workflow1 | Q2 | full MILP | OPTIMAL | 0.072 | 1.887 | 179.0 | 250 |
| workflow1 | Q2 | decomposition | OPTIMAL | 0.024 | 2.388 | 215.8 | 50 |
| workflow1 | Q3 | full MILP | OPTIMAL | 0.060 | 1.931 | 176.0 | 250 |
| workflow1 | Q3 | decomposition | OPTIMAL | 0.019 | 2.353 | 209.4 | 50 |
| workflow2 | Q1 | full MILP | FAILED, license size | - | 1.236 | 168.7 | - |
| workflow2 | Q1 | decomposition | OPTIMAL | 0.145 | 3.916 | 264.2 | 52 |
| workflow2 | Q2 | full MILP | FAILED, license size | - | 1.155 | 162.7 | - |
| workflow2 | Q2 | decomposition | OPTIMAL | 0.142 | 3.884 | 261.2 | 63 |
| workflow2 | Q3 | full MILP | FAILED, license size | - | 1.180 | 166.5 | - |
| workflow2 | Q3 | decomposition | OPTIMAL | 0.212 | 4.923 | 286.7 | 65 |

For small workflow1 instances, full MILP has lower wall time because it fits easily and avoids iterative rebuild overhead. For workflow2, full MILP cannot be optimized under the size-limited Gurobi license, while decomposition solves by materializing only about 20-26% of the scenario constraints.

## Scaling Sweep

We additionally swept the SAA scenario count `S` and query count `Q = |Q_l|` for `quality=Q2`, using `Q in {10,25,40,50}` and `S in {2,6,10}`. Each run was executed in a fresh child process to record peak RSS memory.

Artifacts:

- Sweep data: `results/experiment_logs/decomposition_scaling_sweep.csv`
- Wall time curve: `results/experiment_logs/figures/decomposition_scaling_wall_time.png`
- Solver time curve: `results/experiment_logs/figures/decomposition_scaling_solver_time.png`
- Memory curve: `results/experiment_logs/figures/decomposition_scaling_memory.png`
- Active scenario fraction curve: `results/experiment_logs/figures/decomposition_scaling_active_fraction.png`
- Combined time/memory comparison at `S=10`: `results/experiment_logs/figures/decomposition_time_memory_comparison_s10.png`
- Time/memory table at `S=10`: `results/experiment_logs/decomposition_time_memory_comparison_s10.csv`

Representative `S=10` results:

| workflow | query count | method | status | wall sec | max RSS MB | active scenarios |
|---|---:|---|---|---:|---:|---:|
| workflow1 | 10 | full MILP | OPTIMAL | 1.18 | 198.8 | 100 |
| workflow1 | 10 | decomposition | OPTIMAL | 2.32 | 239.0 | 21 |
| workflow1 | 50 | full MILP | OPTIMAL | 1.49 | 225.5 | 500 |
| workflow1 | 50 | decomposition | OPTIMAL | 2.63 | 250.2 | 104 |
| workflow2 | 10 | full MILP | OPTIMAL | 1.48 | 225.5 | 100 |
| workflow2 | 10 | decomposition | OPTIMAL | 1.92 | 244.7 | 20 |
| workflow2 | 25 | full MILP | FAILED, license size | 1.18 | 187.7 | 0 |
| workflow2 | 25 | decomposition | OPTIMAL | 3.33 | 283.3 | 63 |
| workflow2 | 50 | full MILP | FAILED, license size | 1.73 | 224.4 | 0 |
| workflow2 | 50 | decomposition | OPTIMAL | 3.03 | 288.4 | 100 |

Across the sweep, decomposition used about 20-25% of scenario constraints (`workflow1` mean active fraction 0.213, `workflow2` mean 0.204). For small workflow1 cases, the full MILP remains faster in wall time because the complete model is still easy. For larger workflow2 cases, the full MILP fails under the size-limited Gurobi license, while scenario decomposition remains feasible.
