# SkyFlow Simulation Framework

SkyFlow is an offline deployment optimizer for multi-cloud agentic video analytics workflows. This repository provides a **research prototype** that simulates workflow DAGs, heterogeneous cloud endpoints, stochastic query workloads, and solves a SAA-CVaR mixed-integer program (MILP) for cost-minimizing deployment under tail latency risk constraints.

## Features

- Four workflow DAGs covering sampled-frame, caption, temporal grounding, and retrieval-augmented video QA
- Heterogeneous endpoints across GCP, AWS, Aliyun, and Azure
- Quality levels Q1 / Q2 / Q3 with model and shot-aware sampling mappings
- Synthetic data generation (replaceable via CSV)
- Full SAA-CVaR MILP and critical-path-aware scenario-path cut generation
- Modular ablation flags for future experiments
- Result export and matplotlib plots

## Requirements

- Python 3.10+
- **Gurobi** (required; no solver fallback)
- See `requirements.txt` for Python packages

**License note:** The restricted (size-limited) Gurobi license may not build the larger full MILPs at high query–scenario scale, especially workflow4. Current defaults are tuned for decomposition; use an unrestricted license for large full-MILP runs.

Install Gurobi and obtain a license from [Gurobi](https://www.gurobi.com/). Set `GRB_LICENSE_FILE` if needed.

```bash
pip install -r requirements.txt
```

## Project Structure

```text
configs/          YAML workflow and default parameters
data/synthetic/   Generated CSV inputs (endpoints, network, queries, scenarios)
results/          Optimization outputs and plots
src/              Core library modules
scripts/          CLI entry points
```

## Generate Synthetic Data

```bash
python scripts/generate_synthetic_data.py
```

This writes CSV files under `data/synthetic/` with deterministic seed `42` by default (see `configs/default.yaml`). Cloud **pricing**, **data conversion ratios** (rho), and **region network** tables live in `configs/pricing.yaml` and `configs/region_network.yaml`; endpoint costs and scenario rho values are derived from them in `src/pricing.py`.
Generated synthetic CSVs, solver outputs under `results/`, and rendered figures under `fig/` are intentionally ignored by Git to keep the repository lightweight.

**Fixed experiment hyperparameters** (in `configs/default.yaml`):

| Symbol | Config key | Value |
|--------|------------|-------|
| — | `random_seed` | 42 |
| Q_train | `num_train_queries_per_workflow_quality` | 1000 |
| Q_test | `num_test_queries_per_workflow_quality` | 1000 |
| S | `num_scenarios_per_query` | 50 per train/test query |
| η | `eta` | 0.05 |
| K | `active_batch_fraction` with `top_k=0` | 5% of `|W^l|` |

Data generation, measurement latency scaling (`populate_from_measurements`), SAA-CVaR MILP, and post-hoc CVaR evaluation all read these values from config. SkyFlow optimizes on train queries and reports final metrics on fresh held-out test queries. It initializes the active scenario-path cut set as empty and iteratively adds the top 5% most violated critical-path cuts from `W^l`.

**Query workload** (in `configs/default.yaml` + `pricing.yaml` → `queries.csv`): 1000 train requests and 1000 fresh test requests per workflow-quality pair; video duration ~ Uniform(1 min, 60 min); `fps=30`. Only **documented** `(operation, provider, region[, model])` combinations appear in `endpoints.csv`.

## Validate Against Paper Model

Run automated checks that compare the implementation to the SkyFlow formulation (DAG, data propagation, cost/latency, McCormick, SAA-CVaR MILP, decomposition):

```bash
python scripts/validate_against_paper.py
```

## Run Simulations

### Single experiment (one workflow × one quality)

Each run loads **only** that workflow's queries and writes to `results/<workflow>_<quality>/`:

```bash
python scripts/run_simulation.py --workflow workflow1 --quality Q1 --method decomposition
python scripts/run_simulation.py --workflow workflow2 --quality Q3 --method decomposition
```

### Run all 12 experiments

```bash
python scripts/run_all.py
```

Runs **12 isolated SkyFlow jobs**: `workflow1`--`workflow4` × `Q1|Q2|Q3`. Solver method defaults to `default_solver_method` in `configs/default.yaml`. Aggregated metrics append to `results/metrics.csv`; each run's plan and plots live under `results/workflow1_Q1/decomposition/`, etc.

## Methods

| Method | Module | Description |
|--------|--------|-------------|
| `full_milp` | `src/milp_full.py` | Full SAA-CVaR MILP with latency excess constraints on all query–scenario pairs |
| `decomposition` | `src/milp_decomposition.py` | SkyFlow critical-path-aware scenario-path cut generation for the SAA-CVaR MILP |
| `single_cloud` | `src/baselines.py` | **SC:** one cloud provider; min cost subject to empirical SLA violation threshold |
| `logical_optimal` | `src/baselines.py` | **LO:** per-node argmax capability μ_k (ignores cross-node effects) |
| `greedy` | `src/baselines.py` | **Greedy:** topological DAG pass; per-node min expected execution/storage cost with latency tie-break |
| `dpgm` | `src/baselines.py` | **DPGM:** deterministic profile-guided MILP; minimum-slack profiled solution if hard profile constraints are infeasible |
| `mtgp` | `src/mtgp_baseline.py` | **MTGP:** multi-tree GP hyper-heuristic baseline with task/cloud/resource selection rules inspired by Sun et al., IEEE TSC 2024 |

```bash
# One baseline for one experiment
python scripts/run_simulation.py --workflow workflow1 --quality Q1 --method dpgm

# 12 experiments × 4 baselines = 48 runs
python scripts/run_baselines.py
```

Results for each method are under `results/<workflow>_<quality>/<method>/`.

## Outputs

Results are written to `results/`:

| File | Description |
|------|-------------|
| `selected_plan.json` | Latest deployment plan and summary metrics |
| `deployment_plan.csv` | Per-node endpoint assignment |
| `metrics.csv` | Aggregated optimization metrics (appended on `run_all`) |
| `convergence.csv` | Decomposition iteration history |
| `latency_distribution.png` | Latency histogram |
| `cost_latency_summary.png` | Cost vs latency scatter |
| `convergence.png` | Decomposition convergence (decomposition only) |

## Replacing Synthetic Data with Real Measurements

1. Keep the same CSV schemas under `data/synthetic/` (or point loaders to a new directory).
2. **endpoints.csv** — measured base latency, cost, storage rates per logical operation / provider / region / quality.
3. **network.csv** — bandwidth, RTT, egress cost between endpoint pairs.
4. **queries.csv** — real query workloads (video size, duration, SLA) with `split=train|test`.
5. **scenarios.csv** — measured data conversion ratios (`rho_*`) and stochastic multipliers.
6. Adjust `configs/default.yaml` for `eta`, scenario counts, and solver limits.

No code changes are required if column names match the generated files.

## Ablation Studies

Toggle components via `AblationConfig` in `src/schemas.py` (wired through `SolverConfig`):

- `enable_cvar`, `enable_network_latency`, `enable_network_cost`, `enable_storage_cost`
- `enable_client_upload_download`, `fixed_provider`, `fixed_region`

## Future Extensions

- **Baselines**: implement `BaselineSolver` in `src/baselines.py`
- **Additional workflows**: add YAML under `configs/` and register in `src/config.py`
- **Real cloud traces**: replace CSV inputs only

## License

Research prototype — not intended for production deployment.
