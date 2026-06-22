# Current Budget W1-Q1/W2-Q1 Algorithm Comparison

Metrics are held-out evaluation metrics from selected_plan.json. SVR is the empirical SLA violation rate.

## workflow1-Q1

| Algorithm | Status | Init | Cost | Mean Lat | P95 Lat | P99 Lat | SVR | CVaR | Runtime(s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SkyFlow | INFEASIBLE | qbr | 1.8014 | 4623.32 | 9024.67 | 11508.60 | 3.196% | 741.04 | 117.32 |
| SC | optimal |  | 2.3915 | 4893.41 | 9508.60 | 11981.43 | 4.262% | 943.00 | 5.83 |
| Greedy | optimal |  | 1.8014 | 4897.58 | 9801.74 | 12972.44 | 6.354% | 1648.23 | 16.88 |
| DPGM | optimal |  | 1.8014 | 4897.58 | 9801.74 | 12972.44 | 6.354% | 1648.23 | 29.73 |
| MTGP | optimal |  | 1.8015 | 4604.54 | 8990.69 | 11476.71 | 3.148% | 728.68 | 18.37 |

SkyFlow relative to baselines:

| Baseline | Cost Reduction | SVR Delta | P95 Delta |
|---|---:|---:|---:|
| SC | +24.67% | -1.066 pp | -5.09% |
| Greedy | -0.00% | -3.158 pp | -7.93% |
| DPGM | -0.00% | -3.158 pp | -7.93% |
| MTGP | +0.00% | +0.048 pp | +0.38% |

## workflow2-Q1

| Algorithm | Status | Init | Cost | Mean Lat | P95 Lat | P99 Lat | SVR | CVaR | Runtime(s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SkyFlow | INFEASIBLE | qbq | 3.4000 | 4892.08 | 9850.87 | 12967.02 | 6.478% | 1604.36 | 499.57 |
| SC | optimal |  | 8.7778 | 4886.78 | 9600.10 | 11986.81 | 4.434% | 899.02 | 13.17 |
| Greedy | optimal |  | 3.4000 | 4892.32 | 9851.11 | 12967.26 | 6.478% | 1604.59 | 51.83 |
| DPGM | optimal |  | 3.4000 | 4895.20 | 9856.36 | 12973.29 | 6.498% | 1607.94 | 107.38 |
| MTGP | optimal |  | 3.4001 | 4599.06 | 9053.72 | 11474.19 | 3.276% | 678.90 | 48.42 |

SkyFlow relative to baselines:

| Baseline | Cost Reduction | SVR Delta | P95 Delta |
|---|---:|---:|---:|
| SC | +61.27% | +2.044 pp | +2.61% |
| Greedy | -0.00% | +0.000 pp | -0.00% |
| DPGM | -0.00% | -0.020 pp | -0.06% |
| MTGP | +0.00% | +3.202 pp | +8.80% |
