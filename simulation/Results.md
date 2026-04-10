# 实验结果（当前配置）

日期：2026-04-09  
系统：Linux + bash  
目录：`/home/heng/Documents/skybridge/simulation`

## 运行命令

```bash
python simulation.py --queries 8 --runs 5 --train-runs 1 --no-progress
```

## 输出结果

```text
== Proposed selection on train ==
baseline               mean_U    mean_cost  mean_lat(s)   viol_c   viol_t      n
--------------------------------------------------------------------------------
proposed(train)        0.9088       0.8209       599.80    0.000    0.000      4

== Test comparison ==
baseline               mean_U    mean_cost  mean_lat(s)   viol_c   viol_t      n
--------------------------------------------------------------------------------
single_cloud           0.9077       1.4350      1204.01    0.000    0.000     20
logical_optimal        0.9088       1.5771      1248.05    0.000    0.000     20
greedy                 0.8980       0.9464      1521.31    0.000    0.000     20
deterministic          0.9088       1.5624      1349.11    0.000    0.000      4
proposed               0.9088       1.5718      1359.29    0.000    0.000     20
```

## 简要观察

- `proposed` 与 `logical_optimal` 的平均 utility 基本一致（`0.9088`）。
- `greedy` 成本更低（`0.9464`），但延迟显著更高（`1521.31s`）。
- 本次所有方法在该样本规模下 `cost/latency violation rate` 均为 `0`。
- 由于全量候选搜索计算量较大，本次采用了轻量参数（`queries=8, runs=5, train-runs=1`）。

