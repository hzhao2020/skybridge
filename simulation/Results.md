# 实验结果（重跑版，中文）

日期：2026-04-08  
系统：Linux + bash  
目录：`/home/heng/Documents/simulation`

本次结果基于最新代码重跑，特别注意：
- `simulation.py` 中已去掉 top-k 截断，使用**全量候选组合**搜索；
- `baselines.py` 已按 `config.yaml` 中 `utility_weight` 做加权效用口径；
- `algo.py` 为严格 black-box（仅通过 `Workflow.calculate()` 采样拟合系数）。

---

## 1）`simulation.py`（全量候选）重跑结果

命令：

```bash
python simulation.py --queries 4 --train-ratio 0.5 --runs 2 --train-runs 1 --eta-cost 0.1 --eta-lat 0.1 --penalty 1.0
```

输出：

```
== Proposed selection on train ==
baseline               mean_U    mean_cost  mean_lat(s)   viol_c   viol_t      n
--------------------------------------------------------------------------------
proposed(train)        0.8917       2.2730      1779.82    0.000    0.000      2

== Test comparison ==
baseline               mean_U    mean_cost  mean_lat(s)   viol_c   viol_t      n
--------------------------------------------------------------------------------
single_cloud           0.8587       2.2308      2452.65    0.250    0.000      4
logical_optimal        0.8917       1.4629      1982.04    0.250    0.000      4
greedy                 0.8820       3.0186      2879.96    0.750    0.000      4
deterministic          0.8917       2.3153      2614.36    0.000    0.000      2
proposed               0.8917       1.1130      3237.60    0.250    0.000      4
```

说明：
- 由于全量候选组合非常大，这里为了保证可完成，使用了较小样本规模（`queries=4, runs=2, train-runs=1`）。
- 在这组参数下，`proposed` 成本最低，但延迟均值较高。

---

## 2）`baselines.py` 重跑结果

命令：

```bash
python baselines.py --queries 10 --runs 10
```

输出：

```
baseline               mean_U    mean_cost  mean_lat(s)   viol_c   viol_t      n
--------------------------------------------------------------------------------
single_cloud           0.8587       2.0229      2626.38    0.080    0.060    100
logical_optimal        0.8917       2.5543      3045.57    0.240    0.050    100
greedy                 0.8820       2.7125      3001.05    0.300    0.050    100
deterministic          0.8917       2.4195      2731.70    0.000    0.000     10
```

---

## 3）`algo.py`（black-box MILP）重跑结果

命令：

```bash
python algo.py
```

输出：

```
Status: Optimal
Objective U(x): 0.8917216675043008
  node 0 (segment): p2_r1_segment (mu=0.8676538703114536)
  node 1 (split): p2_r6_split (mu=1.0)
  node 2 (caption): p2_r6_caption (mu=0.8919347291938112)
  node 3 (query): p4_m3_query (mu=0.8968570051909786)
```

---

## 4）结论（本次重跑）

1. 三个主脚本均可正常运行并产出结果。  
2. `simulation.py` 在全量候选下可运行，但计算量显著增大，需要小样本参数才能较快完成。  
3. `baselines.py` 的效用口径已与配置权重一致。  
4. `algo.py` 在当前预算配置下仍可得到 `Optimal` 解。  

