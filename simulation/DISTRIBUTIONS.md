# 仿真参数分布说明

本文档说明 `simulation/distribution.py` 中 **各类参数是如何设置与采样** 的。

核心设计原则：

- **分布参数随机采样一次，之后固定**（在全局随机种子不变时可复现）。
- **数值样本**（例如某条链路的 RTT/bandwidth，或一次采样得到的 conversion ratio）是否每次调用都会变化，取决于具体接口：有些接口返回“固定分布上的随机样本”，有些接口直接返回“固定矩阵/固定值”。

## 全局随机数与确定性

在 `simulation/distribution.py` 中设置了全局随机种子：

- `random.seed(42)`

本项目采用“**显式采样参数**”的方式来获得确定性：

- 你先调用 `sample_*_params()`（例如 `sample_ratio_params()` / `sample_comm_params()` / `sample_exec_time_params()`），**只运行一次**，拿到一组分布参数。
- 后续所有采样都基于这组参数进行，从而保证“**分布参数固定**（seed 不变时可复现）”。

如果你修改随机种子（或重启进程后重新采样参数），就会得到另一套不同但仍然“自洽”的参数集合。

## Workflow 类型

目前定义的 workflow 包括：

- `segment`
- `split`
- `caption`
- `query`

## Provider 与 Region

provider 名称统一规范化为 `p1..p5`。

### 云 provider（具备存储能力）

- `p1`, `p2`, `p3` 为云 provider，带 region。
- region 标签总数为 `r1..r6`。
- 各 provider 拥有的 region 如下：
  - `p1`: `r1,r2,r3,r4`
  - `p2`: `r1,r2,r5,r6`
  - `p3`: `r3,r4,r5,r6`

### LLM-only provider（无存储、无 region）

- `p4`, `p5` **没有 `region`**（`region=None`）

语言模型选项总数为 6 个：`m1..m6`：

- `p4`: `m1,m2,m3,m4`
- `p5`: `m3,m4,m5,m6`

## 节点构建（12 + 12 + 20 + 20）

节点由 `build_nodes(workflow)` 构建。

节点命名规则：

- 云节点：`{provider}_{region}_{workflow}`（例如 `p1_r1_segment`）
- LLM 节点：`{provider}_{model}_{workflow}`（例如 `p4_m3_caption`）

数量：

- `segment`: 12 nodes (cloud only)
- `split`: 12 nodes (cloud only)
- `caption`: 20 nodes (12 cloud + 8 LLM)
- `query`: 20 nodes (12 cloud + 8 LLM)

代码在 import 时会断言这些数量是否满足预期。

### 存储能力与存储价格

- 云节点：`can_store=True`，并且 `storage_price_usd_per_gb_month` 按如下方式采样：
  - `Uniform(0.015, 0.025)`（单位：USD per GB per month）
- LLM-only 节点：`can_store=False` 且 `storage_price_usd_per_gb_month=None`

存储价格的“固定方式”采用显式参数采样：

- 你在初始化时调用一次 `sample_storage_prices(workflow)`，得到 `dict[node_name -> price]`
- 后续构建节点时，通过 `build_nodes_with_params(..., storage_prices=...)` 传入并复用

只要你不重新调用 `sample_storage_prices`，**每个节点的存储价格就会保持固定**。

## 数据量（MB）采样

`get_data_size_mb(min_mb, max_mb)` 使用均匀分布采样数据量：

- `Uniform(min_mb, max_mb)`（单位：MB）

## Data conversion ratio（输出/输入）

`get_data_conversion_ratio(workflow)` 返回一个正数 ratio，来自 **对数正态分布（LogNormal）** 的采样。

### 为什么用 LogNormal

- ratio 必须 **> 0**
- LogNormal 很适合表达“乘法尺度”的波动（相对比例的随机性）

### 参数化方式

对每个 workflow，我们用两类参数配置其 ratio 分布：

- `mean`：ratio 的目标均值
- `sigma`：log 空间标准差；越小表示方差越小

从 `mean` 推导 LogNormal 的 \(\mu\)：

\[
E[X] = \exp(\mu + 0.5\sigma^2)\quad\Rightarrow\quad \mu=\ln(mean)-0.5\sigma^2
\]

### 参数如何设置（随机一次后固定）

每个 workflow 的 `(mean, sigma)` 会在“合理区间”内 **随机采样一次并固定**（通过 `sample_ratio_params()` 的返回值复用）。这些区间来自你的定性约束：

- `segment`：均值固定为 1，`sigma` **极小**（ratio ≈ 1，方差极小）
- `split`：均值固定为 1，`sigma` **极小**
- `caption`：video → text，均值在 **很小** 的范围内，`sigma` **更大**
- `query`：text → text，输入（query+caption）通常大于输出（answer），均值在 (0,1)，`sigma` 适中

每次调用 `get_data_conversion_ratio(workflow)` 都会从该“固定分布”再抽一个样本（分布固定、样本可变）。

## 节点执行时间（仅 segment/split）

对 `segment` 与 `split` 节点，`Node.exec_time_s` 是一个函数：

- 输入：`video_size_mb`
- 输出：执行时间（秒）

对 `caption` 与 `query` 节点：

- `exec_time_s=None`

### 执行时间模型

\[
T = T_{io} + T_{comp}
\]

1) IO time (linear in size):

\[
T_{io} = a\cdot video\_size\_{mb}
\]

其中 `a`（seconds per MB）在创建该函数时随机采样一次并固定。

2）计算时间（条件 Gamma）：

\[
T_{comp}\mid video\_size\_{mb} \sim \Gamma(k,\theta(video\_size\_{mb}))
\]

其中：

- `k` 随机采样一次并固定（shape）
- \(\theta(video\_size) = \theta_0 + \theta_1\cdot video\_size\_{mb}\)
- `theta0`、`theta1` 随机采样一次并固定

### 确定性行为

- **分布参数** \((a,k,\theta_0,\theta_1)\) 会按 workflow 固定（通过 `sample_exec_time_params()` 的返回值复用）。
- 每次调用 `exec_time_s(video_size_mb)` 会重新从该 Gamma 分布采样，因此是 **分布不变、样本会变**。

## 网络：RTT + bandwidth 矩阵（15 × 15）

通信端点由 `get_comm_endpoints()` 构建（共 15 个）：

1. `local`
2. `p1_r1,p1_r2,p1_r3,p1_r4` (4)
3. `p2_r1,p2_r2,p2_r5,p2_r6` (4)
4. `p3_r3,p3_r4,p3_r5,p3_r6` (4)
5. `p4`
6. `p5`

### 构建结果

`build_comm_matrix()` 返回一个 `CommMatrix`，包含：

- `endpoints`：长度为 15 的端点有序元组（索引含义由此确定）
- `rtt_ms[i][j]`：对称 RTT 矩阵（单位：ms）
- `bandwidth_mbps[i][j]`：对称带宽矩阵（单位：Mbps）
- `index_by_name`：端点名 → 索引 的映射

查询方式：

- 按索引：`matrix.link(0, 3)`
- 按名字：`matrix.link("p1_r1", "p2_r1")`

### 分布形式

- RTT：**LogNormal**（ms）
- 带宽：**Pareto**（Mbps），用 Python 的 `random.paretovariate(alpha)` 并用 `xm` 做缩放：
  - `bw = xm * paretovariate(alpha)` where the Pareto variate has support \([1,\infty)\).

### 约束如何编码

对每一对端点，会先分配到一个类别，并按优先级选择参数：

1. **同 region 标签**（cloud↔cloud）：RTT 最低  
2. **同云 provider**（p1/p2/p3 内部）：相比跨云 RTT 更低、带宽更高  
3. 涉及 **local**：通常 RTT 更高、带宽更低  
4. 涉及 **p4/p5**：通常 RTT 更高、带宽更低  
5. 其他情况：跨云 baseline

类别级参数会在合理的均匀分布区间内采样一次并固定（通过 `sample_comm_params()` 的返回值复用），然后每条链路再引入一个小的乘性抖动（jitter）增加差异性。

### 确定性行为

如果你只调用一次 `build_comm_matrix(comm_params)` 并保存返回的矩阵对象复用，那么 **15×15 矩阵的数值也会固定不变**（在 seed 不变且不重新生成 `comm_params` 时）。

如果你希望“分布参数固定，但每次重新采样一张矩阵”，可以复用同一份 `comm_params`，但每次重新调用 `build_comm_matrix(comm_params)` 生成新矩阵即可。

