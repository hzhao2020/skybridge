# Simulation Random Variables & APIs

本文件说明 `simulation/distribution.py` 中所有随机变量的**分布假设**、**函数接口**、**参数对象**、**单位**与**示例用法**。

> 约定：除非特别说明，所有采样函数都支持可选的 `rng: np.random.Generator` 参数，用于复现实验。

---

## 1) RTT（Round-Trip Time）

### 分布

- **对数正态分布**  
  \[
  RTT \sim \mathrm{LogNormal}(\mu, \sigma)
  \]

### Region/Node 级参数生成（当前实现）

`simulation/distribution.py` 现在支持按 **src/dst region（node）对**生成 RTT 分布参数：

- **节点集合**：`NODES = ["local", "gcp_us", "gcp_tw", "gcp_sg", "aws_us", "aws_sg", "aliyun_us", "aliyun_se"]`
- **均值表（ms）**：`RTT_MEAN_MS[(src, dst)] = mean_ms`
- **固定离散度**：`RTT_SIGMA = 0.25`
- **对称查表**：如果 `(src, dst)` 不在表里，会尝试 `(dst, src)`；都没有则抛错
- **同节点**：`src == dst` 时，使用 `mean = 2ms`（近似同机房/同 region）
- **由均值反推 lognormal 参数**：
  \[
  \mu = \ln(\text{mean}) - \frac{1}{2}\sigma^2
  \]

### 参数对象：`RttLognormalParams`

- **字段**
  - `mu: float`：对数空间均值（注意不是 RTT 的均值）
  - `sigma: float`：对数空间标准差
- **单位**
  - RTT 输出单位为 **毫秒 (ms)**；`mu/sigma` 是对数正态的参数（无单位，但与输出尺度一致）

### 采样接口：`sample_rtt_ms(...) -> float`

- **签名**
  - `sample_rtt_ms(params: Optional[RttLognormalParams] = None, rng: Optional[np.random.Generator] = None) -> float`
- **返回值**
  - `rtt_ms: float`（ms），最小被截断为 `>= 0.1ms`
- **示例**

```python
from simulation.distribution import RttLognormalParams, sample_rtt_ms

rtt_params = RttLognormalParams(mu=3.2, sigma=0.35)
rtt_ms = sample_rtt_ms(rtt_params)
```

### 推荐接口：`get_rtt_params(src, dst) -> RttLognormalParams`

```python
from simulation.distribution import get_rtt_params, sample_rtt_ms

rtt_ms = sample_rtt_ms(get_rtt_params("local", "gcp_tw"))
```

---

## 2) Bandwidth（链路带宽）

### 分布

- **帕累托分布（numpy 形式）**
  - 代码使用 `numpy.pareto(alpha)` 生成 \(Y \ge 0\)
  - 带宽定义为：
    \[
    BW = x_m \cdot (1 + Y), \quad BW \ge x_m
    \]

### 参数对象：`BandwidthParetoParams`

- **字段**
  - `xm_mbps: float`：尺度参数（最小带宽），单位 Mbps
  - `alpha: float`：形状参数（越大尾部越短、波动越小）

### 采样接口：`sample_bandwidth_mbps(...) -> float`

- **签名**
  - `sample_bandwidth_mbps(params: Optional[BandwidthParetoParams] = None, rng: Optional[np.random.Generator] = None) -> float`
- **返回值**
  - `bandwidth_mbps: float`（Mbps），最小被截断为 `>= 1.0Mbps`
- **示例**

```python
from simulation.distribution import BandwidthParetoParams, sample_bandwidth_mbps

bw_params = BandwidthParetoParams(xm_mbps=200, alpha=2.2)
bw_mbps = sample_bandwidth_mbps(bw_params)
```

### Region/Node 级参数生成（当前实现）

`simulation/distribution.py` 现在支持按 **src/dst region（node）对**生成带宽分布参数：

- **最小带宽表（Mbps）**：`BW_XM_MBPS[(src, dst)] = xm_mbps`
- **固定形状参数**：`BW_ALPHA = 2.5`
- **对称查表**：如果 `(src, dst)` 不在表里，会尝试 `(dst, src)`；都没有则抛错
- **同节点**：`src == dst` 时，使用 `xm = 2000Mbps`
- **采样方式**（与 numpy 一致）：
  - `Y = numpy.pareto(alpha)`, \(Y \ge 0\)
  - `BW = xm * (1 + Y)`, \(BW \ge xm\)

### 推荐接口：`get_bw_params(src, dst) -> BandwidthParetoParams`

```python
from simulation.distribution import get_bw_params, sample_bandwidth_mbps

bw_mbps = sample_bandwidth_mbps(get_bw_params("local", "gcp_tw"))
```

---

## 3) Link（RTT + Bandwidth 联合采样）

### 分布

- RTT 与带宽分别按各自分布采样（当前实现**未显式建相关性**）。

### 接口：`sample_link(...) -> tuple[float, float]`

- **签名**
  - `sample_link(rtt_params: Optional[RttLognormalParams] = None, bw_params: Optional[BandwidthParetoParams] = None, rng: Optional[np.random.Generator] = None) -> tuple[float, float]`
- **返回值**
  - `(rtt_ms, bandwidth_mbps)`

### 示例

```python
from simulation.distribution import sample_link, RttLognormalParams, BandwidthParetoParams

rtt_params = RttLognormalParams(mu=3.0, sigma=0.4)
bw_params = BandwidthParetoParams(xm_mbps=100, alpha=2.5)

rtt_ms, bw_mbps = sample_link(rtt_params=rtt_params, bw_params=bw_params)
```

### 推荐接口：`sample_link_between(src, dst, rng=None) -> (rtt_ms, bw_mbps)`

当你希望“按 region 对”直接采样链路指标时，推荐用这个接口（内部会调用 `get_rtt_params/get_bw_params` 再采样）：

```python
import numpy as np
from simulation.distribution import sample_link_between

rng = np.random.default_rng(123)
rtt_ms, bw_mbps = sample_link_between("local", "gcp_tw", rng=rng)
```

---

## 4) Data Conversion Ratio（输出数据大小 / 输入数据大小）

### 分布

- **对数正态分布**
  \[
  ratio \sim \mathrm{LogNormal}(\mu_{op}, \sigma_{op})
  \]

### 参数对象：`OperationSizeLognormalParams`

- **字段（默认值概览）**
  - `sigma_small`：小方差等级（用于 segment/split）
  - `sigma_medium`：中等方差等级（用于 caption/query）
  - `mean_segment`：segment 的目标均值（默认 1.0）
  - `mean_split`：split 的目标均值（默认 1.0）
  - `mean_caption`：caption 的目标均值（默认 0.1）
  - `mean_query`：query 的目标均值（默认 1.0）
- **说明**
  - 文件内通过 \(\mu = \ln(\text{mean}) - \frac{1}{2}\sigma^2\) 来近似保证 \(E[ratio]\approx \text{mean}\)。

### 接口：`sample_operation_size_ratio(operation, ...) -> float`

- **签名**
  - `sample_operation_size_ratio(operation: str, params: Optional[OperationSizeLognormalParams] = None, rng: Optional[np.random.Generator] = None) -> float`
- **支持的 operation 名称（大小写不敏感）**
  - `"video_segment"` / `"segment"`
  - `"video_split"` / `"split"`
  - `"caption"`
  - `"llm_query"` / `"query"`
- **示例**

```python
from simulation.distribution import sample_operation_size_ratio

ratio = sample_operation_size_ratio("caption")
```

---

## 5) LLM Tokens（与输入 videos 相关的 token 数）

### 分布

- **对数正态分布 + 取整**
  - 先构造一个输入规模标量 `total_units`
  - \[
    X \sim \mathrm{LogNormal}(\mu(total\_units), \sigma)
    \]
  - 返回 `round(X)` 并截断到 `>= 1`

### 参数对象：`LlmTokenLognormalParams`

- **字段**
  - `base_mu`：当视频规模接近 0 时的 \(\mu\)
  - `mu_per_unit`：每单位输入规模对 \(\mu\) 的线性增量
  - `sigma`：对数空间标准差

### 接口：`sample_llm_tokens_from_videos(videos, ...) -> int`

- **签名**
  - `sample_llm_tokens_from_videos(videos, *, params: Optional[LlmTokenLognormalParams] = None, rng: Optional[np.random.Generator] = None, unit_extractor=None) -> int`
- **`videos` 的推荐输入形式**
  - 直接传一个标量（比如“总视频分钟数/总大小MB”）
  - 传一个 list（每个元素是该视频的标量规模）
  - 传自定义对象 list，并传 `unit_extractor=lambda v: v.xxx`
- **示例**

```python
from simulation.distribution import sample_llm_tokens_from_videos

tokens = sample_llm_tokens_from_videos([2.0, 1.5, 3.0])  # 例如单位=分钟
```

---

## 6) Video Size（视频大小）

### 分布

- **均匀分布**
  \[
  Size \sim \mathrm{Uniform}(\text{min\_mb}, \text{max\_mb})
  \]

### 接口：`sample_video_size_mb(...) -> float`

- **签名**
  - `sample_video_size_mb(*, min_mb: float = 5.0, max_mb: float = 500.0, rng: Optional[np.random.Generator] = None) -> float`
- **单位**
  - 返回单位为 **MB**
- **示例**

```python
from simulation.distribution import sample_video_size_mb

size_mb = sample_video_size_mb()  # 默认 5~500MB
```

---

## 7) Execution Time（节点执行时间，给定输入大小的条件分布）

### 条件分布

- **Gamma 条件分布**
  \[
  T \mid S \sim \mathrm{Gamma}(k, \theta(S))
  \]
  \[
  E[T\mid S] = k\theta(S) = \text{base\_time} \cdot f(S)
  \]

### 参数对象：`OperationGammaTimeParams`

- **字段**
  - `shape: float`：Gamma 形状参数 \(k\)
  - `base_time_per_mb: float`：基础系数（用于把视频大小映射到均值），单位秒/MB（当 `complexity='linear'` 时更直观）
  - `complexity: str`：确定性复杂度形式
    - `"linear"`：\(f(S)=S\)
    - `"s_log_s"`：\(f(S)=S\ln(1+S)\)

### 默认参数表：`DEFAULT_EXECUTION_TIME_PARAMS`

当前文件内提供了 **Video Segmentation** / **Video Splitting** 的默认 operation 参数（可后续按 profile 数据替换）：

- Segmentation：`seg_aws_us`, `seg_aws_sg`, `seg_google_us`, `seg_google_tw`
- Splitting：`split_aws_us`, `split_aws_sg`, `split_google_us`, `split_google_sg`

### 接口：`sample_execution_time_seconds(task_name, operation_name, video_size_mb, ...) -> float`

- **签名**
  - `sample_execution_time_seconds(task_name: str, operation_name: str, video_size_mb: float, *, params_overrides: Optional[dict[str, OperationGammaTimeParams]] = None, rng: Optional[np.random.Generator] = None) -> float`
- **输入**
  - `task_name`：任务名称（当前实现不参与计算，仅保留语义位）
  - `operation_name`：operation id（必须在默认参数表里，或在 `params_overrides` 覆盖表里）
  - `video_size_mb`：输入规模 \(S\)（MB）
- **输出**
  - `time_seconds: float`（秒）
- **覆盖默认参数**

```python
from simulation.distribution import OperationGammaTimeParams, sample_execution_time_seconds

overrides = {
    "seg_aws_us": OperationGammaTimeParams(shape=2.5, base_time_per_mb=0.03, complexity="s_log_s")
}

t = sample_execution_time_seconds(
    task_name="Video Segmentation",
    operation_name="seg_aws_us",
    video_size_mb=200.0,
    params_overrides=overrides,
)
```

---

## Notes

- **单位一致性**：`video_size_mb` 作为执行时间模型的输入规模 \(S\)，当前默认单位为 **MB**。
- **可重复性**：若需可重复采样，请在外部创建并传入同一个 `np.random.default_rng(seed)`。

