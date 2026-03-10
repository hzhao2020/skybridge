## Profile 子系统说明

本目录包含 SkyBridge 项目中与 **性能评测（profiling）** 相关的脚本与配置，主要用于评估：

- **边缘存储节点之间的网络性能**（RTT、带宽）
- **各 Operation 节点的端到端计算延迟**
- **后续结合价格表 / 精度表进行性价比分析**

当前目录结构（仅列出与 profile 相关的核心文件）：

- `test_bucket_transmission.py`：存储桶传输性能单次测试工具
- `edge_latency.py`：基于存储桶传输测试的长时间边缘网络监控
- `node_latency.py`：对注册的 Operation 进行端到端计算延迟测试
- `profile_table.json`：价格与（预留）精度配置表

---

## 一、存储桶传输性能测试（test_bucket_transmission.py）

### 1. 功能概述

`test_bucket_transmission.py` 用于测试不同云厂商、不同 Region 之间 **对象存储 bucket 的传输性能**，包括：

- **RTT（Round-Trip Time）**：这里指一次对象从源 bucket 复制到目标 bucket 的平均耗时（注意并非传统意义上的“网络往返时延”）。
- **带宽（Bandwidth）**：对不同大小文件进行传输测试，估算跨 bucket 复制时的有效带宽（Mbps）。

目前内置的 bucket 配置包括：

- GCP：`gcp_us`、`gcp_tw`、`gcp_sg`
- AWS：`aws_us`、`aws_sg`

### 2. 实现细节

- 测试底层依赖 `ops.utils.DataTransmission`，通过其：
  - `upload_local_to_cloud`：将本地测试文件上传到指定 bucket
  - `smart_move`：在不同 bucket（可跨云）之间移动对象
- 文件生成：
  - 通过 `_create_test_file(size_bytes)` 使用 `os.urandom` 生成指定大小的随机二进制数据，避免被压缩等优化影响测试结果。
- RTT 测试：
  - 先生成一个 **1KB** 文件并上传到源 bucket。
  - 然后多次（默认 3 次）调用 `smart_move` 将对象从源 bucket 复制到目标 bucket，记录每次复制的耗时。
  - 取平均值并转换为毫秒视为该 bucket 对的“RTT”。
- 带宽测试：
  - 对一组文件大小（默认 `[1MB, 10MB, 100MB]`）逐个测试。
  - 每个大小的流程为：本地生成文件 → 上传至源 bucket → `smart_move` 复制到目标 bucket。
  - 带宽计算仅基于“复制阶段”的耗时：`bandwidth_mbps = size_bytes * 8 / (transfer_time * 1_000_000)`。
- 结果数据结构：
  - 使用 `TransmissionTestResult` dataclass 记录每一次测试的详细结果，便于序列化为 JSON。
  - `run_all_tests` 支持：
    - 自动遍历所有 bucket 对（排除 self-loop）
    - 或通过 `--pairs src1:dst1 src2:dst2` 只测试特定组合
    - 可选只测 RTT 或只测带宽。

### 3. 使用方式

- 完整测试所有 bucket 对的 RTT + 带宽：

```bash
python -m profile.test_bucket_transmission
```

- 仅测试 RTT：

```bash
python -m profile.test_bucket_transmission --rtt-only
```

- 仅测试带宽，且只针对部分 bucket 对：

```bash
python -m profile.test_bucket_transmission \
  --bandwidth-only \
  --pairs gcp_us:aws_us gcp_tw:gcp_sg
```

### 4. 已知缺陷 / 局限

- **RTT 定义不完全准确**：当前实现中的 RTT 是“一次跨 bucket 复制的平均耗时”，并非传统 ICMP 网络 RTT，需要在对外展示时明确说明。
- **bucket 配置写死在代码中**：新增 / 修改 bucket 需改代码并重新部署，缺乏统一配置中心。
- **依赖 DataTransmission 内部实现**：例如 `_parse_uri`、`gcs_client`、`s3_client` 等私有属性，一旦实现变更可能导致该脚本失效。
- **成本与安全性**：大文件、多 bucket 对的带宽测试可能带来显著的云费用，当前没有速率限制和预算保护机制。

---

## 二、边缘网络长期监控（edge_latency.py）

### 1. 功能概述

`edge_latency.py` 在 `test_bucket_transmission.py` 的基础上，提供 **多天级别的定时监控**，周期性地在指定 bucket 对之间执行 RTT / 带宽测试，并将每一轮测试结果落盘。

主要用途：

- 观察跨云 / 跨 Region 网络质量的 **长时间趋势和波动**；
- 为运维策略（如多云路由选择、容灾切换）提供数据支撑。

### 2. 实现细节

- 动态加载底层测试模块：
  - 为避免与标准库 `profile` 模块冲突，通过 `_load_bucket_tester_module()` 使用文件路径动态加载 `test_bucket_transmission.py`。
- 核心类 `EdgeLatencyMonitor`：
  - 关键参数：
    - `duration_days`：总测试时长（天），默认 `7.0`。
    - `interval_seconds`：默认测试间隔（秒），未单独指定 RTT / 带宽时使用。
    - `rtt_interval_seconds` / `bandwidth_interval_seconds`：可分别指定 RTT / 带宽测试的独立间隔。
    - `test_rtt` / `test_bandwidth`：分别控制是否启用对应测试。
    - `bucket_pairs`：待测的 bucket 对列表，默认为底层 `BucketTransmissionTester` 的全部组合。
    - `results_dir`：结果输出目录，默认 `results/edge_latency`。
  - 时间调度：
    - 主循环以 `tick_interval = min(rtt_interval_seconds, bandwidth_interval_seconds)` 为周期 tick。
    - 使用 `last_rtt_time` / `last_bandwidth_time` 控制在每个 tick 是否需要执行 RTT / 带宽测试。
  - 每轮测试 `_run_single_round`：
    - 实例化新的 `BucketTransmissionTester`。
    - 调用 `run_all_tests` 执行当前轮应有的 RTT / 带宽测试。
    - 汇总结果（起止时间、耗时、test flags、bucket 对、底层结果列表），写入单独 JSON 文件。
- 结果存储：
  - 每轮生成一个文件，命名形如：
    - `edge_latency_round0001_20250101_120000.json`
  - 存放路径默认为 `results/edge_latency/`。

### 3. 使用方式

- 典型用法：同时监控 RTT（10 分钟一次）和带宽（1 小时一次），持续 7 天：

```bash
python -m profile.edge_latency \
  --duration-days 7 \
  --interval-seconds 600 \
  --rtt-interval 600 \
  --bandwidth-interval 3600
```

- 只监控 RTT：

```bash
python -m profile.edge_latency --duration-days 3 --rtt-only --interval-seconds 300
```

- 只对部分 bucket 对进行长期监控：

```bash
python -m profile.edge_latency \
  --duration-days 1 \
  --interval-seconds 600 \
  --pairs gcp_us:aws_us gcp_tw:gcp_sg
```

### 4. 已知缺陷 / 局限

- **无断点恢复能力**：进程中断后无法自动续跑，只能重新启动新的监控任务。
- **异常处理有限**：对 `KeyboardInterrupt` 有优雅退出，对其他异常可能直接中断整个监控流程。
- **资源与费用控制缺失**：未内置请求频率 / 每日配额 / 云费用估计等保护机制，长时间运行可能带来较高成本。
- **调度精度有限**：采用简单的 `time.sleep` 加 wall-clock 逻辑，tick 时间可能存在秒级误差，但一般对 profile 场景可接受。

---

## 三、Operation 端到端延迟测试（node_latency.py）

### 1. 功能概述

`node_latency.py` 面向 **所有已经在 `ops.registry.REGISTRY` 中注册的 Operation**，以三个视频问答相关数据集的 train 集合作为输入，对下列类型的 Operation 进行端到端延迟评估：

- `segment`：视频分段（`VideoSegmenter`）
- `split`：视频裁剪 / 切片（`VideoSplitter`）
- `caption`：视频视觉描述生成（`VisualCaptioner`）
- `llm`：大模型问答（`LLMQuery`）

输出结果包括：

- 每一次 Operation 调用的原始延迟记录（`OperationLatencyRecord` 列表）
- 按 Operation 聚合的统计指标（平均 / 最小 / 最大延迟）及按数据集拆分的统计

### 2. 实现细节

- 数据集加载：
  - 使用 `utils.dataset.build_dataset(name, "train", datasets_root)` 预加载：
    - `EgoSchema`
    - `NExTQA`
    - `ActivityNetQA`
  - `max_samples_per_dataset`：
    - `None` 或 `<=0` 表示“使用该数据集所有训练样本”。
    - 否则对每个数据集仅使用前 `max_samples_per_dataset` 条样本。
- Operation 分类：
  - `_categorize_operation` 依赖 `isinstance` 判断 Operation 类型：
    - `VideoSegmenter` → `segment`
    - `VideoSplitter` → `split`
    - `VisualCaptioner` → `caption`
    - `LLMQuery` → `llm`
  - CLI 可通过 `--categories` 选择只测部分类别。
- 测试逻辑：
  - 对于每个 Operation（按 `pid` 唯一标识），遍历所选数据集的样本，并根据其类别调用对应的 `_test_*_op` 方法：
    - `segment`：
      - 从样本中取 `video_path`，缺失则跳过。
      - 统一将输出写至 `videos/node_latency/{dataset_name_lower}`。
      - 调用 `op.execute(video_path, target_path=target_path)` 并计时。
    - `split`：
      - 同样使用 `video_path`，默认固定一个 `[0.0, 5.0]` 的短片段列表作为切分区间。
      - 调用 `op.execute(video_path, segments=segments, target_path=target_path)` 并计时。
    - `caption`：
      - 传入 `video_path` + `target_path`，调用 `op.execute(video_path, target_path=target_path)` 并计时。
    - `llm`：
      - 从样本中取 `question` 作为 prompt，缺失则该样本跳过。
      - 对 OpenAI 实现使用 `max_tokens=256`，其他实现使用 `max_output_tokens=256` 以控制输出长度。
      - 调用 `op.execute(question, **llm_kwargs)` 并计时。
- 结果与统计：
  - 每条记录使用 `OperationLatencyRecord` 表达，包含：
    - 标识信息：`pid`、`category`、`provider`、`region`、`model_name`
    - 数据集信息：`dataset`、`sample_index`、`qid`、`video_name`
    - 性能信息：`latency_seconds`、`success`、`error_message`
  - `_build_summary` 聚合逻辑：
    - 按 `pid` 聚合，统计：
      - 全局 `total_calls` / `success_calls`
      - 全局延迟统计：count / mean / min / max
      - 按数据集维度的延迟统计
    - 为控制 JSON 体积，会在聚合后移除原始的 `latencies` 列表，仅保留聚合结果。
- 结果保存：
  - `save_results` 将结果写入 `results/node_latency/` 目录：
    - 若未指定 `filename`，默认命名为 `node_latency_YYYYMMDD_HHMMSS.json`。
    - 文件中包含：
      - 全局元信息（数据集根路径、数据集列表、样本数配置、测试类别）
      - 按 Operation 聚合的 `summary`
      - 所有原始 `records`。

### 3. 使用方式

- 全量测试所有 Operation、所有类别、使用全部训练集样本（**成本较高，不建议默认运行**）：

```bash
python -m profile.node_latency
```

- 仅测试部分类别，限制每个数据集的样本数，例如只测 split + caption，每个数据集取前 100 条样本：

```bash
python -m profile.node_latency \
  --categories split caption \
  --max-samples-per-dataset 100
```

- 指定数据集根目录，并自定义输出文件名：

```bash
python -m profile.node_latency \
  --datasets-root /data/qa_datasets \
  --max-samples-per-dataset 50 \
  --output node_latency_ablation.json
```

### 4. 已知缺陷 / 局限

- **数据规模与成本**：
  - 当 `max_samples_per_dataset <= 0` 时，会遍历三个数据集的全部 train 集合，对计算/云资源 / LLM token 消耗都可能非常大；实际使用中应显式设置较小的样本上限。
- **云端中间产物未自动清理**：
  - `segment` / `split` / `caption` 产生的中间结果统一写入 `videos/node_latency/...`，当前脚本不会自动清理，需由运维或后续工具清理，以避免长期堆积。
- **Operation 接口契约依赖强**：
  - 默认假设所有 Operation 的 `execute` 函数签名和线上使用一致，但某些复杂 pipeline（例如需要额外上下文或链式调用）可能无法完全通过单次 `execute` 调用模拟。
- **LLM 延迟视角有限**：
  - 当前仅测量单次问答调用的模型延迟，没有覆盖检索、重排序、多轮对话等真实业务逻辑中的额外延迟。

---

## 四、价格与精度表（profile_table.json）

### 1. 功能概述

`profile_table.json` 用于维护与 Operation 相关的 **价格配置**（以及预留的精度指标），为后续**性价比分析**提供数据来源。

当前结构主要包括：

- `price.billing_categories.per_minute.operations`：按分钟计费的 Operation（例如 `seg_*`）
- `price.billing_categories.per_token.operations`：按 token 计费的 caption / llm Operation
- `price.billing_categories.per_gb_storage.operations`：存储费用（不同云、不同 Region）
- `price.billing_categories.per_gb_egress.operations`：出网费用（跨 Region / 跨云）
- `accuracy`：目前为空 `{}`，预留用于记录各 Operation / 模型在标准数据集上的精度指标。

### 2. 与 profile 结果的结合

通过将 `profile_table.json` 中的价格配置与：

- `edge_latency` 生成的长期 bucket 传输性能结果，
- `node_latency` 生成的各 Operation 延迟结果，

结合起来，可以进一步计算：

- 不同云 / 不同 Region 的 **单位时间 / 单位数据传输成本**；
- 不同 Operation 的 **吞吐量（QPS）与单位请求成本**；
- 在给定预算与时延约束下的 **最优部署 / 调用策略**。

当前代码中尚未直接消费 `profile_table.json`，主要作为离线分析和后续优化的配置基础。

### 3. 已知缺陷 / 局限

- **accuracy 信息缺失**：目前精度部分为空，无法直接进行完整的“速度 × 价格 × 精度”的三维权衡分析。
- **与 REGISTRY 未自动联动**：新增 Operation 后需要手动在 `profile_table.json` 中维护对应价格条目，缺乏自动校验和同步机制。

---

## 五、后续改进方向建议

- **配置解耦**：
  - 将 bucket / Operation 的 profile 目标（例如测试哪些 bucket 对、哪些 pid、样本数等）从代码中抽离到独立配置文件，或与 `ops.registry` 做更紧密的绑定。
- **成本与安全控制**：
  - 在长时间测试脚本中增加速率限制、预算上限、test-mode 等机制，避免在生产环境误触造成高额费用。
- **结果分析与可视化**：
  - 基于 `results/edge_latency/` 与 `results/node_latency/` 的 JSON 结果，增加 Jupyter Notebook / 可视化脚本，统一产出报表与图表。
- **精度指标填充**：
  - 在 `accuracy` 字段填入各 Operation / 模型在标准数据集上的评测结果，形成完整的“性能画像”。

