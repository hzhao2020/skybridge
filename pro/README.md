# Skybridge Prototype

这个目录实现了一个“用户侧作为中转站”的跨云 video QA workflow prototype：

```text
video(user)
  -> shot detection       cloud service: AWS / GCP / Aliyun
  -> split & sample       serverless function: AWS / GCP / Aliyun
  -> frame caption        LLM node: GCP / Anthropic API
  -> Q/A                  LLM node: GCP / Anthropic API
  -> answer(user)
```

关键点是：调度不放在任意云厂商内部，而是由本地 `UserSideBroker` 执行。每个节点的输出都会先回到 `UserRelay` 落盘，再由 `UserRelay` 发给下一个节点；因此不会出现 AWS 直接把 provider-owned object 传给 GCP 或 Anthropic 的路径。

## Run the mock workflow

```bash
PYTHONPATH=src python -m skybridge_prototype.cli \
  --config configs/prototype.mock.json \
  --video samples/demo_video.txt \
  --question "What is happening in the video?"
```

运行后会生成 `runs/<run_id>/`：

- `artifacts/input_video.*`
- `artifacts/shots.json`
- `artifacts/frame-*.txt`
- `artifacts/captions.json`
- `artifacts/answer.json`
- `result.json`

`result.json` 里的 `transfers` 字段就是用户侧中转日志，包含每个 node、provider、region、方向和传输字节数。

## Provider configuration

`configs/prototype.mock.json` 默认跑通端到端：

- `shot_detection`: `aws`
- `split_sample`: `gcp`
- `caption`: `gcp`
- `qa`: `anthropic`

可以用 CLI 覆盖：

```bash
PYTHONPATH=src python -m skybridge_prototype.cli \
  --config configs/prototype.mock.json \
  --video samples/demo_video.txt \
  --question "What is happening?" \
  --shot-provider aliyun \
  --split-provider aws \
  --caption-provider anthropic \
  --qa-provider gcp
```

`configs/prototype.cloud.example.json` 展示了真实云接入方式：

- AWS/GCP/Aliyun 的 shot detection 和 split/sample 使用 `http_json` adapter，对接 API Gateway、Cloud Run、Cloud Functions、Aliyun FC 等 HTTP endpoint。
- Anthropic 使用 `anthropic` adapter，需要设置 `ANTHROPIC_API_KEY` 和 `ANTHROPIC_MODEL`，也可以在配置里直接写 `model`。
- GCP LLM 节点可以先包一层 Cloud Run / Function，暴露为 `http_json` contract，后续再替换成更直接的 Vertex AI adapter。

## Deploy Split/Sample

The split/sample serverless node lives under `cloud/split_sample/`. It samples
`3` frames per shot and uses `ffmpeg` inside the runtime.

Target resources:

```text
CPU:    2 vCPU
Memory: 2 GiB
Timeout: 900 sec
```

GCP Cloud Run can be deployed directly from source:

```bash
PROJECT_ID=project-525a8937-7589-4708-842 \
scripts/deploy/deploy_split_sample_gcp.sh
```

AWS Lambda and Aliyun Function Compute can be deployed with the prepared scripts:

```text
scripts/deploy/deploy_split_sample_aws.sh
scripts/deploy/deploy_split_sample_aliyun.sh
```

AWS Lambda does not expose a direct vCPU setting; the deployment script sets
memory to `2048 MB`, and CPU is managed by Lambda. Aliyun FC config sets
`cpu: 2` and `memorySize: 2048`. The Aliyun script packages the HTTP runtime as
a zip with Linux static `ffmpeg`, uploads it to each regional OSS bucket, and
creates or updates the FC3 HTTP trigger.

## Runtime planning algorithms

prototype 里也实现了 `sim` 中同一类部署选择算法，但输入不是 synthetic simulation CSV，而是 runtime profile：

- `logical_optimal`: 每个逻辑节点独立选择 capability 最高的 endpoint。
- `single_cloud`: 只允许一个云覆盖全 workflow，在满足 `eta` 违约率时选最低成本，否则选最低违约率。
- `greedy`: 按 DAG 拓扑顺序，为每个节点选择期望节点成本最低、延迟次优的 endpoint。
- `dpgm`: deterministic profile-guided MILP，最小化 profile 成本并约束 profile critical-path latency；不可行时使用 latency slack。
- `mtgp`: runtime profile 版 multi-tree GP hyper-heuristic，进化 priority rule 后选择 deployment。
- `decomposition` / `skyflow`: SkyFlow-style active-scenario CVaR MILP，逐步把违反 SLA 的 scenario 加入 active set。

和 `sim` 的关键区别是，prototype 的网络模型显式使用用户侧中转：每条 workflow 边都按 `provider -> user relay -> next provider` 计算传输 latency/cost，而不是云到云直连。

运行 planner 并执行 workflow：

```bash
PYTHONPATH=src python -m skybridge_prototype.cli \
  --config configs/prototype.mock.json \
  --profile configs/planner.mock.json \
  --planner decomposition \
  --video samples/demo_video.txt \
  --question "What is happening in the video?"
```

输出 JSON 会多一个 `planning` 字段，里面包含 assignments、metrics、active scenario 数量和 convergence history。planner 选出的 `selected_providers` 会自动覆盖 broker 的 provider 选择。

## HTTP JSON contracts

Shot detection endpoint:

```json
{
  "video": {
    "name": "input.mp4",
    "data_b64": "..."
  }
}
```

returns:

```json
{
  "shots": [
    {"shot_id": "shot-001", "start_ms": 0, "end_ms": 8000, "confidence": 0.98}
  ]
}
```

Split/sample endpoint:

```json
{
  "video": {"name": "input.mp4", "data_b64": "..."},
  "shots": [{"shot_id": "shot-001", "start_ms": 0, "end_ms": 8000, "confidence": 0.98}],
  "samples_per_shot": 3
}
```

returns:

```json
{
  "frames": [
    {
      "frame_id": "frame-001-01",
      "shot_id": "shot-001",
      "timestamp_ms": 4000,
      "media_type": "image/jpeg",
      "data_b64": "...",
      "metadata": {}
    }
  ]
}
```

Caption endpoint:

```json
{
  "frame": {"name": "frame-001-01.jpg", "data_b64": "..."},
  "timestamp_ms": 4000
}
```

returns:

```json
{
  "frame_id": "frame-001-01",
  "timestamp_ms": 4000,
  "text": "A concise caption."
}
```

QA endpoint:

```json
{
  "question": "What is happening?",
  "captions": [
    {"frame_id": "frame-001-01", "timestamp_ms": 4000, "text": "A concise caption.", "provider": "gcp"}
  ]
}
```

returns:

```json
{
  "text": "Final answer.",
  "evidence_frame_ids": ["frame-001-01"]
}
```

## Code layout

- `src/skybridge_prototype/broker.py`: local workflow scheduler.
- `src/skybridge_prototype/relay.py`: user-side artifact relay and transfer log.
- `src/skybridge_prototype/providers.py`: provider protocols.
- `src/skybridge_prototype/adapters/mock.py`: deterministic mock providers.
- `src/skybridge_prototype/adapters/http_json.py`: generic HTTP bridge for cloud services/serverless nodes.
- `src/skybridge_prototype/adapters/anthropic.py`: minimal Anthropic Messages API adapter for caption and QA.
- `src/skybridge_prototype/factory.py`: config-driven provider construction.
