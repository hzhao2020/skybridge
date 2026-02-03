# 存储桶传输性能测试工具

## 简介

`test_bucket_transmission.py` 用于测试不同 storage bucket 之间的传输性能，包括：

- **RTT (Round-Trip Time)**: 往返时间测试，测量小文件（1KB）的传输延迟
- **Bandwidth**: 带宽测试，测量不同大小文件（1MB, 10MB, 100MB）的传输速度

## 支持的存储桶

- **GCP**: `gcp_us`, `gcp_tw`, `gcp_sg`
- **AWS**: `aws_us`, `aws_sg`
- **Azure**: `azure_ea`, `azure_wu`

## 使用方法

### 基本用法

测试所有bucket对的所有性能指标：

```bash
cd profile
python test_bucket_transmission.py
```

### 仅测试RTT

```bash
python test_bucket_transmission.py --rtt-only
```

### 仅测试带宽

```bash
python test_bucket_transmission.py --bandwidth-only
```

### 测试指定的bucket对

```bash
# 测试单个bucket对
python test_bucket_transmission.py --pairs gcp_us:aws_us

# 测试多个bucket对
python test_bucket_transmission.py --pairs gcp_us:aws_us gcp_sg:azure_ea
```

### 指定输出文件

```bash
python test_bucket_transmission.py --output my_results.json
```

## 测试结果

测试完成后会：

1. 在控制台打印测试结果摘要
2. 将详细结果保存到JSON文件（默认：`transmission_test_results.json`）

### 结果格式

JSON文件包含：
- `test_time`: 测试时间
- `results`: 测试结果列表，每个结果包含：
  - `source_bucket`: 源bucket
  - `target_bucket`: 目标bucket
  - `source_provider`: 源云提供商
  - `target_provider`: 目标云提供商
  - `test_type`: 测试类型（'rtt' 或 'bandwidth'）
  - `file_size_bytes`: 文件大小（字节）
  - `duration_seconds`: 传输耗时（秒）
  - `bandwidth_mbps`: 带宽（Mbps，仅带宽测试）
  - `rtt_ms`: RTT（毫秒，仅RTT测试）
  - `success`: 是否成功
  - `error_message`: 错误信息（如果失败）

## 注意事项

1. **测试时间**: 完整测试所有bucket对可能需要较长时间，建议：
   - 使用 `--pairs` 参数测试特定的bucket对
   - 使用 `--rtt-only` 或 `--bandwidth-only` 分别测试

2. **成本**: 测试会产生云存储的读写操作，可能产生少量费用

3. **网络环境**: 测试结果会受到当前网络环境的影响，建议在稳定的网络环境下运行

4. **权限**: 确保已配置好各云提供商的认证信息：
   - AWS: `~/.aws/credentials` 或环境变量
   - GCP: `gcloud auth application-default login`
   - Azure: `config.py` 中的连接字符串

## 示例输出

```
================================================================
存储桶传输性能测试
================================================================
开始时间: 2026-02-03 10:00:00

将测试 42 个bucket对

进度: 1/42
============================================================
RTT测试: gcp_us -> gcp_tw
============================================================
创建测试文件: /tmp/test_xxx.bin (1024 bytes)

运行 1/3...
  上传到源bucket: gcp_us...
  上传完成 (0.15s)
  传输到目标bucket: gcp_tw...
  传输完成 (2.34s)
  总耗时: 2.49s

平均RTT: 2456.78ms
...
```
