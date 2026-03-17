from simulation.distribution import sample_execution_time_seconds, sample_link_between, sample_video_size_mb


def estimate_total_latency_local_to_gcp_us_for_segmentation(
    *, min_mb: float = 5.0, max_mb: float = 500.0
) -> float:
    """
    采样一个视频大小（MB），假设从 local 传输到 gcp_us，然后在 gcp_us 做 video segmentation，
    返回总延迟（秒）。

    组成：
    - network_rtt_seconds = RTT(ms) / 1000
    - transfer_seconds = (video_size_MB * 8) / bandwidth_Mbps
    - execution_seconds = sample_execution_time_seconds(..., operation_name="seg_google_us", video_size_mb=video_size_MB)
    """
    video_size_mb = sample_video_size_mb(min_mb=min_mb, max_mb=max_mb)

    rtt_ms, bw_mbps = sample_link_between("local", "gcp_us")
    network_rtt_seconds = rtt_ms / 1000.0
    transfer_seconds = (video_size_mb * 8.0) / bw_mbps

    execution_seconds = sample_execution_time_seconds(
        task_name="Video Segmentation",
        operation_name="seg_google_us",
        video_size_mb=video_size_mb,
    )

    return network_rtt_seconds + transfer_seconds + execution_seconds


# def main() -> None:
#     # 固定随机种子，保证每次运行结果可复现
#     rng = np.random.default_rng(123)

#     src, dst = "local", "gcp_tw"

#     # 1) 一次性采样该链路的 RTT(ms) + bandwidth(Mbps)
#     rtt_ms, bw_mbps = sample_link_between(src, dst, rng=rng)
#     print(f"[{src} -> {dst}] rtt_ms={rtt_ms:.2f}, bw_mbps={bw_mbps:.2f}")

#     # 2) 只采样 RTT
#     rtt_params = get_rtt_params(src, dst)
#     rtt_only = sample_rtt_ms(rtt_params, rng=rng)
#     print(f"[{src} -> {dst}] rtt_only_ms={rtt_only:.2f}")

#     # 3) 只采样 bandwidth
#     bw_params = get_bw_params(src, dst)
#     bw_only = sample_bandwidth_mbps(bw_params, rng=rng)
#     print(f"[{src} -> {dst}] bw_only_mbps={bw_only:.2f}")

#     # 4) 查看支持的节点列表
#     print("NODES:", ", ".join(NODES))

#     # 5) 示例：采样 video size -> 传输到 gcp_us -> 在 gcp_us 做 segmentation -> 总延迟
#     total_seconds = estimate_total_latency_local_to_gcp_us_for_segmentation(rng=rng)
#     print(f"[local -> gcp_us + seg_google_us] total_latency_s={total_seconds:.3f}")


if __name__ == "__main__":
    print(estimate_total_latency_local_to_gcp_us_for_segmentation())

