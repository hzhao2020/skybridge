import numpy as np
from distribution import build_distribution_parameters, Query
from workflow import Workflow
from algos import selection_single_cloud, selection_greedy_locality

def calibrate_budget(num_samples=50, baseline_type="SC", slack=1.2):
    """
    通过运行基线策略并进行线性拟合，寻找合适的 Budget 参数。
    
    :param num_samples: 采样的视频大小数量
    :param baseline_type: 锚定的基线类型 ("SC" 单云, "GL" 贪心本地化)
    :param slack: 松弛因子，1.2 表示给基线增加 20% 的容忍度
    """
    params = build_distribution_parameters()
    workflow = Workflow(params=params)

    # 模拟从 5MB 到 500MB 的视频大小 (与你 param.yaml 中的 data_size 区间一致)
    data_sizes = np.linspace(5.0, 500.0, num_samples)
    costs = []
    latencies = []

    print(f"正在使用 {baseline_type} 基线收集数据以拟合 Budget 参数...")

    for ds in data_sizes:
        q = Query()
        q.data_size_MB = ds
        
        # 选择一个基线策略来锚定预算
        if baseline_type == "SC":
            sel = selection_single_cloud(workflow)
        elif baseline_type == "GL":
            sel = selection_greedy_locality(workflow, [q], ref_size="mean")
        else:
            from algos import selection_logical_optimal
            sel = selection_logical_optimal(workflow)

        # 关键：使用 deterministic=True 获取无抖动的理论期望值，保证线性回归的完美拟合
        obs = workflow.sample_observation(sel, float(ds), deterministic=True)
        costs.append(obs.cost)
        latencies.append(obs.latency)

    # 使用最小二乘法 (一阶多项式) 计算斜率和截距
    cost_slope, cost_int = np.polyfit(data_sizes, costs, 1)
    lat_slope, lat_int = np.polyfit(data_sizes, latencies, 1)

    # 打印 YAML 格式的配置
    print("\n=== 请将以下内容复制替换到 param.yaml 中 ===")
    print("budget:")
    print("  baseline:")
    print(f"    latency_intercept_s: {lat_int:.6f}")
    print(f"    latency_slope_per_MB: {lat_slope:.6f}")
    print(f"    cost_intercept_usd: {cost_int:.6f}")
    print(f"    cost_slope_per_MB: {cost_slope:.6f}")
    print("  slack_factor:")
    print(f"    latency: {slack}")
    print(f"    cost: {slack}")
    print("============================================\n")
    print("💡 提示：")
    print("1. 你的 distribution.py 已经原生支持解析带有 slack_factor 的配置。")
    print(f"2. 当前设置的松弛因子为 {slack}（允许在完美基线基础上有 {(slack-1)*100:.0f}% 的环境抖动空间）。")
    print("3. 如果你的 Proposed 算法在仿真中总是 'SLO_ok=N'，可以尝试调大 slack_factor；如果想要更高的门槛，可以将其下调至 1.05。")

if __name__ == "__main__":
    # 推荐使用 SC (Single Cloud) 作为评估基线，因为它避免了跨云 Egress Cost，是一个合理的成本参考点。
    calibrate_budget(baseline_type="SC", slack=1.2)