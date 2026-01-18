import os
import json
import subprocess
import sys
import concurrent.futures
from tqdm import tqdm

# ================= 配置区域 =================
ROOT = "../datasets/ActivityNetQA"
VIDEO_DIR = f"{ROOT}/videos"
ANN_DIR = f"{ROOT}/annotations"
TMP_DIR = f"{ROOT}/tmp"
MAX_WORKERS = 8  # 线程数，建议设置为 4-8。过高可能导致被 YouTube 封 IP


# ===========================================

def load_data():
    """加载必要的 JSON 数据"""
    print("📂 Loading metadata...")
    if not os.path.exists(f"{ANN_DIR}/val_q.json") or not os.path.exists(f"{TMP_DIR}/activitynet_meta.json"):
        print(f"❌ 错误：找不到必要的 JSON 文件。请确保 {ANN_DIR} 和 {TMP_DIR} 下有文件。")
        sys.exit(1)

    val_q = json.load(open(f"{ANN_DIR}/val_q.json"))
    database = json.load(open(f"{TMP_DIR}/activitynet_meta.json"))["database"]

    # 提取去重后的视频列表
    needed_vids = set()
    for item in val_q:
        if "video_name" in item:
            needed_vids.add(item["video_name"])

    return needed_vids, database


def download_single_video(vid, database):
    """单个视频下载工作函数"""

    # 1. 匹配 ID (处理 v_ 前缀问题)
    key = vid
    if key not in database:
        if key.startswith("v_") and key[2:] in database:
            key = key[2:]
        else:
            return "MISSING_META"  # 元数据里找不到

    out_path = os.path.join(VIDEO_DIR, f"{vid}.mp4")

    # 2. 检查文件是否已存在
    if os.path.exists(out_path):
        # 简单检查文件大小，如果是 0KB 的空文件则重新下载
        if os.path.getsize(out_path) > 1024:
            return "EXISTS"

    url = database[key]["url"]

    # 3. 构建下载命令
    # 使用 sys.executable 确保调用的是当前 conda 环境的 python
    cmd = [
        sys.executable, "-m", "yt_dlp",
        url,
        "-f", "best[ext=mp4]/best",
        "-o", out_path,
        "--no-warnings",
        "--ignore-errors",  # 遇到错误继续，不报错退出
        "--quiet",  # 安静模式，不输出进度条，以免多线程乱码
        "--no-part"  # 不生成 .part 文件，下载完直接重命名
    ]

    try:
        # 设置 timeout 防止某个进程卡死
        result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=300)

        if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            return "SUCCESS"
        else:
            # 即使命令运行完了，文件没生成，通常是 Video unavailable
            return "UNAVAILABLE"
    except subprocess.TimeoutExpired:
        return "TIMEOUT"
    except Exception as e:
        return "ERROR"


def main():
    # 1. 创建目录
    os.makedirs(VIDEO_DIR, exist_ok=True)

    # 2. 加载数据
    needed_vids, database = load_data()
    video_list = list(needed_vids)
    total = len(video_list)
    print(f"🚀 准备下载 {total} 个视频 (使用 {MAX_WORKERS} 线程)...")

    # 3. 统计计数器
    stats = {
        "SUCCESS": 0,
        "EXISTS": 0,
        "UNAVAILABLE": 0,
        "MISSING_META": 0,
        "TIMEOUT": 0,
        "ERROR": 0
    }

    # 4. 开启多线程下载池
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # 提交所有任务
        future_to_vid = {executor.submit(download_single_video, vid, database): vid for vid in video_list}

        # 使用 tqdm 显示总体进度
        with tqdm(total=total, desc="Downloading") as pbar:
            for future in concurrent.futures.as_completed(future_to_vid):
                result_status = future.result()
                stats[result_status] = stats.get(result_status, 0) + 1
                pbar.update(1)

                # 在进度条后缀显示实时统计
                pbar.set_postfix(ok=stats["SUCCESS"] + stats["EXISTS"], fail=stats["UNAVAILABLE"])

    # 5. 最终报告
    print("\n" + "=" * 40)
    print("📊 下载任务结束报告")
    print("=" * 40)
    print(f"✅ 成功下载 (New): {stats['SUCCESS']}")
    print(f"⏭️ 跳过已存在 (Exists): {stats['EXISTS']}")
    print(f"❌ 视频失效 (Unavailable): {stats['UNAVAILABLE']}")
    print(f"❓ 元数据缺失 (Missing Meta): {stats['MISSING_META']}")
    print(f"⚠️ 其他错误/超时: {stats['TIMEOUT'] + stats['ERROR']}")

    final_count = len([n for n in os.listdir(VIDEO_DIR) if n.endswith(".mp4")])
    print(f"\n📂 最终文件夹内视频总数: {final_count} / {total}")
    print("=" * 40)


if __name__ == "__main__":
    main()