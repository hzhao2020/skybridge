# import os
# import json
# import random
# import pandas as pd
# from collections import defaultdict
# from typing import List, Dict, Any, Tuple
#
#
# def activitynetqa_sample(path: str, train_num: int = 300, test_num: int = 200, seed: int = 42) -> Tuple[
#     List[Dict], List[Dict]]:
#     """
#     1. 采样 (train_num + test_num) 个样本，保证每个样本对应唯一的 Video。
#     2. 将结果划分为 Training set 和 Test set。
#     3. 将结果保存为 train.json 和 test.json 到 path 目录下。
#     """
#     total_num = train_num + test_num
#
#     # 局部随机种子，保证可复现
#     rng = random.Random(seed)
#
#     video_dir = os.path.join(path, 'videos')
#     annotation_dir = os.path.join(path, 'annotations')
#     q_file = os.path.join(annotation_dir, 'val_q.json')
#     a_file = os.path.join(annotation_dir, 'val_a.json')
#
#     assert os.path.exists(video_dir), f"Video directory not found: {video_dir}"
#     assert os.path.exists(annotation_dir), f"Annotation directory not found: {annotation_dir}"
#
#     # --- 加载与分组 ---
#     print(f"[ActivityNetQA] 正在加载 JSON 数据...")
#     with open(q_file, 'r', encoding='utf-8') as f:
#         questions_data = json.load(f)
#     with open(a_file, 'r', encoding='utf-8') as f:
#         answers_data = json.load(f)
#
#     # 建立 Answer 索引
#     answers_map = {item['question_id']: item for item in answers_data}
#
#     # 按 video_name 对 Question 进行分组
#     video_to_questions = defaultdict(list)
#
#     for q_item in questions_data:
#         qid = q_item.get('question_id')
#         vname = q_item.get('video_name')
#
#         if qid in answers_map and vname:
#             ans_item = answers_map[qid]
#             q_item['ground_truth_answer'] = ans_item.get('answer') or ans_item.get('answers')
#             video_to_questions[vname].append(q_item)
#
#     # --- 筛选有效 Video ---
#     all_video_names = sorted(list(video_to_questions.keys()))  # 排序保证复现
#
#     valid_videos = []
#     for vname in all_video_names:
#         full_path = os.path.join(video_dir, f"{vname}.mp4")
#         if os.path.exists(full_path):
#             valid_videos.append(vname)
#
#     total_valid_videos = len(valid_videos)
#     print(f"[ActivityNetQA] 有效视频总数: {total_valid_videos}")
#
#     if total_num > total_valid_videos:
#         raise ValueError(f"请求 {total_num} 个不重复视频，但数据集中只有 {total_valid_videos} 个有效视频。")
#
#     # --- 随机采样 Video & 抽取 Question ---
#     sampled_video_names = rng.sample(valid_videos, total_num)
#     final_samples = []
#
#     for vname in sampled_video_names:
#         possible_questions = video_to_questions[vname]
#         possible_questions.sort(key=lambda x: x['question_id'])  # 排序保证复现
#         selected_q = rng.choice(possible_questions)
#
#         final_samples.append({
#             "video_path": os.path.join(video_dir, f"{vname}.mp4"),
#             "question": selected_q['question'],
#             "answer": selected_q['ground_truth_answer'],
#             "question_id": selected_q['question_id'],
#             "video_name": vname
#         })
#
#     # --- 划分 Train / Test ---
#     rng.shuffle(final_samples)  # 再次打乱
#
#     train_set = final_samples[:train_num]
#     test_set = final_samples[train_num:]
#
#     # --- 保存结果 ---
#     train_output_path = os.path.join(path, 'train.json')
#     test_output_path = os.path.join(path, 'test.json')
#
#     print(f"[ActivityNetQA] 正在保存文件到: {path}")
#     with open(train_output_path, 'w', encoding='utf-8') as f:
#         json.dump(train_set, f, ensure_ascii=False, indent=4)
#     with open(test_output_path, 'w', encoding='utf-8') as f:
#         json.dump(test_set, f, ensure_ascii=False, indent=4)
#
#     # --- 验证 ---
#     print("-" * 30)
#     print(f"Total Request: {total_num} (Train: {train_num}, Test: {test_num})")
#     print(f"Saved Train  : {len(train_set)}")
#     print(f"Saved Test   : {len(test_set)}")
#
#     # 验证重叠
#     train_vids = set(x['video_name'] for x in train_set)
#     test_vids = set(x['video_name'] for x in test_set)
#     intersection = train_vids.intersection(test_vids)
#
#     if len(intersection) == 0:
#         print("验证成功: Training集和Test集无视频重叠。")
#     else:
#         print("严重错误: 数据集划分存在视频泄露！")
#
#     return train_set, test_set
#
#
# def nextqa_sample(path: str, train_num: int = 300, test_num: int = 200, seed: int = 42) -> Tuple[
#     List[Dict], List[Dict]]:
#     """
#     NextQA 采样函数。
#     """
#     total_num = train_num + test_num
#     rng = random.Random(seed)
#
#     csv_path = os.path.join(path, "dataset", "nextqa", "val.csv")
#     video_root = os.path.join(path, "NExTVideo")
#
#     assert os.path.exists(csv_path), f"Metadata CSV not found: {csv_path}"
#     assert os.path.exists(video_root), f"Video root dir not found: {video_root}"
#
#     # --- 建立视频文件索引 ---
#     print(f"[NextQA] 正在扫描 NExTVideo 目录建立文件索引...")
#     video_path_map = {}
#     for root, dirs, files in os.walk(video_root):
#         for file in files:
#             if file.endswith(".mp4"):
#                 video_id = os.path.splitext(file)[0]
#                 video_path_map[str(video_id)] = os.path.join(root, file)
#     print(f"[NextQA] 扫描完成，共找到 {len(video_path_map)} 个视频文件。")
#
#     # --- 加载 CSV 并分组 ---
#     print(f"[NextQA] 正在读取 CSV 元数据...")
#     df = pd.read_csv(csv_path)
#     video_to_questions = defaultdict(list)
#
#     for _, row in df.iterrows():
#         vid = str(row['video'])
#         if vid in video_path_map:
#             ans_idx = int(row['answer'])
#             ans_col = f"a{ans_idx}"
#             ground_truth_text = row[ans_col]
#             options = [str(row[f'a{i}']) for i in range(5)]
#
#             q_obj = {
#                 "video_path": video_path_map[vid],
#                 "video_name": vid,
#                 "question": row['question'],
#                 "answer": ground_truth_text,
#                 "answer_idx": ans_idx,
#                 "options": options,
#                 "qid": str(row['qid']),
#                 "type": row['type']
#             }
#             video_to_questions[vid].append(q_obj)
#
#     # --- 筛选与采样 ---
#     all_video_ids = sorted(list(video_to_questions.keys()))
#     total_valid_videos = len(all_video_ids)
#
#     print(f"[NextQA] 匹配成功的有效视频数: {total_valid_videos}")
#
#     if total_num > total_valid_videos:
#         raise ValueError(f"请求采样 {total_num} 个，但只有 {total_valid_videos} 个有效视频。")
#
#     sampled_video_ids = rng.sample(all_video_ids, total_num)
#     final_samples = []
#     for vid in sampled_video_ids:
#         candidates = video_to_questions[vid]
#         candidates.sort(key=lambda x: x['qid'])
#         selected_q = rng.choice(candidates)
#         final_samples.append(selected_q)
#
#     # --- 划分 Train / Test ---
#     rng.shuffle(final_samples)
#
#     train_set = final_samples[:train_num]
#     test_set = final_samples[train_num:]
#
#     # --- 保存 ---
#     print(f"[NextQA] 正在保存结果到: {path}")
#     with open(os.path.join(path, "train.json"), 'w', encoding='utf-8') as f:
#         json.dump(train_set, f, ensure_ascii=False, indent=4)
#     with open(os.path.join(path, "test.json"), 'w', encoding='utf-8') as f:
#         json.dump(test_set, f, ensure_ascii=False, indent=4)
#
#     # --- 验证 ---
#     print("-" * 30)
#     print(f"Total Request: {total_num} (Train: {train_num}, Test: {test_num})")
#     print(f"Saved Train  : {len(train_set)}")
#     print(f"Saved Test   : {len(test_set)}")
#
#     train_vids = set(x['video_name'] for x in train_set)
#     test_vids = set(x['video_name'] for x in test_set)
#     if len(train_vids.intersection(test_vids)) == 0:
#         print("验证成功: Train/Test 视频无重叠。")
#     else:
#         print("错误: 视频泄露检测失败！")
#
#     return train_set, test_set
#
#
# def egoschema_sample(path: str, train_num: int = 300, test_num: int = 200, seed: int = 42) -> Tuple[
#     List[Dict], List[Dict]]:
#     """
#     EgoSchema 采样函数。
#     """
#     total_num = train_num + test_num
#     rng = random.Random(seed)
#
#     # 注意：这里保留了你原始代码中的绝对路径逻辑
#     video_root = r"D:\Document\proj\SkyBridge\egochema\videos\good_clips_git"
#     subset_answers_path = os.path.join(path, "subset_answers.json")
#     questions_path = os.path.join(path, "questions.json")
#
#     assert os.path.exists(subset_answers_path), f"Answers file missing: {subset_answers_path}"
#     assert os.path.exists(questions_path), f"Questions file missing: {questions_path}"
#     assert os.path.exists(video_root), f"Video root missing: {video_root}"
#
#     # --- 加载数据 ---
#     print(f"[EgoSchema] 正在加载 JSON 数据...")
#     with open(subset_answers_path, 'r', encoding='utf-8') as f:
#         subset_answers = json.load(f)
#     with open(questions_path, 'r', encoding='utf-8') as f:
#         questions_list = json.load(f)
#
#     # --- 数据合并与分组 ---
#     video_to_questions = defaultdict(list)
#     valid_video_count = 0
#
#     for q_item in questions_list:
#         qid = q_item.get("q_uid")
#         if qid not in subset_answers:
#             continue
#
#         video_name = qid
#         video_full_path = os.path.join(video_root, f"{video_name}.mp4")
#
#         if not os.path.exists(video_full_path):
#             continue
#
#         ans_idx = subset_answers[qid]
#         ans_key = f"option {ans_idx}"
#         ground_truth_text = q_item.get(ans_key)
#
#         options = []
#         for i in range(5):
#             opt_val = q_item.get(f"option {i}")
#             if opt_val:
#                 options.append(opt_val)
#
#         final_obj = {
#             "video_path": video_full_path,
#             "video_name": video_name,
#             "question": q_item.get("question"),
#             "answer": ground_truth_text,
#             "answer_idx": ans_idx,
#             "options": options,
#             "q_uid": qid
#         }
#
#         if len(video_to_questions[video_name]) == 0:
#             valid_video_count += 1
#         video_to_questions[video_name].append(final_obj)
#
#     print(f"[EgoSchema] 有效视频数: {valid_video_count}")
#
#     if total_num > valid_video_count:
#         raise ValueError(f"请求采样 {total_num} 个，但只有 {valid_video_count} 个有效视频。")
#
#     # --- 采样与划分 ---
#     all_video_names = sorted(list(video_to_questions.keys()))
#     sampled_video_names = rng.sample(all_video_names, total_num)
#
#     final_samples = []
#     for vname in sampled_video_names:
#         candidates = video_to_questions[vname]
#         candidates.sort(key=lambda x: x['q_uid'])
#         selected_q = rng.choice(candidates)
#         final_samples.append(selected_q)
#
#     rng.shuffle(final_samples)
#
#     train_set = final_samples[:train_num]
#     test_set = final_samples[train_num:]
#
#     # --- 保存 ---
#     print(f"[EgoSchema] 正在保存结果到: {path}")
#     with open(os.path.join(path, "train.json"), 'w', encoding='utf-8') as f:
#         json.dump(train_set, f, ensure_ascii=False, indent=4)
#     with open(os.path.join(path, "test.json"), 'w', encoding='utf-8') as f:
#         json.dump(test_set, f, ensure_ascii=False, indent=4)
#
#     # --- 验证 ---
#     print("-" * 30)
#     print(f"Total Request: {total_num} (Train: {train_num}, Test: {test_num})")
#     print(f"Saved Train  : {len(train_set)}")
#     print(f"Saved Test   : {len(test_set)}")
#
#     train_vids = set(x['video_name'] for x in train_set)
#     test_vids = set(x['video_name'] for x in test_set)
#     if len(train_vids.intersection(test_vids)) == 0:
#         print("验证成功: Train/Test 视频无重叠。")
#     else:
#         print("错误: 视频泄露检测失败！")
#
#     return train_set, test_set
#
#
# if __name__ == '__main__':
#     # 设置所需的数量
#     TRAIN_NUM = 300
#     TEST_NUM = 200
#
#     print(">>> 1. Processing ActivityNetQA")
#     activitynetqa_sample(
#         path=r'D:\Document\proj\SkyBridge\src\datasets\ActivityNetQA_VAL',
#         train_num=TRAIN_NUM,
#         test_num=TEST_NUM
#     )
#     print("\n" + "=" * 50 + "\n")
#
#     print(">>> 2. Processing NExTQA")
#     nextqa_sample(
#         path=r'/datasets/NExTQA',
#         train_num=TRAIN_NUM,
#         test_num=TEST_NUM
#     )
#     print("\n" + "=" * 50 + "\n")
#
#     print(">>> 3. Processing EgoSchema")
#     # EgoSchema 的 video_root 已经在函数内部 hardcode，这里传入 base path 即可
#     egoschema_sample(
#         path=r'D:\Document\proj\SkyBridge\src\datasets\EgoSchema',
#         train_num=TRAIN_NUM,
#         test_num=TEST_NUM
#     )



from util import *
import os
import json
import random
import pandas as pd
from collections import defaultdict
from typing import List, Dict, Any, Tuple
from tqdm import tqdm


def activitynetqa_sample(train_num: int = 300, test_num: int = 200, seed: int = 42) -> Tuple[
    List[Dict], List[Dict]]:
    """
    1. 采样 (train_num + test_num) 个样本，保证每个样本对应唯一的 Video。
    2. 将结果划分为 Training set 和 Test set。
    3. 将结果保存为 train.json 和 test.json 到 path 目录下。
    """
    total_sampled_num = train_num + test_num

    # 局部随机种子，保证可复现
    rng = random.Random(seed)

    video_dir = r"D:\Document\proj\SkyBridge\src\datasets\ActivityNetQA\videos"
    questions_file = r"D:\Document\proj\SkyBridge\src\datasets\ActivityNetQA\annotations\val_q.json"
    answers_file = r"D:\Document\proj\SkyBridge\src\datasets\ActivityNetQA\annotations\val_a.json"
    sampled_video_dir = r"D:\Document\proj\SkyBridge\src\datasets\ActivityNetQA\videos_sampled"
    makedir(sampled_video_dir)

    questions_data = load_json(questions_file)
    answers_data = load_json(answers_file)

    # 建立 Answer 索引
    answers_map = {item['question_id']: item for item in answers_data}

    # 按 video_name 对 Question 进行分组
    video_to_questions = defaultdict(list)

    for q_item in questions_data:
        qid = q_item.get('question_id')
        vname = q_item.get('video_name')

        if qid in answers_map and vname:
            ans_item = answers_map[qid]
            q_item['ground_truth_answer'] = ans_item.get('answer')
            video_to_questions[vname].append(q_item)

    # --- 筛选有效 Video ---
    all_video_names = sorted(list(video_to_questions.keys()))  # 排序保证复现

    valid_videos = []
    for vname in all_video_names:
        full_path = os.path.join(video_dir, f"{vname}.mp4")
        if os.path.exists(full_path):
            valid_videos.append(vname)

    total_valid_videos = len(valid_videos)
    print(f"[ActivityNetQA] 有效视频总数: {total_valid_videos}")

    if total_sampled_num > total_valid_videos:
        raise ValueError(f"请求 {total_sampled_num} 个不重复视频，但数据集中只有 {total_valid_videos} 个有效视频。")

    # --- 随机采样 Video & 抽取 Question ---
    sampled_video_names = rng.sample(valid_videos, total_sampled_num)
    assert len(set(sampled_video_names)) == total_sampled_num
    final_samples = []

    for vname in tqdm(sampled_video_names):
        possible_questions = video_to_questions[vname]
        possible_questions.sort(key=lambda x: x['question_id'])  # 排序保证复现
        selected_q = rng.choice(possible_questions)
        video_src_path = os.path.join(video_dir, f"{vname}.mp4")
        video_des_path = os.path.join(sampled_video_dir, f"{vname}.mp4")
        copy(video_src_path, video_des_path)
        final_samples.append({
            "question": selected_q['question'],
            "answer": selected_q['ground_truth_answer'],
            "qid": selected_q['question_id'],
            "video_name": f"{vname}.mp4"
        })

    # --- 划分 Train / Test ---
    rng.shuffle(final_samples)  # 再次打乱

    train_set = final_samples[:train_num]
    test_set = final_samples[train_num:]

    # --- 保存结果 ---
    train_output_path = os.path.join(r'D:\Document\proj\SkyBridge\src\datasets\ActivityNetQA', 'train.json')
    test_output_path = os.path.join(r'D:\Document\proj\SkyBridge\src\datasets\ActivityNetQA', 'test.json')

    save_json(train_set, train_output_path)
    save_json(test_set, test_output_path)


def nextqa_sample(train_num: int = 300, test_num: int = 200, seed: int = 42) -> Tuple[
    List[Dict], List[Dict]]:
    """
    NextQA 采样函数。
    """
    total_sampled_num = train_num + test_num
    rng = random.Random(seed)

    csv_path = r'D:\Document\proj\SkyBridge\src\datasets\NExTQA\dataset\nextqa\train.csv'
    video_root = r'D:\Document\proj\SkyBridge\src\datasets\NExTQA\NExTVideo'
    sampled_video_dir = r'D:\Document\proj\SkyBridge\src\datasets\NExTQA\videos_sampled'
    makedir(sampled_video_dir)

    # --- 建立视频文件索引 ---
    print(f"[NextQA] 正在扫描 NExTVideo 目录建立文件索引...")
    video_path_map = {}
    for root, dirs, files in os.walk(video_root):
        for file in files:
            if file.endswith(".mp4"):
                video_id = os.path.splitext(file)[0]
                video_path_map[str(video_id)] = os.path.join(root, file)
    print(f"[NextQA] 扫描完成，共找到 {len(video_path_map)} 个视频文件。")

    # --- 加载 CSV 并分组 ---
    print(f"[NextQA] 正在读取 CSV 元数据...")
    df = pd.read_csv(csv_path)
    video_to_questions = defaultdict(list)

    for _, row in df.iterrows():
        vid = str(row['video'])
        if vid in video_path_map:
            ans_idx = int(row['answer'])
            ans_col = f"a{ans_idx}"
            ground_truth_text = row[ans_col]
            options = [str(row[f'a{i}']) for i in range(5)]

            q_obj = {
                "video_path": video_path_map[vid],
                "video_name": f"{vid}.mp4",
                "qid": vid,
                "question": row['question'],
                "answer": ground_truth_text,
                "answer_idx": ans_idx,
                "options": options,
                "type": row['type'],
                "frame_count": row['frame_count']
            }
            video_to_questions[vid].append(q_obj)

    # --- 筛选与采样 ---
    all_video_ids = sorted(list(video_to_questions.keys()))
    total_valid_videos = len(all_video_ids)

    print(f"[NextQA] 匹配成功的有效视频数: {total_valid_videos}")

    if total_sampled_num > total_valid_videos:
        raise ValueError(f"请求采样 {total_sampled_num} 个，但只有 {total_valid_videos} 个有效视频。")

    sampled_video_ids = rng.sample(all_video_ids, total_sampled_num)
    final_samples = []
    for vid in tqdm(sampled_video_ids):
        candidates = video_to_questions[vid]
        candidates.sort(key=lambda x: x['qid'])
        selected_q = rng.choice(candidates)
        # 复制video
        video_src_path = selected_q['video_path']
        video_des_path = os.path.join(sampled_video_dir, selected_q['video_name'])
        copy(video_src_path, video_des_path)
        selected_q.pop("video_path", None)
        final_samples.append(selected_q)

    # --- 划分 Train / Test ---
    rng.shuffle(final_samples)

    train_set = final_samples[:train_num]
    test_set = final_samples[train_num:]

    train_output_path = os.path.join(r'D:\Document\proj\SkyBridge\src\datasets\NExTQA', 'train.json')
    test_output_path = os.path.join(r'D:\Document\proj\SkyBridge\src\datasets\NExTQA', 'test.json')

    save_json(train_set, train_output_path)
    save_json(test_set, test_output_path)


def egoschema_sample(train_num: int = 300, test_num: int = 200, seed: int = 42) -> Tuple[
    List[Dict], List[Dict]]:
    """
    EgoSchema 采样函数。
    """
    total_sampled_num = train_num + test_num
    rng = random.Random(seed)

    # 注意：这里保留了你原始代码中的绝对路径逻辑
    video_root = r"D:\Document\proj\SkyBridge\egochema\videos\good_clips_git"
    subset_answers_path = r"D:\Document\proj\SkyBridge\src\datasets\EgoSchema\subset_answers.json"
    questions_path = r"D:\Document\proj\SkyBridge\src\datasets\EgoSchema\questions.json"
    sampled_video_dir = r"D:\Document\proj\SkyBridge\src\datasets\EgoSchema\videos_sampled"
    makedir(sampled_video_dir)

    # --- 加载数据 ---
    print(f"[EgoSchema] 正在加载 JSON 数据...")
    subset_answers = load_json(subset_answers_path)
    questions_list = load_json(questions_path)

    # --- 数据合并与分组 ---
    video_to_questions = defaultdict(list)
    valid_video_count = 0

    for q_item in questions_list:
        qid = q_item.get("q_uid")
        if qid not in subset_answers:
            continue

        video_name = qid
        video_full_path = os.path.join(video_root, f"{video_name}.mp4")

        if not os.path.exists(video_full_path):
            continue

        ans_idx = subset_answers[qid]
        ans_key = f"option {ans_idx}"
        ground_truth_text = q_item.get(ans_key)

        options = []
        for i in range(5):
            opt_val = q_item.get(f"option {i}")
            if opt_val:
                options.append(opt_val)

        final_obj = {
            "video_path": video_full_path,
            "video_name": f'{video_name}.mp4',
            "question": q_item.get("question"),
            "answer": ground_truth_text,
            "answer_idx": ans_idx,
            "options": options,
            "qid": qid
        }

        if len(video_to_questions[video_name]) == 0:
            valid_video_count += 1
        video_to_questions[video_name].append(final_obj)

    print(f"[EgoSchema] 有效视频数: {valid_video_count}")

    if total_sampled_num > valid_video_count:
        raise ValueError(f"请求采样 {total_sampled_num} 个，但只有 {valid_video_count} 个有效视频。")

    # --- 采样与划分 ---
    all_video_names = sorted(list(video_to_questions.keys()))
    sampled_video_names = rng.sample(all_video_names, total_sampled_num)

    final_samples = []
    for vname in tqdm(sampled_video_names):
        candidates = video_to_questions[vname]
        candidates.sort(key=lambda x: x['qid'])
        selected_q = rng.choice(candidates)
        video_src_path = selected_q['video_path']
        video_des_path = os.path.join(sampled_video_dir, selected_q['video_name'])
        copy(video_src_path, video_des_path)
        selected_q.pop("video_path", None)
        final_samples.append(selected_q)

    rng.shuffle(final_samples)

    train_set = final_samples[:train_num]
    test_set = final_samples[train_num:]

    train_output_path = os.path.join(r'D:\Document\proj\SkyBridge\src\datasets\EgoSchema', 'train.json')
    test_output_path = os.path.join(r'D:\Document\proj\SkyBridge\src\datasets\EgoSchema', 'test.json')

    save_json(train_set, train_output_path)
    save_json(test_set, test_output_path)


if __name__ == '__main__':
    # activitynetqa_sample()
    # nextqa_sample()
    egoschema_sample()