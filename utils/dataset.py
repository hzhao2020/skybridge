from pathlib import Path
from torch.utils.data import Dataset

from utils.util import load_json


class BaseDataset(Dataset):
    """
    通用的 JSON 数据集基类。

    - 负责加载 train.json / test.json
    - 子类只需要关心如何根据 JSON 条目构造返回样本即可
    """

    def __init__(self, json_path: str | Path, video_root: str | Path):
        self.json_path = Path(json_path)
        self.video_root = Path(video_root)
        self.data = load_json(self.json_path)

    def load_question(self, idx: int) -> str:
        return self.data[idx]["question"]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        """默认行为：返回原始 JSON 条目，子类一般会重写该方法。"""
        return self.data[idx]


class EgoSchemaDataset(BaseDataset):
    """
    EgoSchema 数据集

    目录结构假定为：
        datasets/EgoSchema/train.json
        datasets/EgoSchema/test.json
        datasets/EgoSchema/videos_sampled/<video_name>.mp4
    """

    def __init__(self, root_dir: str | Path, split: str = "train"):
        """
        Args:
            root_dir: 数据集根目录，通常为 'datasets/EgoSchema'
            split: 'train' 或 'test'
        """
        root_dir = Path(root_dir)
        assert split in {"train", "test"}, "split 必须是 'train' 或 'test'"

        json_path = root_dir / f"{split}.json"
        video_root = root_dir / "videos_sampled"

        super().__init__(json_path=json_path, video_root=video_root)

    def __getitem__(self, idx: int) -> dict:
        item = self.data[idx]
        video_name = item["video_name"]

        sample = {
            "qid": item.get("qid"),
            "question": item["question"],
            "options": item.get("options"),
            "answer": item.get("answer"),
            "answer_idx": item.get("answer_idx"),
            "video_name": video_name,
            # 绝对视频路径
            "video_path": str(self.video_root / video_name),
        }
        return sample


class NExTQADataset(BaseDataset):
    """
    NExTQA 数据集

    目录结构假定为：
        datasets/NExTQA/train.json
        datasets/NExTQA/test.json
        datasets/NExTQA/videos_sampled/<video_name>.mp4
    """

    def __init__(self, root_dir: str | Path, split: str = "train"):
        root_dir = Path(root_dir)
        assert split in {"train", "test"}, "split 必须是 'train' 或 'test'"

        json_path = root_dir / f"{split}.json"
        video_root = root_dir / "videos_sampled"

        super().__init__(json_path=json_path, video_root=video_root)

    def __getitem__(self, idx: int) -> dict:
        item = self.data[idx]
        video_name = item["video_name"]

        sample = {
            "qid": item.get("qid"),
            "question": item["question"],
            "options": item.get("options"),
            "answer": item.get("answer"),
            "answer_idx": item.get("answer_idx"),
            "type": item.get("type"),
            "frame_count": item.get("frame_count"),
            "video_name": video_name,
            "video_path": str(self.video_root / video_name),
        }
        return sample


class ActivityNetQADataset(BaseDataset):
    """
    ActivityNetQA 数据集

    目录结构假定为：
        datasets/ActivityNetQA/train.json
        datasets/ActivityNetQA/test.json
        datasets/ActivityNetQA/videos_sampled/<video_name>.mp4
    """

    def __init__(self, root_dir: str | Path, split: str = "train"):
        root_dir = Path(root_dir)
        assert split in {"train", "test"}, "split 必须是 'train' 或 'test'"

        json_path = root_dir / f"{split}.json"
        video_root = root_dir / "videos_sampled"

        super().__init__(json_path=json_path, video_root=video_root)

    def __getitem__(self, idx: int) -> dict:
        item = self.data[idx]
        video_name = item["video_name"]

        sample = {
            "qid": item.get("qid"),
            "question": item["question"],
            "answer": item.get("answer"),
            "video_name": video_name,
            "video_path": str(self.video_root / video_name),
        }
        return sample


def build_dataset(name: str, split: str = "train", datasets_root: str | Path = "datasets") -> BaseDataset:
    """
    工厂函数：根据数据集名称和 split 构造对应的 Dataset 实例。

    Args:
        name: 数据集名称，支持 'EgoSchema' / 'NExTQA' / 'ActivityNetQA'
        split: 'train' 或 'test'
        datasets_root: 所有数据集所在的根目录，默认 'datasets'
    """
    name = name.lower()
    datasets_root = Path(datasets_root)

    if name == "egoschema":
        return EgoSchemaDataset(datasets_root / "EgoSchema", split)
    if name == "nextqa":
        return NExTQADataset(datasets_root / "NExTQA", split)
    if name == "activitynetqa":
        return ActivityNetQADataset(datasets_root / "ActivityNetQA", split)

    raise ValueError(f"Unknown dataset name: {name}")
