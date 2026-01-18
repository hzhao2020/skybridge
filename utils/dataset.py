from torch.utils.data import Dataset
from util import *


class BaseDataset(Dataset):
    def __init__(self, json_path):
        self.data = load_json(json_path)

    def load_question(self, idx):
        return self.data[idx]['question']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class EgoSchemaDataset(BaseDataset):
    def __init__(self, json_path):
        super().__init__(json_path)