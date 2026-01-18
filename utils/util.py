import json
import shutil
from pathlib import Path


def copy(src_path, dst_path):
    shutil.copy2(src_path, dst_path)


def load_json(fn):
    with open(fn, 'r') as f:
        data = json.load(f)
    return data


def save_json(data, fn, indent=4):
    with open(fn, 'w') as f:
        json.dump(data, f, indent=indent)


def makedir(path):
    Path(path).mkdir(parents=True, exist_ok=True)