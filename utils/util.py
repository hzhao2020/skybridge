"""
通用工具函数模块
"""
import json
import shutil
from pathlib import Path
from typing import Tuple
from urllib.parse import urlparse


def copy(src_path, dst_path):
    """复制文件"""
    shutil.copy2(src_path, dst_path)


def load_json(fn):
    """加载 JSON 文件"""
    with open(fn, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def save_json(data, fn, indent=4):
    """保存数据到 JSON 文件"""
    with open(fn, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def makedir(path):
    """创建目录（如果不存在）"""
    Path(path).mkdir(parents=True, exist_ok=True)


def parse_uri(uri: str) -> Tuple[str, str, str]:
    """
    解析云存储 URI，返回 scheme, bucket, path
    
    Args:
        uri: 云存储 URI (例如: gs://bucket/path/to/file 或 s3://bucket/path/to/file)
        
    Returns:
        Tuple[str, str, str]: (scheme, bucket, path)
        
    Examples:
        >>> parse_uri("gs://my-bucket/path/to/file.mp4")
        ('gs', 'my-bucket', 'path/to/file.mp4')
        >>> parse_uri("s3://my-bucket/path/to/file.mp4")
        ('s3', 'my-bucket', 'path/to/file.mp4')
    """
    parsed = urlparse(uri)
    return parsed.scheme, parsed.netloc, parsed.path.lstrip('/')


def parse_cloud_uri(uri: str) -> Tuple[str, str]:
    """
    解析云存储 URI，返回 bucket 和 key/blob_path（向后兼容）
    
    Args:
        uri: 云存储 URI
        
    Returns:
        Tuple[str, str]: (bucket, key/blob_path)
    """
    _, bucket, path = parse_uri(uri)
    return bucket, path