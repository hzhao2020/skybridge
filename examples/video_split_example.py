"""
视频分割使用示例

演示如何使用 VideoSplitter 将视频切割成多个片段
"""

from ops.registry import get_operation

def example_google_cloud_function_split():
    """使用 Google Cloud Functions 进行视频分割"""
    
    # 获取操作实例
    split_op = get_operation("split_google_us")
    
    # 定义要切割的片段
    segments = [
        {"start": 0.0, "end": 10.0},
        {"start": 10.0, "end": 20.0},
        {"start": 20.0, "end": 30.0},
    ]
    
    # 执行分割
    video_uri = "gs://your-bucket/videos/sample.mp4"
    result = split_op.execute(
        video_uri,
        segments=segments,
        target_path="videos/split_output",
        output_format="mp4"
    )
    
    print("分割结果：")
    print(f"  输入视频: {result['input_video']}")
    print(f"  输出片段数量: {result['output_count']}")
    print(f"  输出 URI 列表:")
    for uri in result['output_uris']:
        print(f"    - {uri}")


def example_aws_lambda_split():
    """使用 AWS Lambda 进行视频分割"""
    
    # 获取操作实例
    split_op = get_operation("split_aws_us")
    
    # 定义要切割的片段
    segments = [
        {"start": 0.0, "end": 15.0},
        {"start": 15.0, "end": 30.0},
    ]
    
    # 执行分割
    video_uri = "s3://your-bucket/videos/sample.mp4"
    result = split_op.execute(
        video_uri,
        segments=segments,
        target_path="videos/split_output",
        output_format="mp4"
    )
    
    print("分割结果：")
    print(f"  输入视频: {result['input_video']}")
    print(f"  输出片段数量: {result['output_count']}")
    print(f"  输出 URI 列表:")
    for uri in result['output_uris']:
        print(f"    - {uri}")


def example_with_custom_function_url():
    """使用自定义函数 URL"""
    
    # 创建操作实例（需要先部署 Cloud Function）
    from ops.impl.google_ops import GoogleCloudFunctionSplitImpl
    
    # 使用自定义函数 URL
    custom_url = "https://us-west1-YOUR_PROJECT_ID.cloudfunctions.net/video-split-custom"
    split_op = GoogleCloudFunctionSplitImpl(
        provider="google",
        region="us-west1",
        storage_bucket="your-bucket",
        function_url=custom_url
    )
    
    segments = [{"start": 0.0, "end": 10.0}]
    result = split_op.execute(
        "gs://your-bucket/video.mp4",
        segments=segments
    )
    
    print(result)


if __name__ == "__main__":
    print("=== Google Cloud Functions 示例 ===")
    # example_google_cloud_function_split()
    
    print("\n=== AWS Lambda 示例 ===")
    # example_aws_lambda_split()
    
    print("\n注意：需要先部署对应的 Cloud Function 或 Lambda 函数")
    print("参考 cloud_functions/ 和 lambda_functions/ 目录下的部署说明")
