"""
Storage 和 Transmission Operation 使用示例

演示如何使用独立的 Storage 和 Transmission operations
"""

from ops.registry import get_operation, list_available_ops


def example_storage_operations():
    """演示存储操作的使用"""
    
    print("=== Storage Operations 示例 ===\n")
    
    # 1. 上传文件
    print("1. 上传文件到 Google Cloud Storage")
    storage_op = get_operation("storage_google_us")
    
    # 假设有一个本地文件
    local_file = "/path/to/video.mp4"  # 替换为实际路径
    
    try:
        result = storage_op.execute(
            operation="upload",
            local_path=local_file,
            target_path="videos/"
        )
        print(f"   上传成功: {result['cloud_uri']}\n")
    except FileNotFoundError:
        print(f"   文件不存在: {local_file}\n")
    
    # 2. 列出文件
    print("2. 列出 GCS bucket 中的文件")
    try:
        result = storage_op.execute(
            operation="list",
            cloud_uri=f"gs://{storage_op.storage_bucket}/videos/",
            prefix="videos/"
        )
        print(f"   找到 {result['count']} 个文件")
        for file_uri in result['files'][:5]:  # 只显示前5个
            print(f"     - {file_uri}")
        print()
    except Exception as e:
        print(f"   错误: {e}\n")
    
    # 3. 下载文件
    print("3. 从 S3 下载文件")
    aws_storage_op = get_operation("storage_aws_us")
    
    try:
        result = aws_storage_op.execute(
            operation="download",
            cloud_uri="s3://bucket/path/to/file.mp4",
            local_path="/tmp/downloaded_file.mp4"
        )
        print(f"   下载成功: {result['local_path']}\n")
    except Exception as e:
        print(f"   错误: {e}\n")
    
    # 4. 删除文件
    print("4. 删除云存储文件")
    try:
        result = storage_op.execute(
            operation="delete",
            cloud_uri="gs://bucket/path/to/file.mp4"
        )
        print(f"   删除成功: {result['success']}\n")
    except Exception as e:
        print(f"   错误: {e}\n")


def example_transmission_operations():
    """演示传输操作的使用"""
    
    print("=== Transmission Operations 示例 ===\n")
    
    # 1. 智能传输（自动判断是否需要传输）
    print("1. 智能传输（从 S3 到 GCS）")
    transmission_op = get_operation("transmission_google_us")
    
    try:
        result = transmission_op.execute(
            source_uri="s3://source-bucket/video.mp4",
            target_provider="google",
            target_bucket="target-bucket",
            target_path="videos/"
        )
        print(f"   源: {result['source_uri']}")
        print(f"   目标: {result['target_uri']}")
        print(f"   是否传输: {result['transferred']}\n")
    except Exception as e:
        print(f"   错误: {e}\n")
    
    # 2. S3 -> GCS 专用传输
    print("2. S3 -> GCS 专用传输")
    s3_to_gcs_op = get_operation("transmission_s3_to_gcs_us")
    
    try:
        result = s3_to_gcs_op.execute(
            source_uri="s3://source-bucket/video.mp4",
            target_bucket="target-gcs-bucket",
            target_path="videos/"
        )
        print(f"   传输完成: {result['target_uri']}\n")
    except Exception as e:
        print(f"   错误: {e}\n")
    
    # 3. GCS -> S3 专用传输
    print("3. GCS -> S3 专用传输")
    gcs_to_s3_op = get_operation("transmission_gcs_to_s3_us")
    
    try:
        result = gcs_to_s3_op.execute(
            source_uri="gs://source-bucket/video.mp4",
            target_bucket="target-s3-bucket",
            target_path="videos/"
        )
        print(f"   传输完成: {result['target_uri']}\n")
    except Exception as e:
        print(f"   错误: {e}\n")


def example_combined_workflow():
    """演示组合使用 Storage 和 Transmission"""
    
    print("=== 组合工作流示例 ===\n")
    
    # 场景：从本地文件上传到 GCS，然后传输到 S3
    print("场景：本地文件 -> GCS -> S3")
    
    # 步骤1：上传到 GCS
    print("步骤1: 上传到 GCS")
    storage_op = get_operation("storage_google_us")
    
    try:
        upload_result = storage_op.execute(
            operation="upload",
            local_path="/path/to/video.mp4",
            target_path="temp/"
        )
        gcs_uri = upload_result['cloud_uri']
        print(f"   GCS URI: {gcs_uri}\n")
        
        # 步骤2：传输到 S3
        print("步骤2: 传输到 S3")
        transmission_op = get_operation("transmission_gcs_to_s3_us")
        
        transfer_result = transmission_op.execute(
            source_uri=gcs_uri,
            target_bucket="target-s3-bucket",
            target_path="videos/"
        )
        print(f"   S3 URI: {transfer_result['target_uri']}\n")
        
    except Exception as e:
        print(f"   错误: {e}\n")


def list_all_storage_transmission_ops():
    """列出所有可用的 Storage 和 Transmission operations"""
    
    print("=== 可用的 Storage Operations ===")
    storage_pids = [
        "storage_google_us", "storage_google_eu", "storage_google_sg", "storage_google_tw",
        "storage_aws_us", "storage_aws_eu", "storage_aws_sg"
    ]
    
    for pid in storage_pids:
        try:
            op = get_operation(pid)
            print(f"  {pid}: provider={op.provider}, region={op.region}, bucket={op.storage_bucket}")
        except Exception as e:
            print(f"  {pid}: 不可用 ({e})")
    
    print("\n=== 可用的 Transmission Operations ===")
    transmission_pids = [
        "transmission_google_us", "transmission_google_eu",
        "transmission_aws_us", "transmission_aws_eu",
        "transmission_s3_to_gcs_us", "transmission_s3_to_gcs_eu",
        "transmission_gcs_to_s3_us", "transmission_gcs_to_s3_eu"
    ]
    
    for pid in transmission_pids:
        try:
            op = get_operation(pid)
            print(f"  {pid}: provider={op.provider}, region={op.region}, bucket={op.storage_bucket}")
        except Exception as e:
            print(f"  {pid}: 不可用 ({e})")


if __name__ == "__main__":
    print("Storage 和 Transmission Operations 使用示例\n")
    print("=" * 60 + "\n")
    
    # 列出所有可用的操作
    list_all_storage_transmission_ops()
    print("\n" + "=" * 60 + "\n")
    
    # 演示存储操作
    example_storage_operations()
    
    # 演示传输操作
    example_transmission_operations()
    
    # 演示组合使用
    example_combined_workflow()
    
    print("\n注意：这些示例需要配置好云服务凭证才能实际运行")
