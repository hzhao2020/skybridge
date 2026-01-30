import requests
import time


# 1. 纯净的容器基础地址 (不要带任何问号或参数)
BASE_URL = "https://videoea.blob.core.windows.net/video-ea"

# 2. 具体的视频文件名
video_filename = "0a8b2c9d-b54c-4811-acf3-5977895d2445.mp4"

# 3. 你的 SAS 令牌 (确保开头有问号 ?)
# 注意：如果你复制的令牌没有问号，请手动加一个 "?"
SAS_TOKEN = "?sp=racw&st=2026-01-28T06:17:32Z&se=2026-01-30T14:32:32Z&sv=2024-11-04&sr=c&sig=tEhcn16m8wEm%2BkAaLlx%2BQBKmPpEAbqIzo%2Bff%2Fskyb2o%3D"

# 4. 正确的拼接顺序：基础地址 + / + 文件名 + SAS令牌
VIDEO_URL = f"{BASE_URL}/{video_filename}{SAS_TOKEN}"

# print(f"最终生成的可用 URL: {VIDEO_URL}")

# Blob SAS 令牌
# sp=racw&st=2026-01-28T06:17:32Z&se=2026-01-30T14:32:32Z&sv=2024-11-04&sr=c&sig=tEhcn16m8wEm%2BkAaLlx%2BQBKmPpEAbqIzo%2Bff%2Fskyb2o%3D
# Blob SAS URL
# https://videoea.blob.core.windows.net/video-ea?sp=racw&st=2026-01-28T06:17:32Z&se=2026-01-30T14:32:32Z&sv=2024-11-04&sr=c&sig=tEhcn16m8wEm%2BkAaLlx%2BQBKmPpEAbqIzo%2Bff%2Fskyb2o%3D


API_KEY = '8827883d00b84cbf8d615e901a5901fa'
ACCOUNT_ID = '4a4d662f-d0b3-4800-9d50-fc82945d1f59'
LOCATION = 'trial'  # 如果是付费版，请填写对应区域，如 'eastus'
# VIDEO_URL = "https://videoea.blob.core.windows.net/video-ea/0a8b2c9d-b54c-4811-acf3-5977895d2445.mp4"
API_URL = "https://api.videoindexer.ai"

def get_access_token():
    """获取 API 访问令牌"""
    headers = {
        'Ocp-Apim-Subscription-Key': API_KEY,
    }
    # 获取 Account Access Token
    url = f"{API_URL}/auth/{LOCATION}/Accounts/{ACCOUNT_ID}/AccessToken?allowEdit=true"
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return response.text.strip('"')

def upload_video_from_url(video_url, token):
    """提交视频 URL 进行异步分析"""
    params = {
        'name': 'MyAnalysisVideo',
        'videoUrl': video_url,
        'privacy': 'Private',
        'accessToken': token,
        'language': 'zh-CN' # 可选，指定视频主语言
    }
    upload_url = f"{API_URL}/{LOCATION}/Accounts/{ACCOUNT_ID}/Videos"
    
    print("正在提交视频分析请求...")
    response = requests.post(upload_url, params=params)
    response.raise_for_status()
    return response.json()['id']

def get_video_insights(video_id, token):
    """查询分析结果"""
    params = {
        'accessToken': token,
        'language': 'zh-CN'
    }
    url = f"{API_URL}/{LOCATION}/Accounts/{ACCOUNT_ID}/Videos/{video_id}/Index"
    
    while True:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        # 检查处理状态
        state = data.get('state')
        if state == 'Processed':
            print("\n分析完成！")
            return data
        elif state == 'Failed':
            raise Exception("分析失败")
        else:
            progress = data.get('videos', [{}])[0].get('processingProgress', '0%')
            print(f"处理中... 当前进度: {progress}", end='\r')
            time.sleep(10) # 每10秒轮询一次


import requests
import json

# 配置同上
API_KEY = '8827883d00b84cbf8d615e901a5901fa'
ACCOUNT_ID = '4a4d662f-d0b3-4800-9d50-fc82945d1f59'
LOCATION = 'trial'

def get_latest_video_insights():
    # 1. 获取 Token
    token_url = f"https://api.videoindexer.ai/auth/{LOCATION}/Accounts/{ACCOUNT_ID}/AccessToken?allowEdit=true"
    token = requests.get(token_url, headers={'Ocp-Apim-Subscription-Key': API_KEY}).text.strip('"')

    # 2. 获取账号下的视频列表，找到最近的一个
    list_url = f"https://api.videoindexer.ai/{LOCATION}/Accounts/{ACCOUNT_ID}/Videos"
    videos = requests.get(list_url, params={'accessToken': token}).json()
    
    if not videos.get('results'):
        print("未找到任何视频")
        return

    latest_video_id = videos['results'][0]['id']
    print(f"检测到最近的视频 ID: {latest_video_id}")

    # 3. 获取完整结果
    index_url = f"https://api.videoindexer.ai/{LOCATION}/Accounts/{ACCOUNT_ID}/Videos/{latest_video_id}/Index"
    insights = requests.get(index_url, params={'accessToken': token}).json()
    
    # 漂亮地打印出完整的 JSON
    print(json.dumps(insights, indent=4, ensure_ascii=False))

get_latest_video_insights()


# --- 执行主流程 ---
# try:
#     token = get_access_token()
#     video_id = upload_video_from_url(VIDEO_URL, token)
#     print(f"视频上传成功，Video ID: {video_id}")
    
#     # 获取结果（这是一个耗时过程，取决于视频大小）
#     insights = get_video_insights(video_id, token)
    
#     # 打印关键分析结果（如关键词、人脸、标签等）
#     # print("--- 分析摘要 ---")
#     # summarized_insights = insights.get('summarizedInsights', {})
#     # print(f"关键词: {', '.join([k['name'] for k in summarized_insights.get('keywords', [])])}")
#     # print(f"检测到的标签: {', '.join([l['name'] for l in summarized_insights.get('labels', [])])}")
#     print(insights)


# except Exception as e:
#     print(f"发生错误: {e}")