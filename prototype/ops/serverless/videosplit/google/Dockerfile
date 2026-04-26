FROM python:3.11-slim

# 安装 ffmpeg
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

ENV PORT=8080
EXPOSE 8080

# Cloud Function: 使用 functions-framework 运行
CMD exec functions-framework --target=video_split --port=${PORT} --host=0.0.0.0

