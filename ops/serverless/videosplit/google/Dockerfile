FROM python:3.11-slim

# 安装 ffmpeg
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

ENV PORT=8080
EXPOSE 8080

# Cloud Run: 使用 gunicorn 提供 HTTP 服务
CMD ["sh", "-c", "gunicorn -b :${PORT} main:app --workers 1 --threads 8 --timeout 0"]

