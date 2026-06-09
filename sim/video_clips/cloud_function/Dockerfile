FROM python:3.11-slim

# apt 安装 ffmpeg（含 AV1/H.264 解码），满足「容器 + apt ffmpeg」方案
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY main.py .

ENV PORT=8080
# functions-framework 以 HTTP 方式暴露 split_and_sample
CMD exec functions-framework --target=split_and_sample --port=${PORT}
