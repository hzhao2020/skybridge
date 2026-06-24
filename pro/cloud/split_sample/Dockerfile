FROM python:3.11-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY app.py /app/app.py

ENV SAMPLES_PER_SHOT=3
ENV PORT=8080

CMD ["python", "app.py"]

