# Split & Sample Serverless Node

HTTP contract:

```json
{
  "video": {"name": "input.mp4", "data_b64": "..."},
  "shots": [{"shot_id": "shot-001", "start_ms": 0, "end_ms": 8000, "confidence": 0.98}],
  "samples_per_shot": 3
}
```

Response:

```json
{
  "frames": [
    {
      "frame_id": "frame-001-01",
      "shot_id": "shot-001",
      "timestamp_ms": 2000,
      "media_type": "image/jpeg",
      "data_b64": "...",
      "metadata": {"samples_per_shot": 3}
    }
  ]
}
```

Deployment target: `2 vCPU` and `2 GiB` memory.

