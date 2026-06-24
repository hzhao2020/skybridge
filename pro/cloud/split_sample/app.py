from __future__ import annotations

import base64
import json
import os
import subprocess
import tempfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


DEFAULT_SAMPLES_PER_SHOT = int(os.environ.get("SAMPLES_PER_SHOT", "3"))
JPEG_QUALITY = int(os.environ.get("JPEG_QUALITY", "2"))


def split_sample_payload(payload: dict[str, Any]) -> dict[str, Any]:
    video = payload["video"]
    shots = payload.get("shots", [])
    samples_per_shot = int(payload.get("samples_per_shot", DEFAULT_SAMPLES_PER_SHOT))
    samples_per_shot = max(1, samples_per_shot)

    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        input_path = tmpdir / _safe_name(video.get("name", "input.mp4"))
        input_path.write_bytes(base64.b64decode(video["data_b64"]))

        frames: list[dict[str, Any]] = []
        for shot_index, shot in enumerate(shots, start=1):
            start_ms = int(shot["start_ms"])
            end_ms = int(shot["end_ms"])
            duration_ms = max(1, end_ms - start_ms)
            shot_id = str(shot.get("shot_id", f"shot-{shot_index:03d}"))

            for sample_index in range(samples_per_shot):
                timestamp_ms = start_ms + duration_ms * (sample_index + 1) // (samples_per_shot + 1)
                frame_id = f"frame-{shot_index:03d}-{sample_index + 1:02d}"
                frame_path = tmpdir / f"{frame_id}.jpg"
                _extract_jpeg(input_path, frame_path, timestamp_ms)
                frames.append(
                    {
                        "frame_id": frame_id,
                        "shot_id": shot_id,
                        "timestamp_ms": timestamp_ms,
                        "media_type": "image/jpeg",
                        "data_b64": base64.b64encode(frame_path.read_bytes()).decode("ascii"),
                        "metadata": {
                            "samples_per_shot": samples_per_shot,
                            "source_video": video.get("name", "input.mp4"),
                        },
                    }
                )
        return {"frames": frames}


def _extract_jpeg(input_path: Path, frame_path: Path, timestamp_ms: int) -> None:
    timestamp_sec = max(0.0, timestamp_ms / 1000.0)
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{timestamp_sec:.3f}",
        "-i",
        str(input_path),
        "-frames:v",
        "1",
        "-q:v",
        str(JPEG_QUALITY),
        "-y",
        str(frame_path),
    ]
    subprocess.run(cmd, check=True)


def _safe_name(name: str) -> str:
    cleaned = Path(name).name
    return cleaned or "input.mp4"


class Handler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            response = split_sample_payload(payload)
            body = json.dumps(response).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except Exception as exc:  # noqa: BLE001 - return JSON error to caller
            body = json.dumps({"error": f"{type(exc).__name__}: {exc}"}).encode("utf-8")
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        body = b'{"status":"ok","node":"split_sample"}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args: Any) -> None:
        print(fmt % args)


def main() -> None:
    port = int(os.environ.get("PORT", "8080"))
    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    print(f"split_sample listening on :{port}")
    server.serve_forever()


if __name__ == "__main__":
    main()

