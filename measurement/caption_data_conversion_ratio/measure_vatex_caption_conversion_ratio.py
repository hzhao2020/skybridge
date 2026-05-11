#!/usr/bin/env python3
"""
Measure caption / raw-video byte ratio for VATEX validation clips.

Reads vatex_validation_v1.0.json (list of {videoID, enCap, chCap}).
parses videoID as: <youtube_id>_<start_sec_int>_<end_sec_int>.

For each sampled entry:
  - input size  : local cached clip file bytes (downloads via yt-dlp if missing)
  - output size : UTF-8 byte length of caption text (see --caption-mode)

Conversion ratio := caption_bytes / video_bytes.

Reproducibility: RNG seed fixes which N entries are sampled (deterministic shuffle).
Videos are reused from --cache-dir when already present (*.mp4 with matching stem).

Requires ffmpeg on PATH (`conda activate sky` usually provides it).

Install yt-dlp into the SAME Python env you use to run this script (`python -m pip install -U yt-dlp`).
By default we invoke yt-dlp as `python -m yt_dlp` so PATH cannot accidentally shadow with an old OS binary.

YouTube quirks:
  - "HTTP error 403" from ffmpeg/googlevideo usually means the CDN refused the fetch
    (client fingerprint / headers / signature drift). Prefer a current yt-dlp
    (`python -m pip install -U yt-dlp`) and/or `--cookies-from-browser ...`.
  - Pinning `--youtube-extractor-args` to niche player_client lists can shrink the format table
    and trip "Requested format is not available"; leave it empty unless you need a 403 workaround.
  - "Video unavailable" means the clip no longer exists on YouTube (common for aged datasets).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import shutil
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

# Robust across DASH separated streams + muxed fallback (explicit `bv*+ba/b` often fails).
DEFAULT_YTDLP_FORMAT = "bestvideo*+bestaudio/best/worstvideo+worstaudio/worst/best/worst"
INTERNAL_FORMAT_FALLBACKS = ("bestvideo*+bestaudio/best", "best")


SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--json",
        type=Path,
        default=SCRIPT_DIR / "vatex_validation_v1.0.json",
        help="VATEX validation JSON path",
    )
    p.add_argument(
        "--cache-dir",
        type=Path,
        default=SCRIPT_DIR / "vatex_video_cache",
        help="Directory to store downloaded clips (reused across runs)",
    )
    p.add_argument("-n", type=int, default=100, help="Number of videos to sample")
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling indices (fixes which videos are sampled)",
    )
    p.add_argument(
        "--caption-mode",
        choices=("en", "ch", "both"),
        default="en",
        help="Caption payload: English only, Chinese only, or both (blank line between langs)",
    )
    p.add_argument(
        "--yt-dlp-format",
        default=DEFAULT_YTDLP_FORMAT,
        help=(
            "yt-dlp -f selector (bitrate/resolution affects video_bytes and ratios). "
            f"Built-in retries try {INTERNAL_FORMAT_FALLBACKS!r} if this one is unavailable."
        ),
    )
    p.add_argument(
        "--yt-dlp-mode",
        choices=("module", "executable"),
        default="module",
        help=(
            "How to invoke yt-dlp: `module` runs `%(prog)s`'s interpreter with `-m yt_dlp` "
            "(matches `python -m pip install yt-dlp`); "
            "`executable` uses PATH/--yt-dlp-bin (can accidentally hit an outdated system binary)."
        ),
    )
    p.add_argument(
        "--yt-dlp-bin",
        default="yt-dlp",
        metavar="PATH",
        help="Used only when --yt-dlp-mode=executable: yt-dlp name or absolute path.",
    )
    p.add_argument(
        "--youtube-extractor-args",
        default="",
        metavar="ARGS",
        help=(
            "Optional `yt-dlp --extractor-args` string (empty disables). "
            'Only enable if you hit HTTP 403; forcing mobile clients often breaks `-f bv*+ba`-style merges.'
        ),
    )
    p.add_argument(
        "--no-format-fallback",
        action="store_true",
        help="Do not retry other -f presets when the requested format is unavailable.",
    )
    p.add_argument(
        "--cookies-from-browser",
        default=None,
        metavar="BROWSER",
        help=(
            'If set (e.g. chrome, firefox, chromium), passed to yt-dlp as '
            "`--cookies-from-browser BROWSER` — often fixes 403 when logged-in "
            "web sessions are required."
        ),
    )
    return p.parse_args()


def parse_video_id(video_id: str) -> tuple[str, int, int]:
    parts = video_id.rsplit("_", 2)
    if len(parts) != 3:
        raise ValueError(f"Bad videoID (expected *_start_end): {video_id!r}")
    yt_id, s0, s1 = parts
    return yt_id, int(s0), int(s1)


def caption_utf8_bytes(entry: dict[str, Any], mode: str) -> int:
    if mode == "en":
        text = "\n".join(entry["enCap"])
    elif mode == "ch":
        text = "\n".join(entry["chCap"])
    else:
        text = "\n\n".join(("\n".join(entry["enCap"]), "\n".join(entry["chCap"])))
    return len(text.encode("utf-8"))


def expected_clip_path(cache_dir: Path, video_id: str) -> Path:
    safe = video_id.replace("/", "_")
    return cache_dir / f"{safe}.mp4"


def yt_dlp_cmd_prefix(args: argparse.Namespace) -> list[str]:
    if args.yt_dlp_mode == "module":
        return [sys.executable, "-m", "yt_dlp"]
    return [args.yt_dlp_bin]


def _yt_dlp_version_line(cmd_prefix: Sequence[str]) -> str | None:
    try:
        r = subprocess.run(
            [*cmd_prefix, "--version"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if r.returncode != 0:
            return None
        first = ((r.stdout or r.stderr).strip().split("\n") or [""])[0]
        return first or None
    except OSError:
        return None


def ensure_yt_dlp(cmd_prefix: Sequence[str]) -> None:
    exe = cmd_prefix[0]
    if cmd_prefix[-2:] == ["-m", "yt_dlp"]:
        pass
    else:
        exe_path = Path(exe).expanduser()
        if shutil.which(exe) is None and not exe_path.is_file():
            raise FileNotFoundError(
                f"Cannot run `{exe}` via PATH or as a concrete file — check `--yt-dlp-mode` / `--yt-dlp-bin`."
            )

    ver = _yt_dlp_version_line(cmd_prefix)
    if ver is None:
        proc = subprocess.run(
            [*cmd_prefix, "--version"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        stderr = proc.stderr.strip()[:4000]
        hint = "`python -m pip install -U yt-dlp` in this conda env (`conda activate sky`)."
        if "-m yt_dlp" in " ".join(cmd_prefix):
            raise RuntimeError(f"Unable to invoke yt-dlp as a module. Install yt-dlp: {hint}\n{stderr}")
        raise RuntimeError(
            "Unable to run the configured yt-dlp executable/version probe failed.\n"
            f"{stderr}\n"
            "Tip: rerun with `--yt-dlp-mode module` (recommended) "
            "or inspect `--youtube-extractor-args` narrowing available formats.\n"
            f"If missing package: {hint}"
        )


def _truncate_log(text: str, limit: int = 8000) -> str:
    t = text.strip()
    if len(t) <= limit:
        return t
    digest = hashlib.sha256(t.encode("utf-8", errors="replace")).hexdigest()[:12]
    return t[:limit] + f"\n... [truncated, sha256-prefix={digest}]"


def download_clip(
    out_mp4: Path,
    yt_id: str,
    t0: int,
    t1: int,
    fmt: str,
    yt_dlp_prefix: Sequence[str],
    *,
    youtube_extractor_args: str | None,
    cookies_browser: str | None,
    no_format_fallback: bool,
) -> bool:
    cache_dir = out_mp4.parent
    cache_dir.mkdir(parents=True, exist_ok=True)
    url = f"https://www.youtube.com/watch?v={yt_id}"
    section = f"*{t0}-{t1}"
    template = str(out_mp4.with_suffix(".%(ext)s"))

    fmt_trials: list[str] = []
    for f in (fmt, *(() if no_format_fallback else INTERNAL_FORMAT_FALLBACKS)):
        f = f.strip()
        if f and f not in fmt_trials:
            fmt_trials.append(f)

    last_log = ""
    for attempt_fmt in fmt_trials:
        cmd: list[str] = [
            *yt_dlp_prefix,
            "--no-warnings",
            "--no-playlist",
            "-f",
            attempt_fmt,
            "--download-sections",
            section,
            "--merge-output-format",
            "mp4",
            "-o",
            template,
        ]
        ea = (youtube_extractor_args or "").strip()
        if ea:
            cmd += ["--extractor-args", ea]
        cb = (cookies_browser or "").strip()
        if cb:
            cmd += ["--cookies-from-browser", cb]
        cmd.append(url)

        r = subprocess.run(cmd, capture_output=True, text=True)
        last_log = r.stderr or r.stdout or ""
        if r.returncode == 0 and out_mp4.is_file():
            if attempt_fmt != fmt:
                print(
                    f"[yt-dlp] format fallback succeeded: {fmt!r} -> {attempt_fmt!r} ({yt_id} {t0}-{t1})",
                    flush=True,
                )
            return True

    sys.stderr.write(
        f"[yt-dlp failed] {yt_id} {t0}-{t1} (tried formats {fmt_trials!r})\n"
        f"cmd(last): {_truncate_log(' '.join(cmd), 6000)}\n"
        f"{_truncate_log(last_log)}\n"
    )
    return False


def ensure_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise FileNotFoundError(
            "Cannot find `ffmpeg` on PATH. After `conda activate sky`, ffmpeg is usually available."
        )


def sample_indices(n_total: int, n_sample: int, seed: int) -> list[int]:
    if n_sample > n_total:
        raise ValueError(f"Need at least n_sample={n_sample} entries, JSON has only {n_total}.")
    idx = list(range(n_total))
    rnd = random.Random(seed)
    rnd.shuffle(idx)
    return idx[:n_sample]


def main() -> int:
    args = parse_args()
    prefix = yt_dlp_cmd_prefix(args)
    ensure_yt_dlp(prefix)
    ensure_ffmpeg()
    ver = _yt_dlp_version_line(prefix)
    print(f"yt-dlp: {' '.join(prefix)}  /  {ver or '(version unknown)'}", flush=True)

    if not args.json.is_file():
        print(f"JSON not found: {args.json}", file=sys.stderr)
        return 1

    with args.json.open("r", encoding="utf-8") as f:
        data: list[dict[str, Any]] = json.load(f)

    n_total = len(data)
    picked = sample_indices(n_total, args.n, args.seed)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loaded {n_total} entries; will process {len(picked)} (seed={args.seed}).", flush=True)

    ratios: list[float] = []
    rows: list[str] = []

    for rank, i in enumerate(picked, start=1):
        entry = data[i]
        video_id = entry["videoID"]
        try:
            yt_id, t0, t1 = parse_video_id(video_id)
        except ValueError as e:
            sys.stderr.write(f"[skip] {video_id}: {e}\n")
            continue

        cap_b = caption_utf8_bytes(entry, args.caption_mode)
        out_path = expected_clip_path(args.cache_dir, video_id)

        if not out_path.is_file():
            print(f"[download] ({rank}/{args.n}) {video_id}", flush=True)
            ok = download_clip(
                out_path,
                yt_id,
                t0,
                t1,
                args.yt_dlp_format,
                prefix,
                youtube_extractor_args=args.youtube_extractor_args,
                cookies_browser=args.cookies_from_browser,
                no_format_fallback=args.no_format_fallback,
            )
            if not ok:
                continue

        vid_b = out_path.stat().st_size
        if vid_b <= 0:
            sys.stderr.write(f"[skip] zero-size video: {out_path}\n")
            continue

        r = cap_b / vid_b
        ratios.append(r)
        rows.append(
            f"{rank:3d}  idx={i:4d}  videoID={video_id}  cap={cap_b:6d}B  vid={vid_b:10d}B  ratio={r:.8e}"
        )

    ok_n = len(ratios)
    print(f"Sampled (seed={args.seed}, n_requested={args.n}, ok={ok_n} / total_json={n_total})")
    print(f"Caption mode={args.caption_mode!r}  cache_dir={args.cache_dir}")
    print()

    if ok_n == 0:
        print(
            "No successful downloads / ratios.\n"
            "- HTTP 403 on googlevideo: upgrade yt-dlp + try `--cookies-from-browser ...`.\n"
            "- 'Requested format is not available': clear `--youtube-extractor-args` (empty) and/or "
            "use `--yt-dlp-mode module` with the patched defaults (`-f .../best`).\n"
            "- 'Video unavailable': offline sources — sweep more `-n`/seeds.",
            flush=True,
        )
        return 2

    for line in rows:
        print(line)
    print()

    mean_v = statistics.fmean(ratios)
    if ok_n >= 2:
        std_sample = statistics.stdev(ratios)
    else:
        std_sample = float("nan")

    print(f"mean(ratio): {mean_v:.8e}")
    print(f"std(ratio):  {std_sample if not math.isnan(std_sample) else 'nan (need >=2 OK samples)'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
