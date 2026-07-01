#!/usr/bin/env python3
"""Collect round-aligned local-relay <-> bucket transfer scenarios.

The raw CSV is the primary output.  Each round contains one upload and one
download observation for every selected (bucket, payload size) pair, which
allows the optimizer to resample complete rounds and retain shared local-link
conditions across destinations.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import random
import shutil
import statistics
import subprocess
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


DEFAULT_SIZES_MIB = (1, 5, 10, 20, 40, 60, 80, 100)
NETWORK_MODE = "system"
GCP_ACCESS_TOKEN = ""
GCP_TOKEN_REFRESHED_AT = 0.0
MEASUREMENT_FIELDS = [
    "trial_id",
    "round",
    "order_in_round",
    "timestamp_start_utc",
    "timestamp_end_utc",
    "node_id",
    "provider",
    "region",
    "bucket",
    "direction",
    "size_mib",
    "size_bytes",
    "elapsed_sec",
    "throughput_mib_per_sec",
    "throughput_mbps",
    "status",
    "verified_size_bytes",
    "object_uri",
    "error_type",
    "error_message",
    "cleanup_error",
]
PROBE_FIELDS = [
    "probe_id",
    "round",
    "order_in_round",
    "timestamp_start_utc",
    "timestamp_end_utc",
    "node_id",
    "provider",
    "region",
    "bucket",
    "operation",
    "elapsed_sec",
    "status",
    "verified_size_bytes",
    "object_uri",
    "error_type",
    "error_message",
]
SUMMARY_FIELDS = [
    "node_id",
    "provider",
    "region",
    "direction",
    "size_mib",
    "sample_count",
    "success_count",
    "failure_count",
    "failure_rate",
    "elapsed_sec_mean",
    "elapsed_sec_std",
    "elapsed_sec_p50",
    "elapsed_sec_p90",
    "elapsed_sec_p95",
    "elapsed_sec_min",
    "elapsed_sec_max",
    "throughput_mib_per_sec_mean",
    "throughput_mib_per_sec_p10",
    "throughput_mib_per_sec_p50",
]


@dataclass(frozen=True)
class Node:
    node_id: str
    provider: str
    region: str
    bucket: str


@dataclass(frozen=True)
class TransferTask:
    round_id: int
    node: Node
    direction: str
    size_mib: int

    @property
    def trial_id(self) -> str:
        return f"r{self.round_id:03d}-{self.node.node_id}-{self.direction}-{self.size_mib:03d}mib"


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(value, file, indent=2)
        file.write("\n")


def append_csv(path: Path, fields: list[str], row: dict[str, Any]) -> None:
    new_file = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        if new_file:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in fields})


def load_nodes(path: Path, requested_nodes: str) -> list[Node]:
    requested = {item.strip() for item in requested_nodes.split(",") if item.strip()}
    nodes = []
    for provider, regions in load_json(path)["buckets"].items():
        for region, bucket in regions.items():
            node_id = f"{provider}_{region.replace('-', '_')}"
            if requested and node_id not in requested:
                continue
            nodes.append(Node(node_id, provider, region, bucket))
    missing = requested - {node.node_id for node in nodes}
    if missing:
        raise SystemExit(f"Unknown node ids: {', '.join(sorted(missing))}")
    if not nodes:
        raise SystemExit("No bucket nodes selected")
    return nodes


def parse_sizes(value: str) -> list[int]:
    sizes = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not sizes or any(size <= 0 for size in sizes):
        raise argparse.ArgumentTypeError("sizes must be positive comma-separated MiB integers")
    if len(set(sizes)) != len(sizes):
        raise argparse.ArgumentTypeError("sizes must not contain duplicates")
    return sizes


def object_uri(node: Node, key: str) -> str:
    scheme = {"gcp": "gs", "aws": "s3", "aliyun": "oss"}[node.provider]
    return f"{scheme}://{node.bucket}/{key}"


def aliyun_common(node: Node, timeout_sec: int) -> list[str]:
    return [
        "--region",
        node.region,
        "--endpoint",
        f"oss-{node.region}.aliyuncs.com",
        "--connect-timeout",
        "30",
        "--read-timeout",
        str(timeout_sec),
        "--retry-times",
        "1",
    ]


def provider_cli(provider: str) -> str:
    executable = {"gcp": "gcloud", "aws": "aws", "aliyun": "aliyun"}[provider]
    return find_executable(executable) or executable


def upload_command(node: Node, source: Path, uri: str, timeout_sec: int) -> list[str]:
    if node.provider == "gcp":
        return [provider_cli("gcp"), "storage", "cp", "--quiet", str(source), uri]
    if node.provider == "aws":
        return [
            provider_cli("aws"),
            "s3",
            "cp",
            str(source),
            uri,
            "--region",
            node.region,
            "--only-show-errors",
            "--no-progress",
        ]
    return [provider_cli("aliyun"), "oss", "cp", str(source), uri, *aliyun_common(node, timeout_sec), "--force"]


def download_command(node: Node, uri: str, target: Path, timeout_sec: int) -> list[str]:
    if node.provider == "gcp":
        return [provider_cli("gcp"), "storage", "cp", "--quiet", uri, str(target)]
    if node.provider == "aws":
        return [
            provider_cli("aws"),
            "s3",
            "cp",
            uri,
            str(target),
            "--region",
            node.region,
            "--only-show-errors",
            "--no-progress",
        ]
    return [provider_cli("aliyun"), "oss", "cp", uri, str(target), *aliyun_common(node, timeout_sec), "--force"]


def stat_command(node: Node, uri: str, timeout_sec: int) -> list[str]:
    if node.provider == "gcp":
        return [provider_cli("gcp"), "storage", "objects", "describe", uri, "--format=value(size)"]
    if node.provider == "aws":
        key = uri.split("/", 3)[3]
        return [
            provider_cli("aws"),
            "s3api",
            "head-object",
            "--bucket",
            node.bucket,
            "--key",
            key,
            "--region",
            node.region,
            "--query",
            "ContentLength",
            "--output",
            "text",
        ]
    return [provider_cli("aliyun"), "oss", "stat", uri, *aliyun_common(node, timeout_sec)]


def delete_command(node: Node, uri: str, timeout_sec: int) -> list[str]:
    if node.provider == "gcp":
        return [provider_cli("gcp"), "storage", "rm", "--quiet", uri]
    if node.provider == "aws":
        return [provider_cli("aws"), "s3", "rm", uri, "--region", node.region, "--only-show-errors"]
    return [provider_cli("aliyun"), "oss", "rm", uri, *aliyun_common(node, timeout_sec), "--force"]


def refresh_gcp_access_token(*, force: bool = False) -> None:
    global GCP_ACCESS_TOKEN, GCP_TOKEN_REFRESHED_AT
    if NETWORK_MODE != "direct":
        return
    if not force and GCP_ACCESS_TOKEN and time.monotonic() - GCP_TOKEN_REFRESHED_AT < 1200:
        return
    environment = os.environ.copy()
    environment.pop("CLOUDSDK_AUTH_ACCESS_TOKEN", None)
    detail = ""
    for attempt in range(1, 6):
        try:
            process = subprocess.run(
                [provider_cli("gcp"), "auth", "print-access-token"],
                text=True,
                capture_output=True,
                timeout=120,
                env=environment,
            )
            if process.returncode == 0 and process.stdout.strip():
                GCP_ACCESS_TOKEN = process.stdout.strip()
                GCP_TOKEN_REFRESHED_AT = time.monotonic()
                return
            detail = (process.stderr or process.stdout).strip()
        except subprocess.TimeoutExpired as exc:
            detail = str(exc)
        if attempt < 5:
            time.sleep(min(30, 2 ** (attempt - 1)))
    raise RuntimeError(f"Could not refresh GCP token after 5 attempts: {detail[-2000:]}")


def prepare_network_environment(node: Node) -> None:
    if NETWORK_MODE == "direct" and node.provider == "gcp":
        refresh_gcp_access_token()


def command_environment(aws_max_attempts: int = 1) -> dict[str, str]:
    environment = os.environ.copy()
    # Measured transfers use one SDK attempt so timeout/failure remains an
    # observable stochastic outcome. Reference preparation explicitly raises
    # this value because setup reliability is not part of the sampled profile.
    environment["AWS_MAX_ATTEMPTS"] = str(aws_max_attempts)
    environment["AWS_RETRY_MODE"] = "standard"
    if NETWORK_MODE == "direct":
        for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
            environment[name] = ""
        environment["NO_PROXY"] = "*"
        environment["no_proxy"] = "*"
        if GCP_ACCESS_TOKEN:
            environment["CLOUDSDK_AUTH_ACCESS_TOKEN"] = GCP_ACCESS_TOKEN
    return environment


def run_command(
    command: list[str],
    timeout_sec: int,
    *,
    aws_max_attempts: int = 1,
) -> tuple[float, str, str]:
    started = time.perf_counter()
    process = subprocess.run(
        command,
        text=True,
        capture_output=True,
        timeout=timeout_sec,
        env=command_environment(aws_max_attempts),
    )
    elapsed = time.perf_counter() - started
    if process.returncode:
        detail = (process.stderr or process.stdout).strip()
        raise RuntimeError(f"command exited {process.returncode}: {detail[-2000:]}")
    return elapsed, process.stdout, process.stderr


def run_node_command(
    node: Node,
    command: list[str],
    timeout_sec: int,
    *,
    aws_max_attempts: int = 1,
) -> tuple[float, str, str]:
    prepare_network_environment(node)
    try:
        return run_command(
            command,
            timeout_sec,
            aws_max_attempts=aws_max_attempts,
        )
    except RuntimeError as exc:
        message = str(exc)
        if (
            NETWORK_MODE == "direct"
            and node.provider == "gcp"
            and ("HTTPError 401" in message or "Invalid Credentials" in message)
        ):
            refresh_gcp_access_token(force=True)
            return run_command(
                command,
                timeout_sec,
                aws_max_attempts=aws_max_attempts,
            )
        raise


def parse_remote_size(node: Node, stdout: str) -> int:
    if node.provider in {"gcp", "aws"}:
        return int(stdout.strip())
    for line in stdout.splitlines():
        normalized = line.strip().lower()
        if normalized.startswith(("content-length", "size")):
            return int(line.split(":", 1)[1].strip())
    raise ValueError(f"Could not parse OSS object size: {stdout[-1000:]}")


def remote_size(node: Node, uri: str, timeout_sec: int, *, aws_max_attempts: int = 1) -> int:
    _, stdout, _ = run_node_command(
        node,
        stat_command(node, uri, timeout_sec),
        timeout_sec,
        aws_max_attempts=aws_max_attempts,
    )
    return parse_remote_size(node, stdout)


def payload_path(payload_dir: Path, size_mib: int) -> Path:
    return payload_dir / f"payload_{size_mib:03d}mib.bin"


def ensure_payload(path: Path, size_mib: int) -> None:
    expected = size_mib * 1024 * 1024
    if path.is_file() and path.stat().st_size == expected:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    rng = random.Random(0x5A17 + size_mib)
    block = rng.randbytes(1024 * 1024)
    remaining = expected
    with temporary.open("wb") as file:
        while remaining:
            chunk = block[: min(len(block), remaining)]
            file.write(chunk)
            remaining -= len(chunk)
    temporary.replace(path)


def reference_key(size_mib: int) -> str:
    return f"network_measurement/reference/payload_{size_mib:03d}mib.bin"


def find_executable(name: str) -> str:
    found = shutil.which(name)
    if found:
        return found
    if name == "aliyun":
        local_app_data = os.environ.get("LOCALAPPDATA")
        if local_app_data:
            package_root = Path(local_app_data) / "Microsoft" / "WinGet" / "Packages"
            matches = sorted(package_root.glob("Alibaba.AlibabaCloudCLI*/**/aliyun.exe"))
            if matches:
                executable = matches[-1].resolve()
                os.environ["PATH"] = f"{executable.parent}{os.pathsep}{os.environ.get('PATH', '')}"
                return str(executable)
    return ""


def preflight(nodes: list[Node], *, strict: bool = True) -> dict[str, str]:
    required = sorted({node.provider for node in nodes})
    executables = {"gcp": "gcloud", "aws": "aws", "aliyun": "aliyun"}
    found = {provider: find_executable(executables[provider]) for provider in required}
    missing = [provider for provider, path in found.items() if not path]
    if missing and strict:
        details = ", ".join(f"{provider} ({executables[provider]})" for provider in missing)
        raise SystemExit(f"Missing required cloud CLI: {details}")
    return found


def prepare_references(
    nodes: list[Node],
    sizes_mib: list[int],
    payload_dir: Path,
    timeout_sec: int,
    attempts: int,
) -> None:
    total = len(nodes) * len(sizes_mib)
    index = 0
    for node in nodes:
        for size_mib in sizes_mib:
            prepare_network_environment(node)
            index += 1
            source = payload_path(payload_dir, size_mib)
            expected = size_mib * 1024 * 1024
            uri = object_uri(node, reference_key(size_mib))
            try:
                if remote_size(node, uri, timeout_sec, aws_max_attempts=5) == expected:
                    print(f"[prepare {index}/{total}] exists {node.node_id} {size_mib} MiB", flush=True)
                    continue
            except Exception:
                pass
            last_error: Exception | None = None
            for attempt in range(1, attempts + 1):
                print(
                    f"[prepare {index}/{total}] upload {node.node_id} {size_mib} MiB "
                    f"(attempt {attempt}/{attempts})",
                    flush=True,
                )
                try:
                    run_command(
                        upload_command(node, source, uri, timeout_sec),
                        timeout_sec,
                        aws_max_attempts=5,
                    )
                    actual = remote_size(node, uri, timeout_sec, aws_max_attempts=5)
                    if actual != expected:
                        raise RuntimeError(
                            f"Reference size mismatch for {uri}: expected={expected}, actual={actual}"
                        )
                    last_error = None
                    break
                except Exception as exc:  # noqa: BLE001 - preparation is retried outside measured samples.
                    last_error = exc
                    if attempt < attempts:
                        delay = min(30, 2 ** (attempt - 1))
                        print(
                            f"[prepare] warning: {node.node_id} {size_mib} MiB failed: {exc}; "
                            f"retrying in {delay}s",
                            flush=True,
                        )
                        time.sleep(delay)
            if last_error is not None:
                raise RuntimeError(
                    f"Reference preparation failed after {attempts} attempts for {uri}"
                ) from last_error


def completed_ids(path: Path, id_field: str) -> set[str]:
    if not path.exists():
        return set()
    with path.open(encoding="utf-8-sig") as file:
        return {row[id_field] for row in csv.DictReader(file)}


def measure_probe(
    node: Node,
    round_id: int,
    order: int,
    probe_size_mib: int,
    timeout_sec: int,
) -> dict[str, Any]:
    prepare_network_environment(node)
    uri = object_uri(node, reference_key(probe_size_mib))
    row: dict[str, Any] = {
        "probe_id": f"r{round_id:03d}-{node.node_id}-metadata",
        "round": round_id,
        "order_in_round": order,
        "timestamp_start_utc": iso_now(),
        "node_id": node.node_id,
        "provider": node.provider,
        "region": node.region,
        "bucket": node.bucket,
        "operation": "object_metadata",
        "object_uri": uri,
    }
    try:
        elapsed, stdout, _ = run_node_command(
            node,
            stat_command(node, uri, timeout_sec),
            timeout_sec,
        )
        row.update(
            {
                "elapsed_sec": elapsed,
                "status": "success",
                "verified_size_bytes": parse_remote_size(node, stdout),
            }
        )
    except Exception as exc:  # noqa: BLE001 - failures are part of the measured distribution.
        row.update(
            {
                "status": "error",
                "error_type": type(exc).__name__,
                "error_message": str(exc)[:2000],
            }
        )
    row["timestamp_end_utc"] = iso_now()
    return row


def measure_transfer(
    task: TransferTask,
    order: int,
    run_id: str,
    payload_dir: Path,
    temp_dir: Path,
    timeout_sec: int,
) -> dict[str, Any]:
    node = task.node
    prepare_network_environment(node)
    size_bytes = task.size_mib * 1024 * 1024
    source = payload_path(payload_dir, task.size_mib)
    if task.direction == "upload":
        key = f"network_measurement/runs/{run_id}/{task.trial_id}.bin"
    else:
        key = reference_key(task.size_mib)
    uri = object_uri(node, key)
    target = temp_dir / f"{task.trial_id}.bin"
    row: dict[str, Any] = {
        "trial_id": task.trial_id,
        "round": task.round_id,
        "order_in_round": order,
        "timestamp_start_utc": iso_now(),
        "node_id": node.node_id,
        "provider": node.provider,
        "region": node.region,
        "bucket": node.bucket,
        "direction": task.direction,
        "size_mib": task.size_mib,
        "size_bytes": size_bytes,
        "object_uri": uri,
    }
    try:
        if not source.is_file() or source.stat().st_size != size_bytes:
            raise ValueError(f"Missing or invalid payload: {source}")
        if task.direction == "upload":
            elapsed, _, _ = run_node_command(
                node,
                upload_command(node, source, uri, timeout_sec),
                timeout_sec,
            )
            verified = remote_size(node, uri, timeout_sec)
        else:
            elapsed, _, _ = run_node_command(
                node,
                download_command(node, uri, target, timeout_sec),
                timeout_sec,
            )
            verified = target.stat().st_size
        if verified != size_bytes:
            raise ValueError(f"Transferred size mismatch: expected={size_bytes}, actual={verified}")
        row.update(
            {
                "elapsed_sec": elapsed,
                "throughput_mib_per_sec": task.size_mib / max(elapsed, 0.000001),
                # Decimal megabits/s: bytes * 8 / 1,000,000 / seconds.
                "throughput_mbps": size_bytes * 8 / 1_000_000 / max(elapsed, 0.000001),
                "status": "success",
                "verified_size_bytes": verified,
            }
        )
    except Exception as exc:  # noqa: BLE001 - retain timeout/failure observations.
        row.update(
            {
                "status": "error",
                "error_type": type(exc).__name__,
                "error_message": str(exc)[:2000],
            }
        )
    finally:
        target.unlink(missing_ok=True)
        if task.direction == "upload":
            try:
                run_node_command(
                    node,
                    delete_command(node, uri, timeout_sec),
                    timeout_sec,
                )
            except Exception as exc:  # noqa: BLE001 - retain cleanup issue without erasing measurement.
                row["cleanup_error"] = f"{type(exc).__name__}: {exc}"[:2000]
    row["timestamp_end_utc"] = iso_now()
    return row


def percentile(values: list[float], percentage: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentage / 100
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def summarize(measurement_path: Path, summary_path: Path) -> None:
    with measurement_path.open(encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))
    groups: dict[tuple[str, str, int], list[dict[str, str]]] = {}
    for row in rows:
        groups.setdefault((row["node_id"], row["direction"], int(row["size_mib"])), []).append(row)
    output = []
    for (node_id, direction, size_mib), group in sorted(groups.items()):
        successful = [row for row in group if row["status"] == "success"]
        elapsed = [float(row["elapsed_sec"]) for row in successful]
        throughput = [float(row["throughput_mib_per_sec"]) for row in successful]
        first = group[0]
        output.append(
            {
                "node_id": node_id,
                "provider": first["provider"],
                "region": first["region"],
                "direction": direction,
                "size_mib": size_mib,
                "sample_count": len(group),
                "success_count": len(successful),
                "failure_count": len(group) - len(successful),
                "failure_rate": (len(group) - len(successful)) / len(group),
                "elapsed_sec_mean": statistics.mean(elapsed) if elapsed else math.nan,
                "elapsed_sec_std": statistics.stdev(elapsed) if len(elapsed) > 1 else math.nan,
                "elapsed_sec_p50": percentile(elapsed, 50),
                "elapsed_sec_p90": percentile(elapsed, 90),
                "elapsed_sec_p95": percentile(elapsed, 95),
                "elapsed_sec_min": min(elapsed, default=math.nan),
                "elapsed_sec_max": max(elapsed, default=math.nan),
                "throughput_mib_per_sec_mean": statistics.mean(throughput) if throughput else math.nan,
                "throughput_mib_per_sec_p10": percentile(throughput, 10),
                "throughput_mib_per_sec_p50": percentile(throughput, 50),
            }
        )
    with summary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(output)


def write_scenarios(measurement_path: Path, output_path: Path) -> None:
    with measurement_path.open(encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))
    rounds: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        rounds.setdefault(int(row["round"]), []).append(
            {
                "trial_id": row["trial_id"],
                "node_id": row["node_id"],
                "direction": row["direction"],
                "size_mib": int(row["size_mib"]),
                "elapsed_sec": float(row["elapsed_sec"]) if row["elapsed_sec"] else None,
                "throughput_mbps": float(row["throughput_mbps"]) if row["throughput_mbps"] else None,
                "status": row["status"],
                "timestamp_start_utc": row["timestamp_start_utc"],
            }
        )
    with output_path.open("w", encoding="utf-8") as file:
        for round_id in sorted(rounds):
            json.dump({"scenario_id": round_id, "measurements": rounds[round_id]}, file)
            file.write("\n")


def cleanup_references(nodes: Iterable[Node], sizes_mib: Iterable[int], timeout_sec: int) -> None:
    for node in nodes:
        for size_mib in sizes_mib:
            uri = object_uri(node, reference_key(size_mib))
            try:
                run_command(delete_command(node, uri, timeout_sec), timeout_sec)
                print(f"[cleanup] deleted {uri}", flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"[cleanup] warning: {uri}: {exc}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure stochastic local-relay upload/download scenarios for cloud buckets."
    )
    parser.add_argument("--buckets", default="configs/buckets.json")
    parser.add_argument("--nodes", default="", help="Comma-separated node ids; default: all configured nodes.")
    parser.add_argument("--sizes-mib", type=parse_sizes, default=list(DEFAULT_SIZES_MIB))
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260624)
    parser.add_argument("--timeout-sec", type=int, default=900)
    parser.add_argument(
        "--network-mode",
        choices=("system", "direct"),
        default="system",
        help=(
            "system uses the current Windows proxy rules; direct bypasses the proxy for bucket traffic. "
            "In direct mode, GCP OAuth token refresh still uses the system proxy outside timed transfers."
        ),
    )
    parser.add_argument("--sleep-sec", type=float, default=0.5)
    parser.add_argument("--round-pause-sec", type=float, default=0)
    parser.add_argument(
        "--prepare-attempts",
        type=int,
        default=5,
        help="Retries for creating reference download objects; not applied to measured transfers.",
    )
    parser.add_argument("--payload-dir", default="experiments/network_payloads")
    parser.add_argument("--run-dir", default="")
    parser.add_argument("--skip-probes", action="store_true")
    parser.add_argument("--skip-prepare", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--cleanup-reference", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    global NETWORK_MODE
    args = parse_args()
    NETWORK_MODE = args.network_mode
    if args.rounds <= 0:
        raise SystemExit("--rounds must be positive")
    if args.prepare_attempts <= 0:
        raise SystemExit("--prepare-attempts must be positive")
    sizes_mib = list(args.sizes_mib)
    nodes = load_nodes(Path(args.buckets), args.nodes)
    cli_paths = preflight(nodes, strict=not args.dry_run)
    transfers_per_round = len(nodes) * len(sizes_mib) * 2
    total_transfers = transfers_per_round * args.rounds
    one_direction_gib = sum(sizes_mib) * len(nodes) * args.rounds / 1024
    plan = {
        "nodes": [node.__dict__ for node in nodes],
        "sizes_mib": sizes_mib,
        "rounds": args.rounds,
        "transfers_per_round": transfers_per_round,
        "total_transfers": total_transfers,
        "upload_gib": one_direction_gib,
        "download_gib": one_direction_gib,
        "total_transfer_gib": one_direction_gib * 2,
        "network_mode": args.network_mode,
        "cli_paths": cli_paths,
        "missing_clis": sorted(provider for provider, path in cli_paths.items() if not path),
    }
    print(json.dumps(plan, indent=2), flush=True)
    if args.dry_run:
        return 0

    if args.network_mode == "direct" and any(node.provider == "gcp" for node in nodes):
        refresh_gcp_access_token(force=True)

    payload_dir = Path(args.payload_dir).resolve()
    for size_mib in sizes_mib:
        ensure_payload(payload_path(payload_dir, size_mib), size_mib)

    if not args.skip_prepare:
        prepare_references(
            nodes,
            sizes_mib,
            payload_dir,
            args.timeout_sec,
            args.prepare_attempts,
        )
    if args.prepare_only:
        return 0

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.run_dir or f"experiments/bucket_network_scenarios/run_{run_id}").resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = run_dir / "metadata.json"
    if metadata_path.exists():
        metadata = load_json(metadata_path)
        run_id = metadata["run_id"]
        expected = {
            "sizes_mib": sizes_mib,
            "rounds": args.rounds,
            "seed": args.seed,
            "node_ids": [node.node_id for node in nodes],
            "network_mode": args.network_mode,
        }
        actual = {key: metadata.get(key, "system" if key == "network_mode" else None) for key in expected}
        if actual != expected:
            raise SystemExit(f"Resume configuration mismatch: expected={expected}, existing={actual}")
    else:
        metadata = {
            "run_id": run_id,
            "created_at": iso_now(),
            "host": platform.node(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "buckets_config": str(Path(args.buckets).resolve()),
            "node_ids": [node.node_id for node in nodes],
            "nodes": [node.__dict__ for node in nodes],
            "sizes_mib": sizes_mib,
            "rounds": args.rounds,
            "seed": args.seed,
            "timeout_sec": args.timeout_sec,
            "sleep_sec": args.sleep_sec,
            "round_pause_sec": args.round_pause_sec,
            "prepare_attempts": args.prepare_attempts,
            "network_mode": args.network_mode,
            "cli_paths": cli_paths,
            "plan": plan,
            "method": (
                "Sequential wall-clock cloud CLI object transfers from the local relay. "
                "Each round covers every selected node, direction, and exact payload size in randomized order. "
                "Raw observations, including failures, are retained and aligned by round for joint resampling. "
                f"Bucket network mode: {args.network_mode}."
            ),
        }
        write_json(metadata_path, metadata)

    measurement_path = run_dir / "transfer_measurements.csv"
    probe_path = run_dir / "request_probes.csv"
    completed_trials = completed_ids(measurement_path, "trial_id")
    completed_probes = completed_ids(probe_path, "probe_id")
    temp_dir = Path(tempfile.mkdtemp(prefix=f"skyflow-network-{run_id}-"))
    try:
        for round_id in range(1, args.rounds + 1):
            print(f"[{iso_now()}] round {round_id}/{args.rounds}", flush=True)
            rng = random.Random(args.seed + round_id)
            if not args.skip_probes:
                probe_nodes = nodes.copy()
                rng.shuffle(probe_nodes)
                for order, node in enumerate(probe_nodes, start=1):
                    probe_id = f"r{round_id:03d}-{node.node_id}-metadata"
                    if probe_id in completed_probes:
                        continue
                    row = measure_probe(node, round_id, order, min(sizes_mib), args.timeout_sec)
                    append_csv(probe_path, PROBE_FIELDS, row)
                    completed_probes.add(probe_id)
                    print(
                        f"[probe] {probe_id} {row['status']} {row.get('elapsed_sec', math.nan):.3f}s",
                        flush=True,
                    )
                    if args.sleep_sec > 0:
                        time.sleep(args.sleep_sec)

            tasks = [
                TransferTask(round_id, node, direction, size_mib)
                for node in nodes
                for size_mib in sizes_mib
                for direction in ("upload", "download")
            ]
            rng.shuffle(tasks)
            for order, task in enumerate(tasks, start=1):
                if task.trial_id in completed_trials:
                    continue
                row = measure_transfer(
                    task,
                    order,
                    run_id,
                    payload_dir,
                    temp_dir,
                    args.timeout_sec,
                )
                append_csv(measurement_path, MEASUREMENT_FIELDS, row)
                completed_trials.add(task.trial_id)
                print(
                    f"[{order}/{len(tasks)}] {task.trial_id} {row['status']} "
                    f"{row.get('elapsed_sec', math.nan):.3f}s "
                    f"{row.get('throughput_mbps', math.nan):.2f} Mbps",
                    flush=True,
                )
                if args.sleep_sec > 0:
                    time.sleep(args.sleep_sec)
            summarize(measurement_path, run_dir / "transfer_summary.csv")
            write_scenarios(measurement_path, run_dir / "network_scenarios.jsonl")
            metadata["last_completed_round"] = round_id
            metadata["updated_at"] = iso_now()
            write_json(metadata_path, metadata)
            if round_id < args.rounds and args.round_pause_sec > 0:
                time.sleep(args.round_pause_sec)
        metadata["finished_at"] = iso_now()
        write_json(metadata_path, metadata)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
        if args.cleanup_reference:
            cleanup_references(nodes, sizes_mib, args.timeout_sec)
    print(f"[{iso_now()}] done: {run_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
