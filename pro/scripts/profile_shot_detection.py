#!/usr/bin/env python3
"""Profile cloud shot-detection nodes on the ActivityNet profile videos.

The client-observed latency is bounded by async job polling: the real completion
time is after the last pending poll and before the first completed poll. It
intentionally excludes dataset upload time.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


GCP_REGIONS = ["us-east1", "us-west1", "europe-west1", "asia-east1"]
AWS_REGIONS = ["us-west-2", "us-east-2", "ap-southeast-1", "eu-central-1"]
ALIYUN_REGIONS = ["cn-shanghai"]

ALIYUN_OSS_ENDPOINTS = {
    "cn-shanghai": "oss-cn-shanghai.aliyuncs.com",
}

CSV_FIELDS = [
    "run_id",
    "node_id",
    "provider",
    "region",
    "video_id",
    "filename",
    "object_key",
    "input_uri",
    "size_bytes",
    "duration_sec",
    "submit_start_ts",
    "submit_end_ts",
    "complete_ts",
    "submit_wall_ms",
    "job_elapsed_ms",
    "client_wall_ms",
    "effective_latency_ms",
    "effective_latency_source",
    "latency_uncertainty_ms",
    "observed_completion_lower_bound_ms",
    "observed_completion_upper_bound_ms",
    "provider_reported_start_ts",
    "provider_reported_end_ts",
    "provider_reported_elapsed_ms",
    "poll_interval_sec",
    "poll_count",
    "job_id",
    "status",
    "shot_count",
    "result_bytes",
    "result_path",
    "result_uri",
    "result_upload_status",
    "result_upload_error_type",
    "result_upload_error_message",
    "error_type",
    "error_message",
]


@dataclass(frozen=True)
class Node:
    node_id: str
    provider: str
    region: str


@dataclass(frozen=True)
class ProfileOptions:
    poll_interval_sec: float
    aws_sns_topic_arn: str = ""
    aws_sns_role_arn: str = ""
    aws_sqs_queue_url: str = ""
    aws_sqs_region: str = ""
    aws_event_wait_timeout_sec: int = 3600

    @property
    def aws_event_enabled(self) -> bool:
        return bool(self.aws_sns_topic_arn and self.aws_sns_role_arn and self.aws_sqs_queue_url)


class AwsCompletionEvents:
    def __init__(self, queue_url: str, region: str) -> None:
        import boto3

        self.queue_url = queue_url
        self.client = boto3.client("sqs", region_name=region)
        self.lock = threading.Lock()
        self.events_by_job_id: dict[str, dict[str, Any]] = {}

    def wait_for(self, job_id: str, timeout_sec: int) -> dict[str, Any] | None:
        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline:
            with self.lock:
                event = self.events_by_job_id.get(job_id)
            if event is not None:
                return event

            wait_time = int(min(20, max(1, deadline - time.monotonic())))
            response = self.client.receive_message(
                QueueUrl=self.queue_url,
                MaxNumberOfMessages=10,
                WaitTimeSeconds=wait_time,
            )
            for message in response.get("Messages", []):
                event = parse_aws_completion_message(message)
                if event is None:
                    continue
                event_job_id = event.get("JobId")
                if event_job_id:
                    with self.lock:
                        self.events_by_job_id[event_job_id] = event
                    self.client.delete_message(
                        QueueUrl=self.queue_url,
                        ReceiptHandle=message["ReceiptHandle"],
                    )
            with self.lock:
                event = self.events_by_job_id.get(job_id)
            if event is not None:
                return event
        return None


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso(dt: datetime) -> str:
    return dt.isoformat(timespec="milliseconds")


def elapsed_ms(start: datetime, end: datetime) -> int:
    return int((end - start).total_seconds() * 1000)


def parse_iso_datetime(value: Any) -> datetime | None:
    if not value:
        return None
    if isinstance(value, int | float):
        timestamp = float(value)
        if timestamp > 10_000_000_000:
            timestamp /= 1000
        return datetime.fromtimestamp(timestamp, timezone.utc)
    value = str(value)
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def load_items(path: Path, max_videos: int | None) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    items = data["items"]
    if max_videos:
        items = items[:max_videos]
    return items


def completed_keys(csv_path: Path) -> set[tuple[str, str]]:
    if not csv_path.exists():
        return set()
    done: set[tuple[str, str]] = set()
    with csv_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("status") in {"success", "SUCCEEDED", "PROCESS_SUCCESS"}:
                done.add((row["node_id"], row["filename"]))
    return done


class Recorder:
    def __init__(self, run_dir: Path, run_id: str) -> None:
        self.run_dir = run_dir
        self.run_id = run_id
        self.csv_path = run_dir / "shot_detection_profile.csv"
        self.jsonl_path = run_dir / "shot_detection_profile_raw.jsonl"
        self.lock = threading.Lock()
        run_dir.mkdir(parents=True, exist_ok=True)
        if not self.csv_path.exists():
            with self.csv_path.open("w", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=CSV_FIELDS).writeheader()

    def write(self, row: dict[str, Any], raw: dict[str, Any] | None = None) -> None:
        row = {field: row.get(field, "") for field in CSV_FIELDS}
        row["run_id"] = self.run_id
        with self.lock:
            with self.csv_path.open("a", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=CSV_FIELDS).writerow(row)
            if raw is not None:
                safe_raw = dict(raw)
                safe_raw.pop("signed_url", None)
                with self.jsonl_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(safe_raw, ensure_ascii=False, default=str) + "\n")

    def write_result(self, node: Node, item: dict[str, Any], result: dict[str, Any]) -> tuple[str, int]:
        result_dir = self.run_dir / "results" / node.node_id
        result_dir.mkdir(parents=True, exist_ok=True)
        result_path = result_dir / f"{item['video_id']}.json"
        payload = json.dumps(result, ensure_ascii=False, indent=2, default=str)
        result_path.write_text(payload, encoding="utf-8")
        return str(result_path.relative_to(self.run_dir)), len(payload.encode("utf-8"))


def make_nodes() -> list[Node]:
    nodes: list[Node] = []
    nodes.extend(Node(f"gcp_{r.replace('-', '_')}", "gcp", r) for r in GCP_REGIONS)
    nodes.extend(Node(f"aws_{r.replace('-', '_')}", "aws", r) for r in AWS_REGIONS)
    nodes.extend(Node(f"aliyun_{r.replace('-', '_')}", "aliyun", r) for r in ALIYUN_REGIONS)
    return nodes


def gcp_detect(item: dict[str, Any], node: Node, options: ProfileOptions) -> tuple[dict[str, Any], dict[str, Any]]:
    input_uri = f"gs://skyflow-prototype-gcp-{node.region}/{item['object_key']}"
    token = gcp_access_token()
    project = gcp_project()
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "x-goog-user-project": project,
    }
    submit_start = utc_now()
    response = http_json(
        "POST",
        "https://videointelligence.googleapis.com/v1/videos:annotate",
        headers=headers,
        body={
            "inputUri": input_uri,
            "features": ["SHOT_CHANGE_DETECTION"],
            "locationId": node.region,
        },
        timeout=120,
    )
    submit_end = utc_now()
    job_id = response["name"]

    poll_count = 0
    last_pending_at = submit_end
    operation: dict[str, Any] = response
    while not operation.get("done"):
        if poll_count > 0:
            time.sleep(options.poll_interval_sec)
        poll_count += 1
        operation = http_json(
            "GET",
            f"https://videointelligence.googleapis.com/v1/{job_id}",
            headers=headers,
            timeout=120,
        )
        poll_observed_at = utc_now()
        if not operation.get("done"):
            last_pending_at = poll_observed_at
    complete = utc_now()

    if "error" in operation:
        raise RuntimeError(json.dumps(operation["error"], ensure_ascii=False))

    annotation_results = operation.get("response", {}).get("annotationResults", [])
    annotations = annotation_results[0].get("shotAnnotations", []) if annotation_results else []
    raw = {
        "provider": node.provider,
        "region": node.region,
        "job_id": job_id,
        "input_uri": input_uri,
        "annotation_result_count": len(annotation_results),
        "shot_count": len(annotations),
        "operation": operation,
    }
    provider_timing = gcp_provider_timing(operation)
    row = base_row(
        item,
        node,
        input_uri,
        submit_start,
        submit_end,
        complete,
        options.poll_interval_sec,
        poll_count,
        job_id,
        last_pending_at,
        provider_timing,
    )
    row.update(
        {
            "status": "success",
            "shot_count": len(annotations),
            "result_bytes": len(json.dumps(operation).encode("utf-8")),
        }
    )
    return row, raw


def gcp_access_token() -> str:
    result = subprocess.run(
        ["gcloud", "auth", "print-access-token"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout.strip()


def gcp_project() -> str:
    result = subprocess.run(
        ["gcloud", "config", "get-value", "project"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout.strip()


def http_json(method: str, url: str, headers: dict[str, str], body: dict[str, Any] | None = None, timeout: int = 120) -> dict[str, Any]:
    import urllib.error
    import urllib.request

    data = None if body is None else json.dumps(body).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        payload = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {payload}") from exc


def aws_detect(
    item: dict[str, Any],
    node: Node,
    options: ProfileOptions,
    aws_events: AwsCompletionEvents | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    import boto3

    bucket = f"skyflow-prototype-aws-{node.region}"
    key = item["object_key"]
    input_uri = f"s3://{bucket}/{key}"
    client = boto3.client("rekognition", region_name=node.region)

    submit_start = utc_now()
    start_kwargs: dict[str, Any] = {
        "Video": {"S3Object": {"Bucket": bucket, "Name": key}},
        "SegmentTypes": ["SHOT"],
    }
    if options.aws_event_enabled:
        start_kwargs["NotificationChannel"] = {
            "SNSTopicArn": options.aws_sns_topic_arn,
            "RoleArn": options.aws_sns_role_arn,
        }
        start_kwargs["JobTag"] = f"{node.node_id}:{item['video_id']}"
    response = client.start_segment_detection(**start_kwargs)
    submit_end = utc_now()
    job_id = response["JobId"]

    poll_count = 0
    last_pending_at = submit_end
    status = "IN_PROGRESS"
    result_pages: list[dict[str, Any]] = []
    provider_timing: dict[str, Any] = {}
    completion_event: dict[str, Any] | None = None
    if options.aws_event_enabled and aws_events is not None:
        completion_event = aws_events.wait_for(job_id, options.aws_event_wait_timeout_sec)
        if completion_event is not None:
            event_time = parse_iso_datetime(completion_event.get("Timestamp"))
            event_status = completion_event.get("Status")
            if event_status:
                status = event_status
            if event_time is not None:
                provider_timing = {
                    "provider_reported_end_ts": iso(event_time),
                    "provider_reported_elapsed_ms": max(0, elapsed_ms(submit_end, event_time)),
                    "effective_latency_source": "aws_sns_completion_event",
                }
    if completion_event is None:
        while status == "IN_PROGRESS":
            if poll_count > 0:
                time.sleep(options.poll_interval_sec)
            poll_count += 1
            page = client.get_segment_detection(JobId=job_id, MaxResults=1000)
            poll_observed_at = utc_now()
            status = page["JobStatus"]
            if status != "IN_PROGRESS":
                result_pages.append(page)
            else:
                last_pending_at = poll_observed_at
    complete = utc_now()

    segments: list[dict[str, Any]] = []
    if status == "SUCCEEDED":
        first = result_pages[0] if result_pages else client.get_segment_detection(JobId=job_id, MaxResults=1000)
        segments.extend(first.get("Segments", []))
        token = first.get("NextToken")
        while token:
            page = client.get_segment_detection(JobId=job_id, MaxResults=1000, NextToken=token)
            segments.extend(page.get("Segments", []))
            token = page.get("NextToken")

    shot_count = sum(1 for segment in segments if segment.get("Type") == "SHOT")
    raw = {
        "provider": node.provider,
        "region": node.region,
        "job_id": job_id,
        "input_uri": input_uri,
        "status": status,
        "shot_count": shot_count,
        "page_count": max(1, len(result_pages)),
        "completion_event": completion_event,
        "segments": segments,
    }
    row = base_row(
        item,
        node,
        input_uri,
        submit_start,
        submit_end,
        complete,
        options.poll_interval_sec,
        poll_count,
        job_id,
        last_pending_at,
        provider_timing,
    )
    row.update(
        {
            "status": "success" if status == "SUCCEEDED" else status,
            "shot_count": shot_count,
            "result_bytes": len(json.dumps(raw).encode("utf-8")),
            "error_message": "" if status == "SUCCEEDED" else status,
        }
    )
    return row, raw


def result_object_key(node: Node, item: dict[str, Any]) -> str:
    return f"profile/shot_detection/{node.node_id}/{item['video_id']}.json"


def upload_result_to_bucket(node: Node, item: dict[str, Any], result_path: Path) -> str:
    key = result_object_key(node, item)
    if node.provider == "gcp":
        uri = f"gs://skyflow-prototype-gcp-{node.region}/{key}"
        subprocess.run(["gcloud", "storage", "cp", str(result_path), uri, "--quiet"], check=True)
        return uri
    if node.provider == "aws":
        uri = f"s3://skyflow-prototype-aws-{node.region}/{key}"
        subprocess.run(["aws", "s3", "cp", str(result_path), uri, "--region", node.region], check=True)
        return uri
    if node.provider == "aliyun":
        endpoint = ALIYUN_OSS_ENDPOINTS[node.region]
        uri = f"oss://skyflow-prototype-aliyun-{node.region}/{key}"
        run_dir = result_path.parents[2]
        checkpoint_dir = run_dir / "oss_checkpoint"
        output_dir = run_dir / "oss_output"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "aliyun",
                "oss",
                "cp",
                str(result_path),
                uri,
                "--region",
                node.region,
                "--endpoint",
                endpoint,
                "--checkpoint-dir",
                str(checkpoint_dir),
                "--output-dir",
                str(output_dir),
                "--force",
            ],
            check=True,
        )
        return uri
    raise ValueError(f"Unsupported provider: {node.provider}")


def result_uri_for(node: Node, item: dict[str, Any]) -> str:
    key = result_object_key(node, item)
    if node.provider == "gcp":
        return f"gs://skyflow-prototype-gcp-{node.region}/{key}"
    if node.provider == "aws":
        return f"s3://skyflow-prototype-aws-{node.region}/{key}"
    if node.provider == "aliyun":
        return f"oss://skyflow-prototype-aliyun-{node.region}/{key}"
    raise ValueError(f"Unsupported provider: {node.provider}")


def bucket_result_exists(node: Node, item: dict[str, Any]) -> bool:
    uri = result_uri_for(node, item)
    if node.provider == "gcp":
        cmd = ["gcloud", "storage", "objects", "describe", uri, "--format=value(name)"]
    elif node.provider == "aws":
        bucket, key = uri[5:].split("/", 1)
        cmd = ["aws", "s3api", "head-object", "--bucket", bucket, "--key", key, "--region", node.region]
    elif node.provider == "aliyun":
        endpoint = ALIYUN_OSS_ENDPOINTS[node.region]
        cmd = ["aliyun", "oss", "stat", uri, "--region", node.region, "--endpoint", endpoint]
    else:
        return False
    result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return result.returncode == 0


def aliyun_cli_value(key: str) -> str:
    result = subprocess.run(
        ["aliyun", "configure", "get", key],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    value = result.stdout.strip()
    if value:
        return value

    config_path = Path.home() / ".aliyun" / "config.json"
    if config_path.exists():
        data = json.loads(config_path.read_text(encoding="utf-8"))
        current = data.get("current")
        for profile in data.get("profiles", []):
            if profile.get("name") == current:
                value = str(profile.get(key, "")).strip()
                if value:
                    return value
    return ""


def aliyun_signed_url(region: str, bucket: str, key: str, timeout_sec: int = 7200) -> str:
    endpoint = ALIYUN_OSS_ENDPOINTS[region]
    result = subprocess.run(
        [
            "aliyun",
            "oss",
            "sign",
            f"oss://{bucket}/{key}",
            "--timeout",
            str(timeout_sec),
            "--region",
            region,
            "--endpoint",
            endpoint,
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    for line in result.stdout.splitlines():
        if line.startswith("https://"):
            return line.strip()
    raise RuntimeError("aliyun oss sign did not return a signed URL")


def aliyun_common_request(action: str, params: dict[str, str]) -> dict[str, Any]:
    from aliyunsdkcore.client import AcsClient
    from aliyunsdkcore.request import CommonRequest

    access_key = aliyun_cli_value("access_key_id")
    secret = aliyun_cli_value("access_key_secret")
    client = AcsClient(access_key, secret, "cn-shanghai")
    request = CommonRequest()
    request.set_accept_format("json")
    request.set_method("POST")
    request.set_protocol_type("https")
    request.set_domain("videorecog.cn-shanghai.aliyuncs.com")
    request.set_version("2020-03-20")
    request.set_action_name(action)
    for key, value in params.items():
        request.add_query_param(key, value)
    response = client.do_action_with_exception(request)
    return json.loads(response.decode("utf-8"))


def aliyun_detect(item: dict[str, Any], node: Node, options: ProfileOptions) -> tuple[dict[str, Any], dict[str, Any]]:
    bucket = f"skyflow-prototype-aliyun-{node.region}"
    key = item["object_key"]
    input_uri = f"oss://{bucket}/{key}"
    signed_url = aliyun_signed_url(node.region, bucket, key)

    submit_start = utc_now()
    response = aliyun_common_request("DetectVideoShot", {"VideoUrl": signed_url})
    submit_end = utc_now()
    job_id = response.get("RequestId", "")
    if not job_id:
        raise RuntimeError(f"Aliyun DetectVideoShot returned no RequestId: {response}")

    poll_count = 0
    last_pending_at = submit_end
    status = ""
    result_data: dict[str, Any] = {}
    while True:
        if poll_count > 0:
            time.sleep(options.poll_interval_sec)
        poll_count += 1
        poll_response = aliyun_common_request("GetAsyncJobResult", {"JobId": job_id})
        poll_observed_at = utc_now()
        data = poll_response.get("Data") or {}
        status = data.get("Status", "")
        result_data = data
        if status not in {"PROCESSING", "QUEUING", "RUNNING"}:
            break
        last_pending_at = poll_observed_at
    complete = utc_now()

    result_text = result_data.get("Result") or ""
    parsed_result: Any = None
    shot_count = ""
    if result_text:
        try:
            parsed_result = json.loads(result_text)
            shot_count = infer_aliyun_shot_count(parsed_result)
        except Exception:
            parsed_result = result_text

    raw = {
        "provider": node.provider,
        "region": node.region,
        "job_id": job_id,
        "input_uri": input_uri,
        "status": status,
        "result": parsed_result,
    }
    row = base_row(
        item,
        node,
        input_uri,
        submit_start,
        submit_end,
        complete,
        options.poll_interval_sec,
        poll_count,
        job_id,
        last_pending_at,
    )
    row.update(
        {
            "status": "success" if status == "PROCESS_SUCCESS" else status,
            "shot_count": shot_count,
            "result_bytes": len(json.dumps(raw, ensure_ascii=False, default=str).encode("utf-8")),
            "error_message": "" if status == "PROCESS_SUCCESS" else json.dumps(result_data, ensure_ascii=False)[:1000],
        }
    )
    return row, raw


def infer_aliyun_shot_count(result: Any) -> int | str:
    if isinstance(result, dict):
        for key in ("ShotList", "Shots", "shotList", "shots", "ShotFrameIds"):
            value = result.get(key)
            if isinstance(value, list):
                return len(value)
            if isinstance(value, str) and value.strip().startswith("["):
                try:
                    parsed = json.loads(value)
                    if isinstance(parsed, list):
                        return len(parsed)
                except Exception:
                    pass
        for value in result.values():
            inferred = infer_aliyun_shot_count(value)
            if inferred != "":
                return inferred
    if isinstance(result, list):
        return len(result)
    return ""


def gcp_provider_timing(operation: dict[str, Any]) -> dict[str, Any]:
    progress = operation.get("metadata", {}).get("annotationProgress", [])
    starts: list[datetime] = []
    ends: list[datetime] = []
    for item in progress:
        start = parse_iso_datetime(item.get("startTime"))
        end = parse_iso_datetime(item.get("updateTime"))
        if start is not None:
            starts.append(start)
        if end is not None:
            ends.append(end)
    if not starts or not ends:
        return {}
    start = min(starts)
    end = max(ends)
    return {
        "provider_reported_start_ts": iso(start),
        "provider_reported_end_ts": iso(end),
        "provider_reported_elapsed_ms": elapsed_ms(start, end),
        "effective_latency_source": "gcp_operation_metadata",
    }


def parse_aws_completion_message(message: dict[str, Any]) -> dict[str, Any] | None:
    try:
        body = json.loads(message.get("Body", "{}"))
        if isinstance(body.get("Message"), str):
            payload = json.loads(body["Message"])
            if "Timestamp" not in payload and body.get("Timestamp"):
                payload["Timestamp"] = body["Timestamp"]
            return payload
        return body
    except Exception:
        return None


def base_row(
    item: dict[str, Any],
    node: Node,
    input_uri: str,
    submit_start: datetime,
    submit_end: datetime,
    complete: datetime,
    poll_interval_sec: int,
    poll_count: int,
    job_id: str,
    last_pending_at: datetime | None = None,
    provider_timing: dict[str, Any] | None = None,
) -> dict[str, Any]:
    lower_bound = elapsed_ms(submit_end, last_pending_at) if last_pending_at else ""
    upper_bound = elapsed_ms(submit_end, complete)
    provider_timing = provider_timing or {}
    provider_elapsed = provider_timing.get("provider_reported_elapsed_ms", "")
    if provider_elapsed != "":
        effective_latency = provider_elapsed
        effective_source = provider_timing.get("effective_latency_source", "provider_reported")
        uncertainty = 0
    else:
        effective_latency = upper_bound
        effective_source = "observed_poll_upper_bound"
        uncertainty = upper_bound - lower_bound if lower_bound != "" else ""
    return {
        "node_id": node.node_id,
        "provider": node.provider,
        "region": node.region,
        "video_id": item["video_id"],
        "filename": item["filename"],
        "object_key": item["object_key"],
        "input_uri": input_uri,
        "size_bytes": item["size_bytes"],
        "duration_sec": item["duration_sec"],
        "submit_start_ts": iso(submit_start),
        "submit_end_ts": iso(submit_end),
        "complete_ts": iso(complete),
        "submit_wall_ms": elapsed_ms(submit_start, submit_end),
        "job_elapsed_ms": upper_bound,
        "client_wall_ms": elapsed_ms(submit_start, complete),
        "effective_latency_ms": effective_latency,
        "effective_latency_source": effective_source,
        "latency_uncertainty_ms": uncertainty,
        "observed_completion_lower_bound_ms": lower_bound,
        "observed_completion_upper_bound_ms": upper_bound,
        "provider_reported_start_ts": provider_timing.get("provider_reported_start_ts", ""),
        "provider_reported_end_ts": provider_timing.get("provider_reported_end_ts", ""),
        "provider_reported_elapsed_ms": provider_timing.get("provider_reported_elapsed_ms", ""),
        "poll_interval_sec": poll_interval_sec,
        "poll_count": poll_count,
        "job_id": job_id,
    }


def detect(
    item: dict[str, Any],
    node: Node,
    options: ProfileOptions,
    aws_events: AwsCompletionEvents | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if node.provider == "gcp":
        return gcp_detect(item, node, options)
    if node.provider == "aws":
        return aws_detect(item, node, options, aws_events)
    if node.provider == "aliyun":
        return aliyun_detect(item, node, options)
    raise ValueError(f"Unsupported provider: {node.provider}")


def profile_node(
    node: Node,
    items: list[dict[str, Any]],
    recorder: Recorder,
    already_done: set[tuple[str, str]],
    options: ProfileOptions,
    per_video_sleep_sec: int,
    batch_size: int,
    batch_sleep_sec: int,
    rerun_existing_results: bool,
    aws_events: AwsCompletionEvents | None,
) -> None:
    print(f"[{iso(utc_now())}] start {node.node_id} ({len(items)} videos)", flush=True)
    for index, item in enumerate(items, start=1):
        if (node.node_id, item["filename"]) in already_done:
            print(f"[{node.node_id}] skip {index}/{len(items)} {item['filename']}", flush=True)
            continue
        if not rerun_existing_results and bucket_result_exists(node, item):
            now = utc_now()
            row = {
                "node_id": node.node_id,
                "provider": node.provider,
                "region": node.region,
                "video_id": item["video_id"],
                "filename": item["filename"],
                "object_key": item["object_key"],
                "input_uri": "",
                "size_bytes": item["size_bytes"],
                "duration_sec": item["duration_sec"],
                "submit_start_ts": iso(now),
                "complete_ts": iso(now),
                "status": "skipped_existing",
                "result_uri": result_uri_for(node, item),
                "result_upload_status": "skipped_existing",
            }
            recorder.write(row, {"provider": node.provider, "region": node.region, "skipped_existing": row["result_uri"]})
            print(f"[{node.node_id}] skip-existing {index}/{len(items)} {item['filename']}", flush=True)
            continue
        try:
            row, raw = detect(item, node, options, aws_events)
        except Exception as exc:  # keep the run moving across nodes/videos
            now = utc_now()
            row = {
                "node_id": node.node_id,
                "provider": node.provider,
                "region": node.region,
                "video_id": item["video_id"],
                "filename": item["filename"],
                "object_key": item["object_key"],
                "input_uri": "",
                "size_bytes": item["size_bytes"],
                "duration_sec": item["duration_sec"],
                "submit_start_ts": iso(now),
                "submit_end_ts": "",
                "complete_ts": iso(now),
                "poll_interval_sec": options.poll_interval_sec,
                "status": "error",
                "error_type": type(exc).__name__,
                "error_message": str(exc)[:1000],
            }
            raw = {
                "provider": node.provider,
                "region": node.region,
                "video": item["filename"],
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback": traceback.format_exc(),
            }
        result_path, result_bytes = recorder.write_result(node, item, raw)
        row["result_path"] = result_path
        row["result_bytes"] = result_bytes
        absolute_result_path = recorder.run_dir / result_path
        try:
            row["result_uri"] = upload_result_to_bucket(node, item, absolute_result_path)
            row["result_upload_status"] = "success"
        except Exception as exc:
            row["result_upload_status"] = "error"
            row["result_upload_error_type"] = type(exc).__name__
            row["result_upload_error_message"] = str(exc)[:1000]
        recorder.write(row, raw)
        print(
            f"[{node.node_id}] {index}/{len(items)} {item['filename']} "
            f"status={row.get('status')} effective_ms={row.get('effective_latency_ms', '')} "
            f"source={row.get('effective_latency_source', '')} "
            f"bounds_ms=[{row.get('observed_completion_lower_bound_ms', '')},"
            f"{row.get('observed_completion_upper_bound_ms', '')}]",
            flush=True,
        )
        if per_video_sleep_sec > 0:
            time.sleep(per_video_sleep_sec)
        if batch_size > 0 and index % batch_size == 0 and batch_sleep_sec > 0:
            time.sleep(batch_sleep_sec)
    print(f"[{iso(utc_now())}] done {node.node_id}", flush=True)


def run_stage(
    stage_name: str,
    nodes: list[Node],
    items: list[dict[str, Any]],
    recorder: Recorder,
    already_done: set[tuple[str, str]],
    concurrency: int,
    args: argparse.Namespace,
    options: ProfileOptions,
    aws_events: AwsCompletionEvents | None,
) -> None:
    print(f"[{iso(utc_now())}] stage {stage_name}: {len(nodes)} nodes, concurrency={concurrency}", flush=True)
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(
                profile_node,
                node,
                items,
                recorder,
                already_done,
                options,
                args.per_video_sleep_sec,
                args.batch_size,
                args.batch_sleep_sec,
                args.rerun_existing_results,
                aws_events,
            )
            for node in nodes
        ]
        for future in as_completed(futures):
            future.result()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        default="experiments/activitynet_splits/upload_manifests/profile_videos_upload_manifest_seed20260622.json",
    )
    parser.add_argument("--run-dir", default="")
    parser.add_argument("--max-videos", type=int, default=0)
    parser.add_argument("--poll-interval-sec", type=float, default=1.0)
    parser.add_argument("--aws-sns-topic-arn", default="", help="Optional Rekognition completion SNS topic ARN.")
    parser.add_argument("--aws-sns-role-arn", default="", help="Optional IAM role ARN that lets Rekognition publish to SNS.")
    parser.add_argument("--aws-sqs-queue-url", default="", help="Optional SQS queue subscribed to the SNS topic.")
    parser.add_argument("--aws-sqs-region", default="", help="Region for the SQS queue. Defaults to the first AWS node region.")
    parser.add_argument("--aws-event-wait-timeout-sec", type=int, default=3600)
    parser.add_argument("--per-video-sleep-sec", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--batch-sleep-sec", type=int, default=45)
    parser.add_argument("--nodes", default="", help="Comma-separated node ids. Empty means all 9 shot nodes.")
    parser.add_argument("--stage1-concurrency", type=int, default=3)
    parser.add_argument("--stage2-concurrency", type=int, default=2)
    parser.add_argument("--rerun-existing-results", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = Path(args.manifest)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.run_dir or f"experiments/shot_detection_profile/run_{run_id}")
    recorder = Recorder(run_dir, run_id)
    items = load_items(manifest, args.max_videos or None)
    nodes = make_nodes()
    if args.nodes:
        requested = set(args.nodes.split(","))
        nodes = [node for node in nodes if node.node_id in requested]
        missing = sorted(requested - {node.node_id for node in nodes})
        if missing:
            raise SystemExit(f"Unknown node ids: {', '.join(missing)}")

    aws_nodes = [node for node in nodes if node.provider == "aws"]
    aws_sqs_region = args.aws_sqs_region or (aws_nodes[0].region if aws_nodes else "")
    options = ProfileOptions(
        poll_interval_sec=args.poll_interval_sec,
        aws_sns_topic_arn=args.aws_sns_topic_arn,
        aws_sns_role_arn=args.aws_sns_role_arn,
        aws_sqs_queue_url=args.aws_sqs_queue_url,
        aws_sqs_region=aws_sqs_region,
        aws_event_wait_timeout_sec=args.aws_event_wait_timeout_sec,
    )
    aws_events = AwsCompletionEvents(args.aws_sqs_queue_url, aws_sqs_region) if options.aws_event_enabled else None

    metadata = {
        "run_id": run_id,
        "manifest": str(manifest),
        "run_dir": str(run_dir),
        "videos": len(items),
        "nodes": [node.__dict__ for node in nodes],
        "poll_interval_sec": args.poll_interval_sec,
        "timing": {
            "effective_latency_field": "effective_latency_ms",
            "gcp": "provider metadata when present",
            "aws": "SNS/SQS completion event when configured, otherwise polling upper bound",
            "aliyun": "polling upper bound",
            "aws_event_enabled": options.aws_event_enabled,
            "aws_sqs_region": aws_sqs_region,
        },
        "per_video_sleep_sec": args.per_video_sleep_sec,
        "batch_size": args.batch_size,
        "batch_sleep_sec": args.batch_sleep_sec,
        "started_at": iso(utc_now()),
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(json.dumps(metadata, indent=2), flush=True)
    already_done = completed_keys(recorder.csv_path)

    stage1_ids = {"gcp_asia_east1", "aws_ap_southeast_1", "aliyun_cn_shanghai"}
    stage1 = [node for node in nodes if node.node_id in stage1_ids]
    stage2 = [node for node in nodes if node.node_id not in stage1_ids]
    if stage1:
        run_stage("cross_provider", stage1, items, recorder, already_done, args.stage1_concurrency, args, options, aws_events)
    if stage2:
        run_stage("remaining", stage2, items, recorder, already_done, args.stage2_concurrency, args, options, aws_events)

    metadata["finished_at"] = iso(utc_now())
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"[{iso(utc_now())}] all done: {run_dir}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        raise SystemExit(130)
