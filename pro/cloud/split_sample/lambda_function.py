from __future__ import annotations

import base64
import json
from typing import Any

from app import split_sample_payload


def handler(event: dict[str, Any], context: object) -> dict[str, Any]:
    body = event.get("body", "{}")
    if event.get("isBase64Encoded"):
        body = base64.b64decode(body).decode("utf-8")
    payload = json.loads(body) if isinstance(body, str) else body
    response = split_sample_payload(payload)
    return {
        "statusCode": 200,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(response),
    }

