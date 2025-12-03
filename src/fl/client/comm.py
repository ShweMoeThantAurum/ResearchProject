"""
Client-side communication utilities using S3.
Downloads global models and uploads updates and metadata.
"""

import io
import json
import os
import time

import boto3
import torch

from ..utils.logger import log_event
from ..server.utils_server import get_s3_bucket, get_s3_prefix


def _s3_client():
    region = os.environ.get("AWS_REGION", "us-east-1")
    return boto3.client("s3", region_name=region)


def _global_key(round_id):
    prefix = get_s3_prefix()
    return f"{prefix}/round_{round_id}/global.pt"


def _update_key(round_id, role):
    prefix = get_s3_prefix()
    return f"{prefix}/round_{round_id}/updates/{role}.pt"


def _metadata_key(round_id, role):
    prefix = get_s3_prefix()
    return f"{prefix}/round_{round_id}/metadata/{role}.json"


def download_global_model(round_id, role):
    """
    Download global model for the given round.
    Returns a state_dict (not a file path).
    """
    bucket = get_s3_bucket()
    key = _global_key(round_id)
    s3 = _s3_client()

    while True:
        try:
            start = time.time()
            obj = s3.get_object(Bucket=bucket, Key=key)
            raw = obj["Body"].read()
            latency = time.time() - start

            buf = io.BytesIO(raw)
            state = torch.load(buf, map_location="cpu")

            size_bytes = len(raw)
            print(
                f"[{role}] Downloaded global r={round_id} "
                f"({size_bytes / (1024*1024):.3f} MB, {latency:.3f}s)"
            )

            log_event({
                "type": "client_download",
                "role": role,
                "round": round_id,
                "latency_sec": latency,
                "size_bytes": size_bytes,
                "key": key,
            })

            return state
        except Exception:
            print(f"[{role}] Waiting for global model r={round_id}...")
            time.sleep(3.0)


def upload_update(round_id, role, state_dict):
    """
    Upload processed client update.
    """
    bucket = get_s3_bucket()
    key = _update_key(round_id, role)
    s3 = _s3_client()

    buf = io.BytesIO()
    torch.save(state_dict, buf)
    buf.seek(0)

    start = time.time()
    s3.put_object(Bucket=bucket, Key=key, Body=buf.getvalue())
    latency = time.time() - start
    size_bytes = len(buf.getvalue())

    print(
        f"[{role}] Uploaded update r={round_id} "
        f"({size_bytes / (1024*1024):.3f} MB, {latency:.3f}s)"
    )

    log_event({
        "type": "client_upload_update",
        "role": role,
        "round": round_id,
        "latency_sec": latency,
        "size_bytes": size_bytes,
        "key": key,
    })

    return size_bytes, latency


def upload_metadata(round_id, role, meta):
    bucket = get_s3_bucket()
    key = _metadata_key(round_id, role)
    s3 = _s3_client()

    body = json.dumps(meta).encode("utf-8")

    start = time.time()
    s3.put_object(Bucket=bucket, Key=key, Body=body)
    latency = time.time() - start

    print(
        f"[{role}] Uploaded metadata r={round_id}: "
        f"bandwidth={meta.get('bandwidth_mbps', 0.0):.3f} Mb/s, "
        f"total_energy={meta.get('total_energy_j', 0.0):.2f} J"
    )

    log_event({
        "type": "client_upload_meta",
        "role": role,
        "round": round_id,
        "latency_sec": latency,
        "size_bytes": len(body),
        "key": key,
    })

    return latency
