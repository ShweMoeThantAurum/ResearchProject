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
    """Create a boto3 S3 client."""
    region = os.environ.get("AWS_REGION", "us-east-1")
    return boto3.client("s3", region_name=region)


def _global_key(round_id):
    """Return S3 key for global model."""
    prefix = get_s3_prefix()
    return f"{prefix}/round_{round_id}/global.pt"


def _update_key(round_id, role):
    """Return S3 key for processed client update."""
    prefix = get_s3_prefix()
    return f"{prefix}/round_{round_id}/updates/{role}.pt"


def _metadata_key(round_id, role):
    """Return S3 key for metadata JSON."""
    prefix = get_s3_prefix()
    return f"{prefix}/round_{round_id}/metadata/{role}.json"


def download_global_model(round_id, role):
    """Download global model for a given round and return path and size."""
    bucket = get_s3_bucket()
    key = _global_key(round_id)
    s3 = _s3_client()

    local_path = f"/tmp/global_{role}_round_{round_id}.pt"

    while True:
        start = time.time()
        try:
            s3.download_file(bucket, key, local_path)
            latency = time.time() - start
            size_bytes = os.path.getsize(local_path)
            log_event({
                "type": "client_download",
                "role": role,
                "round": round_id,
                "latency_sec": latency,
                "size_bytes": size_bytes,
                "key": key,
            })
            print(
                f"[{role}] Downloaded global r={round_id} "
                f"({size_bytes / (1024.0 * 1024.0):.3f} MB, {latency:.3f}s)"
            )
            return local_path, size_bytes
        except Exception:
            print(f"[{role}] Waiting for global model r={round_id}...")
            time.sleep(3.0)


def upload_update(round_id, role, state_dict):
    """Upload processed client update to S3 and return bytes and latency."""
    bucket = get_s3_bucket()
    key = _update_key(round_id, role)
    s3 = _s3_client()

    local_path = f"/tmp/update_{role}_round_{round_id}.pt"
    torch.save(state_dict, local_path)

    start = time.time()
    s3.upload_file(local_path, bucket, key)
    latency = time.time() - start
    size_bytes = os.path.getsize(local_path)

    log_event({
        "type": "client_upload_update",
        "role": role,
        "round": round_id,
        "latency_sec": latency,
        "size_bytes": size_bytes,
        "key": key,
    })

    print(
        f"[{role}] Uploaded update r={round_id} "
        f"({size_bytes / (1024.0 * 1024.0):.3f} MB, {latency:.3f}s)"
    )

    return size_bytes, latency


def upload_metadata(round_id, role, meta):
    """Upload metadata JSON to S3."""
    bucket = get_s3_bucket()
    key = _metadata_key(round_id, role)
    s3 = _s3_client()

    body = json.dumps(meta).encode("utf-8")

    start = time.time()
    s3.put_object(Bucket=bucket, Key=key, Body=body)
    latency = time.time() - start

    log_event({
        "type": "client_upload_meta",
        "role": role,
        "round": round_id,
        "latency_sec": latency,
        "size_bytes": len(body),
        "key": key,
    })

    print(
        f"[{role}] Uploaded metadata r={round_id}: "
        f"bandwidth={meta.get('bandwidth_mbps', 0.0):.3f} Mb/s, "
        f"total_energy={meta.get('total_energy_j', 0.0):.2f} J"
    )

    return latency
