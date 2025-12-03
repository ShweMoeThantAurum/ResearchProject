"""
Client-side communication using S3.
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
    """Build S3 key for a global model."""
    prefix = get_s3_prefix()
    return f"{prefix}/round_{round_id}/global.pt"


def _update_key(round_id, role):
    """Build S3 key for a processed client update."""
    prefix = get_s3_prefix()
    return f"{prefix}/round_{round_id}/updates/{role}.pt"


def _metadata_key(round_id, role):
    """Build S3 key for a client metadata JSON file."""
    prefix = get_s3_prefix()
    return f"{prefix}/round_{round_id}/metadata/{role}.json"


def download_global_model(round_id, role):
    """
    Download global model state dict for the given round.
    Retries until the server uploads the model.
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

            size_mb = len(raw) / (1024.0 * 1024.0)
            print(f"[{role}] Downloaded global model r={round_id} ({size_mb:.3f} MB, {latency:.3f}s)")

            log_event(
                f"[{role}] download_global r={round_id} "
                f"size_mb={size_mb:.3f} latency_s={latency:.3f} key={key}"
            )

            return state
        except Exception:
            print(f"[{role}] Waiting for global model r={round_id}...")
            time.sleep(3.0)


def upload_update(round_id, role, state_dict):
    """
    Upload processed client update to S3.
    Returns (size_bytes, latency_sec).
    """
    bucket = get_s3_bucket()
    key = _update_key(round_id, role)
    s3 = _s3_client()

    buf = io.BytesIO()
    torch.save(state_dict, buf)
    body = buf.getvalue()
    size_bytes = len(body)

    start = time.time()
    s3.put_object(Bucket=bucket, Key=key, Body=body)
    latency = time.time() - start

    size_mb = size_bytes / (1024.0 * 1024.0)
    print(f"[{role}] Uploaded update r={round_id} ({size_mb:.3f} MB, {latency:.3f}s)")

    log_event(
        f"[{role}] upload_update r={round_id} "
        f"size_mb={size_mb:.3f} latency_s={latency:.3f} key={key}"
    )

    return size_bytes, latency


def upload_metadata(round_id, role, meta):
    """Upload per-round client metadata JSON for AEFL selection."""
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

    log_event(
        f"[{role}] upload_metadata r={round_id} "
        f"bytes={len(body)} latency_s={latency:.3f} key={key}"
    )

    return latency
