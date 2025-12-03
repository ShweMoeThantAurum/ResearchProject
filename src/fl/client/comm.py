"""
Client-side S3 communication utilities.

Responsible for:
- downloading the global model for each round
- uploading processed updates
- uploading metadata used by AEFL selection.
"""

import io
import json
import os
import time

import boto3
import torch

from src.fl.utils.logger import log_event
from src.fl.server.utils_server import get_s3_bucket, get_s3_prefix


def _s3_client():
    """Create a boto3 S3 client."""
    region = os.environ.get("AWS_REGION", "us-east-1")
    return boto3.client("s3", region_name=region)


def _global_key(round_id):
    """Key for the global model at a specific round."""
    prefix = get_s3_prefix()
    return f"{prefix}/round_{round_id}/global.pt"


def _update_key(round_id, role):
    """Key for a client update at a specific round."""
    prefix = get_s3_prefix()
    return f"{prefix}/round_{round_id}/updates/{role}.pt"


def _metadata_key(round_id, role):
    """Key for client metadata JSON."""
    prefix = get_s3_prefix()
    return f"{prefix}/round_{round_id}/metadata/{role}.json"


def download_global_model(round_id, role):
    """Blocking download of global model state_dict for this round."""
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
            mb = size_bytes / (1024.0 * 1024.0)

            print(f"[{role}] Downloaded global r={round_id} ({mb:.3f} MB, {latency:.3f}s)")
            log_event(
                f"[{role}] download_global round={round_id} "
                f"size_mb={mb:.3f} latency_s={latency:.3f}"
            )

            return state
        except Exception:
            print(f"[{role}] Waiting for global model r={round_id}...")
            time.sleep(3.0)


def upload_update(round_id, role, state_dict):
    """Upload processed client update for this round."""
    bucket = get_s3_bucket()
    key = _update_key(round_id, role)
    s3 = _s3_client()

    buf = io.BytesIO()
    torch.save(state_dict, buf)
    body = buf.getvalue()

    start = time.time()
    s3.put_object(Bucket=bucket, Key=key, Body=body)
    latency = time.time() - start
    size_bytes = len(body)
    mb = size_bytes / (1024.0 * 1024.0)

    print(f"[{role}] Uploaded update r={round_id} ({mb:.3f} MB, {latency:.3f}s)")
    log_event(
        f"[{role}] upload_update round={round_id} size_mb={mb:.3f} latency_s={latency:.3f}"
    )

    return size_bytes, latency


def upload_metadata(round_id, role, meta):
    """Upload JSON metadata for this round and role."""
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
        f"total_energy={meta.get('total_energy_j', 0.0):.3f} J"
    )

    log_event(
        f"[{role}] upload_metadata round={round_id} "
        f"latency_s={latency:.3f} size_bytes={len(body)} "
        f"bandwidth_mbps={meta.get('bandwidth_mbps', 0.0):.3f} "
        f"total_energy_j={meta.get('total_energy_j', 0.0):.4f}"
    )

    return latency
