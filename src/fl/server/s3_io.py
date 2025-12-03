"""
S3 helpers for global/updates/metadata.
"""

import io
import json
import boto3
import torch
import os

from src.fl.server.utils_server import get_s3_bucket, get_s3_prefix


def _client():
    region = os.environ.get("AWS_REGION", "us-east-1")
    return boto3.client("s3", region_name=region)


def _key_global(round_id):
    prefix = get_s3_prefix()
    return f"{prefix}/round_{round_id}/global.pt"


def _key_update(round_id, role):
    prefix = get_s3_prefix()
    return f"{prefix}/round_{round_id}/updates/{role}.pt"


def _key_meta_prefix(round_id):
    prefix = get_s3_prefix()
    return f"{prefix}/round_{round_id}/metadata/"


def upload_global_model(round_id, state_dict):
    """Upload round's global model."""
    bucket = get_s3_bucket()
    key = _key_global(round_id)
    s3 = _client()

    buf = io.BytesIO()
    torch.save(state_dict, buf)
    body = buf.getvalue()

    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=body,
        ContentType="application/octet-stream"
    )

    print(f"[SERVER] Uploaded global model r={round_id} ({len(body)/1e6:.3f} MB)")


def download_client_update(round_id, role):
    """Download processed client update or return None."""
    bucket = get_s3_bucket()
    key = _key_update(round_id, role)
    s3 = _client()

    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
    except Exception:
        return None

    raw = obj["Body"].read()
    buf = io.BytesIO(raw)
    return torch.load(buf, map_location="cpu")


def load_round_metadata(round_id):
    """Load metadata JSON for all clients."""
    bucket = get_s3_bucket()
    prefix = _key_meta_prefix(round_id)
    s3 = _client()

    try:
        listing = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
    except Exception:
        return {}

    if "Contents" not in listing:
        return {}

    result = {}
    for obj in listing["Contents"]:
        key = obj["Key"]
        body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
        meta = json.loads(body.decode("utf-8"))
        role = os.path.basename(key).replace(".json", "")
        result[role] = meta

    return result
