"""
S3 utilities for server-side FL orchestration.
Handles global model upload and client update/metadata downloads.
"""

import io
import json
import os
import boto3
import torch

from ..utils.logger import log_event
from .utils_server import get_s3_bucket, get_s3_prefix, get_results_bucket


def _s3_client():
    """Create a boto3 S3 client."""
    region = os.environ.get("AWS_REGION", "us-east-1")
    return boto3.client("s3", region_name=region)


def _global_key(round_id):
    """Return S3 key for global model at a round."""
    prefix = get_s3_prefix()
    return f"{prefix}/round_{round_id}/global.pt"


def _update_key(round_id, role):
    """Return S3 key for a processed client update."""
    prefix = get_s3_prefix()
    return f"{prefix}/round_{round_id}/updates/{role}.pt"


def _metadata_prefix(round_id):
    """Return S3 prefix under which metadata JSONs are stored."""
    prefix = get_s3_prefix()
    return f"{prefix}/round_{round_id}/metadata/"


def clear_all_rounds():
    """Delete all S3 objects under the dataset FL prefix."""
    bucket = get_s3_bucket()
    prefix = get_s3_prefix()
    s3 = _s3_client()

    paginator = s3.get_paginator("list_objects_v2")
    deleted = 0

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        contents = page.get("Contents", [])
        if not contents:
            continue
        batch = [{"Key": obj["Key"]} for obj in contents]
        s3.delete_objects(Bucket=bucket, Delete={"Objects": batch})
        deleted += len(batch)

    log_event(f"[SERVER] Cleared {deleted} S3 objects under {prefix}")


def upload_global_model(round_id, state_dict):
    """Upload global model state dict to S3 for a given round."""
    bucket = get_s3_bucket()
    key = _global_key(round_id)
    s3 = _s3_client()

    buf = io.BytesIO()
    torch.save(state_dict, buf)
    buf.seek(0)

    s3.put_object(Bucket=bucket, Key=key, Body=buf.getvalue())
    log_event(f"[SERVER] Uploaded global model for round {round_id} to s3://{bucket}/{key}")


def download_client_update(round_id, role):
    """Download a client update state dict from S3."""
    bucket = get_s3_bucket()
    key = _update_key(round_id, role)
    s3 = _s3_client()

    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
    except Exception:
        return None

    raw = obj["Body"].read()
    buf = io.BytesIO(raw)
    state = torch.load(buf, map_location="cpu")

    log_event(f"[SERVER] Downloaded update for round {round_id} from role={role}")
    return state


def load_round_metadata(round_id):
    """Load per-client metadata JSON for a given round."""
    bucket = get_s3_bucket()
    prefix = _metadata_prefix(round_id)
    s3 = _s3_client()

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
        role_name = os.path.basename(key).replace(".json", "")
        result[role_name] = meta

    return result


def upload_results_artifact(local_path, remote_key):
    """Upload a local summary artifact to the results bucket."""
    bucket = get_results_bucket()
    s3 = _s3_client()

    if not os.path.exists(local_path):
        return

    with open(local_path, "rb") as f:
        data = f.read()

    s3.put_object(Bucket=bucket, Key=remote_key, Body=data)
    log_event(f"[SERVER] Uploaded results artifact to s3://{bucket}/{remote_key}")
