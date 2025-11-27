"""Server-side utilities for downloading client updates and metadata from S3."""

import json
import boto3
import torch
import os
import io

from src.fl.logger import log_event, Timer
from src.fl.utils import get_bucket, get_prefix


BUCKET = get_bucket()
PREFIX = get_prefix()
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")

s3 = boto3.client("s3", region_name=AWS_REGION)


def load_client_update(round_id: int, role: str, prefix: str = PREFIX):
    """
    Download a client update from S3 and return its state_dict.

    Expected key:
        <prefix>/round_<r>/updates/<role>.pt
    Returns None if not yet uploaded.
    """
    key = f"{prefix}/round_{round_id}/updates/{role}.pt"
    timer = Timer()

    try:
        timer.start()
        obj = s3.get_object(Bucket=BUCKET, Key=key)
        latency = timer.stop()
        raw = obj["Body"].read()
        size_bytes = len(raw)

        state = torch.load(io.BytesIO(raw), map_location="cpu")

        log_event("server_update_download.log", {
            "round": round_id,
            "role": role,
            "size_bytes": size_bytes,
            "latency_sec": latency,
        })

        print(f"[SERVER] Downloaded update from {role} r={round_id} "
              f"({size_bytes/1e6:.3f} MB, {latency:.3f}s)")
        return state

    except Exception:
        return None


def load_round_metadata(round_id: int, prefix: str = PREFIX):
    """
    Load per-client metadata JSON files for a given round.

    Expected key prefix:
        <prefix>/round_<r>/metadata/*.json
    """
    meta_prefix = f"{prefix}/round_{round_id}/metadata/"

    try:
        listing = s3.list_objects_v2(Bucket=BUCKET, Prefix=meta_prefix)
        if "Contents" not in listing:
            return {}

        results = {}
        for obj in listing["Contents"]:
            key = obj["Key"]
            role = key.split("/")[-1].replace(".json", "")
            raw = s3.get_object(Bucket=BUCKET, Key=key)["Body"].read()
            results[role] = json.loads(raw.decode("utf-8"))

        return results

    except Exception:
        return {}