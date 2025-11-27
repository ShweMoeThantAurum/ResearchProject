"""
Client-side S3 utilities:
- Download global model
- Upload RAW model update (for Lambda offload)
- Upload FINALūr processed update (if Lambda is disabled)
- Upload metadata
"""

import os
import json
import time
from typing import Tuple

import boto3
import torch

from src.fl.logger import log_event, Timer
from src.fl.utils import get_bucket, get_prefix


BUCKET = get_bucket()
PREFIX = get_prefix()
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
s3 = boto3.client("s3", region_name=AWS_REGION)


# ---------------------------------------------------------------------
# S3 key helpers
# ---------------------------------------------------------------------

def global_key(round_id: int) -> str:
    return f"{PREFIX}/round_{round_id}/global.pt"


def raw_update_key(round_id: int, role: str) -> str:
    return f"{PREFIX}/round_{round_id}/raw_updates/{role}.pt"


def processed_update_key(round_id: int, role: str) -> str:
    return f"{PREFIX}/round_{round_id}/updates/{role}.pt"


def metadata_key(round_id: int, role: str) -> str:
    return f"{PREFIX}/round_{round_id}/metadata/{role}.json"


# ---------------------------------------------------------------------
# Download global model
# ---------------------------------------------------------------------

def download_global(round_id: int, role: str) -> Tuple[str, int]:
    """
    Waits until the server uploads the global model for this round.
    Downloads it into /tmp and returns (local_path, num_bytes).
    """
    key = global_key(round_id)
    local_path = f"/tmp/global_{role}_round_{round_id}.pt"

    while True:
        timer = Timer()
        timer.start()
        try:
            s3.download_file(BUCKET, key, local_path)
            latency = timer.stop()
            size_bytes = os.path.getsize(local_path)

            log_event("client_s3_download.log", {
                "role": role,
                "round": round_id,
                "latency_sec": latency,
                "size_bytes": size_bytes,
                "s3_key": key,
            })

            print(
                f"[{role}] Downloaded global model for round {round_id} "
                f"(size={size_bytes / (1024**2):.3f} MB, latency={latency:.3f}s)"
            )
            return local_path, size_bytes

        except Exception:
            print(f"[{role}] Waiting for global model round {round_id}...")
            time.sleep(3.0)


# ---------------------------------------------------------------------
# Upload RAW update (for Lambda offload)
# ---------------------------------------------------------------------

def upload_raw_update(round_id: int, role: str, state_dict: dict) -> Tuple[int, float]:
    """
    Saves the uncompressed, non-DP, raw update locally and uploads to:
      fl/<dataset>/round_r/raw_updates/<role>.pt

    This is used ONLY when Lambda offloading is enabled.
    """
    local_path = f"/tmp/raw_update_{role}_round_{round_id}.pt"
    torch.save(state_dict, local_path)

    key = raw_update_key(round_id, role)

    timer = Timer()
    timer.start()
    s3.upload_file(local_path, BUCKET, key)
    latency = timer.stop()

    size_bytes = os.path.getsize(local_path)

    log_event("client_raw_upload.log", {
        "role": role,
        "round": round_id,
        "latency_sec": latency,
        "size_bytes": size_bytes,
        "s3_key": key,
    })

    print(
        f"[{role}] Uploaded RAW update for round {round_id} "
        f"(size={size_bytes / (1024**2):.3f} MB, latency={latency:.3f}s)"
    )

    return size_bytes, latency


# ---------------------------------------------------------------------
# Upload PROCESSED update (when Lambda is OFF)
# ---------------------------------------------------------------------

def upload_processed_update(round_id: int, role: str, state_dict: dict) -> Tuple[int, float]:
    """
    Uploads the (compressed/DP) processed update to:
      fl/<dataset>/round_r/updates/<role>.pt

    This is used when Lambda offloading is NOT enabled.
    """
    local_path = f"/tmp/update_{role}_round_{round_id}.pt"
    torch.save(state_dict, local_path)

    key = processed_update_key(round_id, role)

    timer = Timer()
    timer.start()
    s3.upload_file(local_path, BUCKET, key)
    latency = timer.stop()

    size_bytes = os.path.getsize(local_path)

    log_event("client_s3_upload.log", {
        "role": role,
        "round": round_id,
        "latency_sec": latency,
        "size_bytes": size_bytes,
        "s3_key": key,
    })

    print(
        f"[{role}] Uploaded PROCESSED update for round {round_id} "
        f"(size={size_bytes / (1024**2):.3f} MB, latency={latency:.3f}s)"
    )

    return size_bytes, latency


# ---------------------------------------------------------------------
# Upload metadata
# ---------------------------------------------------------------------

def upload_metadata(round_id: int, role: str, meta: dict):
    """
    Uploads metadata containing energy, communication, bandwidth.
    """
    key = metadata_key(round_id, role)
    body = json.dumps(meta).encode("utf-8")

    timer = Timer()
    timer.start()
    s3.put_object(Bucket=BUCKET, Key=key, Body=body)
    latency = timer.stop()

    log_event("client_meta_upload.log", {
        "role": role,
        "round": round_id,
        "latency_sec": latency,
        "size_bytes": len(body),
        "s3_key": key,
    })

    print(
        f"[{role}] Uploaded metadata for round {round_id}: "
        f"bandwidth={meta.get('bandwidth_mbps', 0):.3f} Mb/s, "
        f"total_energy={meta.get('total_energy_j', 0):.2f} J"
    )