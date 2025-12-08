"""
Client-side S3 I/O utilities.

Handles:
 - downloading global models from the server
 - uploading processed updates
 - uploading metadata per round
"""

import os
import json
import time
import boto3
import torch

from src.fl.logger import log_event, Timer
from src.fl.utils import get_bucket, get_prefix


BUCKET = get_bucket()
PREFIX = get_prefix()
s3 = boto3.client("s3", region_name=os.environ.get("AWS_REGION", "us-east-1"))


def global_key(r):
    return f"{PREFIX}/round_{r}/global.pt"


def processed_update_key(r, role):
    return f"{PREFIX}/round_{r}/updates/{role}.pt"


def metadata_key(r, role):
    return f"{PREFIX}/round_{r}/metadata/{role}.json"


def download_global(round_id, role):
    """
    Block until global model for this round is available.
    """
    key = global_key(round_id)
    local = f"/tmp/global_{role}_round_{round_id}.pt"

    dataset = os.environ.get("DATASET", "unknown")
    mode = os.environ.get("FL_MODE", "AEFL").lower()
    variant = os.environ.get("VARIANT_ID", "")

    while True:
        timer = Timer()
        timer.start()
        try:
            s3.download_file(BUCKET, key, local)
            latency = timer.stop()
            size = os.path.getsize(local)

            log_event(
                "client_s3_download.log",
                {
                    "role": role,
                    "round": round_id,
                    "dataset": dataset,
                    "mode": mode,
                    "variant": variant,
                    "latency_sec": latency,
                    "size_bytes": size,
                    "s3_key": key,
                },
            )

            print(
                f"[{role}] Downloaded global r={round_id} "
                f"({size/1e6:.3f} MB, {latency:.3f}s)"
            )
            return local, size

        except Exception:
            print(f"[{role}] Waiting for global r={round_id}...")
            time.sleep(3)


def upload_processed_update(round_id, role, state_dict):
    """
    Upload processed update (DP/Compression applied).
    """
    local = f"/tmp/update_{role}_round_{round_id}.pt"
    torch.save(state_dict, local)

    key = processed_update_key(round_id, role)
    timer = Timer()
    timer.start()
    s3.upload_file(local, BUCKET, key)
    latency = timer.stop()

    size = os.path.getsize(local)

    log_event(
        "client_s3_upload.log",
        {
            "role": role,
            "round": round_id,
            "size_bytes": size,
            "latency_sec": latency,
            "s3_key": key,
        },
    )

    print(
        f"[{role}] Uploaded update r={round_id} " f"({size/1e6:.3f} MB, {latency:.3f}s)"
    )
    return size, latency


def upload_metadata(round_id, role, meta):
    """
    Upload JSON metadata for this round.
    """
    key = metadata_key(round_id, role)
    body = json.dumps(meta).encode()

    timer = Timer()
    timer.start()
    s3.put_object(Bucket=BUCKET, Key=key, Body=body)
    latency = timer.stop()

    print(
        f"[{role}] Uploaded metadata r={round_id}: "
        f"bandwidth={meta.get('bandwidth_mbps', 0):.3f} Mb/s, "
        f"energy={meta.get('total_energy_j', 0):.2f} J"
    )
