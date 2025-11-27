"""Server-side utilities for S3 cleanup, model initialisation, and upload."""

import os
import json
import boto3
import torch
import numpy as np

from src.fl.logger import log_event, Timer
from src.fl.utils import get_bucket, get_prefix, get_hidden_size
from src.models.simple_gru import SimpleGRU


BUCKET = get_bucket()
PREFIX = get_prefix()
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
s3 = boto3.client("s3", region_name=AWS_REGION)


def clear_round_data(prefix: str = PREFIX, bucket: str = BUCKET):
    """
    Remove all S3 objects associated with previous FL rounds.

    Ensures that each new experiment starts with a clean S3 directory
    such as fl/sz/round_*/.
    """
    paginator = s3.get_paginator("list_objects_v2")
    deleted = 0
    timer = Timer()
    timer.start()

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        contents = page.get("Contents", [])
        if not contents:
            continue

        batch = [{"Key": o["Key"]} for o in contents]
        s3.delete_objects(Bucket=bucket, Delete={"Objects": batch})
        deleted += len(batch)

    elapsed = timer.stop()
    print(f"[SERVER] Cleared {deleted} objects under {prefix} ({elapsed:.3f}s)")

    log_event("server_cleanup.log", {
        "prefix": prefix,
        "deleted": deleted,
        "time_sec": elapsed,
    })

    return deleted


def infer_num_nodes(proc_dir: str):
    """
    Infer the number of graph nodes by reading X_train.npy.

    Ensures clients and server use the same input dimensionality.
    """
    X = np.load(os.path.join(proc_dir, "X_train.npy"))
    return X.shape[-1]


def init_global_model(num_nodes: int, hidden: int = None):
    """
    Initialise the global GRU model and return its state_dict.

    Hidden size is read from config if not explicitly supplied.
    """
    if hidden is None:
        hidden = get_hidden_size()

    print(f"[SERVER] num_nodes={num_nodes}, hidden={hidden}")
    model = SimpleGRU(num_nodes=num_nodes, hidden_size=hidden)
    return model.state_dict()


def store_global_model(state_dict: dict, round_id: int, prefix: str = PREFIX):
    """
    Upload the global model state_dict to S3 for a given round.

    This makes the model available to all clients for that round.
    """
    key = f"{prefix}/round_{round_id}/global.pt"
    temp = f"/tmp/global_round_{round_id}.pt"

    torch.save(state_dict, temp)

    timer = Timer()
    timer.start()
    s3.upload_file(temp, BUCKET, key)
    latency = timer.stop()

    size_bytes = os.path.getsize(temp)

    print(f"[SERVER] Uploaded global model r={round_id} "
          f"({size_bytes/1e6:.3f} MB, {latency:.3f}s)")

    log_event("server_model_upload.log", {
        "round": round_id,
        "key": key,
        "size_bytes": size_bytes,
        "latency_sec": latency,
    })
