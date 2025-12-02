"""
Server initialisation utilities.

Handles cleaning the S3 directory, creating the initial global model,
and uploading round-specific global models.
"""

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


def clear_round_data(prefix=PREFIX, bucket=BUCKET):
    """Delete all S3 objects under the given prefix."""
    paginator = s3.get_paginator("list_objects_v2")
    deleted = 0
    timer = Timer()
    timer.start()

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        contents = page.get("Contents", [])
        if contents:
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


def infer_num_nodes(proc_dir):
    """Infer node count from the shape of X_train.npy."""
    X = np.load(os.path.join(proc_dir, "X_train.npy"))
    return X.shape[-1]


def init_global_model(num_nodes, hidden=None):
    """Initialise the GRU model and return its state_dict."""
    if hidden is None:
        hidden = get_hidden_size()
    print(f"[SERVER] num_nodes={num_nodes}, hidden={hidden}")
    model = SimpleGRU(num_nodes=num_nodes, hidden_size=hidden)
    return model.state_dict()


def store_global_model(state_dict, round_id, prefix=PREFIX):
    """Upload the global model state_dict for the given round."""
    key = f"{prefix}/round_{round_id}/global.pt"
    temp = f"/tmp/global_round_{round_id}.pt"

    torch.save(state_dict, temp)

    timer = Timer()
    timer.start()
    s3.upload_file(temp, BUCKET, key)
    latency = timer.stop()

    size_bytes = os.path.getsize(temp)

    print(f"[SERVER] Uploaded global model r={round_id} ({size_bytes/1e6:.3f} MB, {latency:.3f}s)")

    log_event("server_model_upload.log", {
        "round": round_id,
        "key": key,
        "size_bytes": size_bytes,
        "latency_sec": latency,
    })
