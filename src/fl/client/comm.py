"""
Client-side communication utilities for S3-based model exchange.

This module:
    - Downloads the global model for each round
    - Uploads the processed client update
    - Uploads per-round metadata for AEFL selection
"""

import os
import json
import time
import io

import boto3
import torch

from src.fl.utils.logger import log_event, Timer
from src.fl.utils.serialization import save_state_dict_to_path
from src.fl.client.utils_client import get_dataset_name


def get_s3_client():
    """
    Create and return a boto3 S3 client using AWS_REGION from the environment.
    """
    region = os.environ.get("AWS_REGION", "us-east-1")
    return boto3.client("s3", region_name=region)


def get_s3_bucket():
    """
    Return the S3 bucket name for federated learning artifacts.

    Reads S3_BUCKET from environment variables, defaults to "aefl".
    """
    return os.environ.get("S3_BUCKET", "aefl").strip()


def fl_prefix_for_dataset(dataset_name):
    """
    Return the S3 key prefix used for a given dataset.

    Example:
        dataset "sz" uses prefix "fl/sz"
    """
    return "fl/%s" % dataset_name


def global_model_key(dataset_name, round_id):
    """
    Return the S3 key for the global model at a given round.
    """
    prefix = fl_prefix_for_dataset(dataset_name)
    return "%s/round_%d/global.pt" % (prefix, round_id)


def client_update_key(dataset_name, round_id, role):
    """
    Return the S3 key for a processed client update.
    """
    prefix = fl_prefix_for_dataset(dataset_name)
    return "%s/round_%d/updates/%s.pt" % (prefix, round_id, role)


def client_metadata_key(dataset_name, round_id, role):
    """
    Return the S3 key for per-round client metadata.
    """
    prefix = fl_prefix_for_dataset(dataset_name)
    return "%s/round_%d/metadata/%s.json" % (prefix, round_id, role)


def download_global_model(round_id, role, dataset_name=None):
    """
    Download the global model for a specific round and client role.

    Blocks until the object is available on S3.

    Returns:
        state_dict     : PyTorch state_dict for the global model
        download_bytes : number of bytes downloaded
    """
    if dataset_name is None:
        dataset_name = get_dataset_name()

    s3 = get_s3_client()
    bucket = get_s3_bucket()
    key = global_model_key(dataset_name, round_id)

    local_path = "/tmp/global_%s_round_%d.pt" % (role, round_id)

    while True:
        timer = Timer()
        timer.start()
        try:
            s3.download_file(bucket, key, local_path)
            latency = timer.stop()
            size_bytes = os.path.getsize(local_path)

            log_event("client_s3_download.log", {
                "role": role,
                "round": round_id,
                "latency_sec": latency,
                "size_bytes": size_bytes,
                "s3_key": key,
            })

            print("[%s] Downloaded global model for round %d (size=%.3f MB, latency=%.3fs)"
                  % (role, round_id, size_bytes / (1024.0 * 1024.0), latency))

            state_dict = torch.load(local_path, map_location="cpu")
            return state_dict, size_bytes

        except Exception:
            print("[%s] Waiting for global model round %d..." % (role, round_id))
            time.sleep(3.0)


def upload_client_update(round_id,
                         role,
                         state_dict,
                         dataset_name=None):
    """
    Upload a processed client update state_dict to S3.

    Returns:
        size_bytes : uploaded payload size in bytes
        latency    : upload latency in seconds
    """
    if dataset_name is None:
        dataset_name = get_dataset_name()

    s3 = get_s3_client()
    bucket = get_s3_bucket()
    key = client_update_key(dataset_name, round_id, role)

    local_path = "/tmp/update_%s_round_%d.pt" % (role, round_id)
    save_state_dict_to_path(state_dict, local_path)

    size_bytes = os.path.getsize(local_path)

    timer = Timer()
    timer.start()
    s3.upload_file(local_path, bucket, key)
    latency = timer.stop()

    log_event("client_s3_upload.log", {
        "role": role,
        "round": round_id,
        "latency_sec": latency,
        "size_bytes": size_bytes,
        "s3_key": key,
    })

    print("[%s] Uploaded processed update for round %d (size=%.3f MB, latency=%.3fs)"
          % (role, round_id, size_bytes / (1024.0 * 1024.0), latency))

    return size_bytes, latency


def upload_client_metadata(round_id,
                           role,
                           meta,
                           dataset_name=None):
    """
    Upload client metadata as a small JSON document to S3.

    Metadata includes:
        - energy statistics
        - communication volume
        - training loss and samples
        - bandwidth estimate
    """
    if dataset_name is None:
        dataset_name = get_dataset_name()

    s3 = get_s3_client()
    bucket = get_s3_bucket()
    key = client_metadata_key(dataset_name, round_id, role)

    body = json.dumps(meta).encode("utf-8")

    timer = Timer()
    timer.start()
    s3.put_object(Bucket=bucket, Key=key, Body=body)
    latency = timer.stop()

    log_event("client_meta_upload.log", {
        "role": role,
        "round": round_id,
        "latency_sec": latency,
        "size_bytes": len(body),
        "s3_key": key,
    })

    bw_mbps = meta.get("bandwidth_mbps", 0.0)

    print("[%s] Uploaded metadata for round %d: bandwidth=%.3f Mb/s, total_energy=%.2f J"
          % (role, round_id, bw_mbps, meta.get("total_energy_j", 0.0)))

    return latency
