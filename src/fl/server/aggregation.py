"""
S3 input and output helpers for the server.

The server uses S3 for three things:
  - storing the global model for each round
  - downloading client updates
  - downloading per-client metadata used by AEFL selection
"""

import os
import io
import json
import boto3
import torch

from src.fl.utils.logger import log_event, Timer


def get_bucket_name():
    """
    Return the S3 bucket name used for FL artefacts.

    Uses S3_BUCKET environment variable with fallback "aefl".
    """
    return os.environ.get("S3_BUCKET", "aefl")


def get_round_prefix(dataset_name, mode_name):
    """
    Build the S3 prefix used to store all round artefacts.

    Example:
        fl/sz/aefl
    """
    dataset = dataset_name.lower()
    mode = mode_name.lower()
    return "fl/{}/{}".format(dataset, mode)


def create_s3_client():
    """
    Create and return a boto3 S3 client.

    Uses AWS_REGION environment variable with default "us-east-1".
    """
    region = os.environ.get("AWS_REGION", "us-east-1")
    return boto3.client("s3", region_name=region)


def clear_prefix(bucket, prefix):
    """
    Delete all objects under the given prefix in the bucket.

    This is used at the start of a training run so that rounds
    do not mix with any previous experiment.
    """
    s3 = create_s3_client()
    paginator = s3.get_paginator("list_objects_v2")

    deleted = 0
    timer = Timer()
    timer.start()

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        contents = page.get("Contents", [])
        if not contents:
            continue

        batch = [{"Key": obj["Key"]} for obj in contents]
        s3.delete_objects(Bucket=bucket, Delete={"Objects": batch})
        deleted += len(batch)

    elapsed = timer.stop()

    print(
        "[SERVER] Cleared {} objects under prefix {} in {:.3f}s".format(
            deleted, prefix, elapsed
        )
    )

    log_event(
        "server_cleanup.log",
        {
            "bucket": bucket,
            "prefix": prefix,
            "deleted_objects": deleted,
            "time_sec": elapsed,
        },
    )


def global_model_key(prefix, round_id):
    """
    Build the S3 key for the global model of a round.
    """
    return "{}/round_{}/global.pt".format(prefix, round_id)


def client_update_key(prefix, round_id, role):
    """
    Build the S3 key for a client update of a round.
    """
    return "{}/round_{}/updates/{}.pt".format(prefix, round_id, role)


def client_metadata_prefix(prefix, round_id):
    """
    Build the prefix that holds all metadata JSON objects for a round.
    """
    return "{}/round_{}/metadata/".format(prefix, round_id)


def upload_global_model(bucket, prefix, round_id, state_dict):
    """
    Upload the global model state dictionary for a given round to S3.

    The state_dict is written to a temporary local file and then
    uploaded to S3 with a measured latency and file size.
    """
    s3 = create_s3_client()
    key = global_model_key(prefix, round_id)

    tmp_path = "/tmp/global_round_{}.pt".format(round_id)
    torch.save(state_dict, tmp_path)

    size_bytes = os.path.getsize(tmp_path)

    timer = Timer()
    timer.start()
    s3.upload_file(tmp_path, bucket, key)
    latency = timer.stop()

    print(
        "[SERVER] Uploaded global model r={} size={:.3f} MB latency={:.3f}s".format(
            round_id, size_bytes / 1e6, latency
        )
    )

    log_event(
        "server_global_upload.log",
        {
            "round": round_id,
            "bucket": bucket,
            "key": key,
            "size_bytes": size_bytes,
            "latency_sec": latency,
        },
    )


def download_client_update(bucket, prefix, round_id, role):
    """
    Download a client update state dictionary for a given round and role.

    Returns a PyTorch state_dict if present, otherwise returns None.
    """
    s3 = create_s3_client()
    key = client_update_key(prefix, round_id, role)

    timer = Timer()
    try:
        timer.start()
        obj = s3.get_object(Bucket=bucket, Key=key)
        latency = timer.stop()

        body = obj["Body"].read()
        size_bytes = len(body)

        state = torch.load(io.BytesIO(body), map_location="cpu")

        print(
            "[SERVER] Downloaded update r={} role={} size={:.3f} MB latency={:.3f}s".format(
                round_id, role, size_bytes / 1e6, latency
            )
        )

        log_event(
            "server_update_download.log",
            {
                "round": round_id,
                "role": role,
                "bucket": bucket,
                "key": key,
                "size_bytes": size_bytes,
                "latency_sec": latency,
            },
        )

        return state

    except Exception:
        # The client may not have uploaded yet
        return None


def load_round_metadata(bucket, prefix, round_id):
    """
    Load all client metadata JSON objects for a given round.

    Returns a dictionary mapping client-role strings to metadata
    dictionaries. If nothing is found, an empty dictionary is returned.
    """
    s3 = create_s3_client()
    meta_prefix = client_metadata_prefix(prefix, round_id)

    try:
        listing = s3.list_objects_v2(Bucket=bucket, Prefix=meta_prefix)
        contents = listing.get("Contents", [])
        if not contents:
            return {}

        results = {}
        for obj in contents:
            key = obj["Key"]
            role = os.path.basename(key).replace(".json", "")

            body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
            meta = json.loads(body.decode("utf-8"))
            results[role] = meta

        return results

    except Exception:
        return {}
