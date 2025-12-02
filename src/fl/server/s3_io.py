"""
S3 read/write utilities for federated learning experiments.
"""

import os
import boto3
import torch
import json

AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")

s3 = boto3.client("s3", region_name=AWS_REGION)


def _key(dataset, mode, round_id, name):
    return "experiments/%s/%s/round_%d/%s" % (dataset, mode, round_id, name)


def upload_bytes(bucket, dataset, mode, round_id, name, data):
    key = _key(dataset, mode, round_id, name)
    s3.put_object(Bucket=bucket, Key=key, Body=data)


def download_bytes(bucket, dataset, mode, round_id, name):
    key = _key(dataset, mode, round_id, name)
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
        return obj["Body"].read()
    except s3.exceptions.NoSuchKey:
        return None
    except Exception:
        return None


def store_global_model(state, dataset, mode, round_id):
    tmp = "/tmp/global_%d.pth" % round_id
    torch.save(state, tmp)
    with open(tmp, "rb") as f:
        upload_bytes(
            os.environ.get("S3_BUCKET", "aefl"),
            dataset,
            mode,
            round_id,
            "global_model",
            f.read(),
        )


def load_client_update(bucket, dataset, mode, round_id, role):
    name = "%s_update" % role
    data = download_bytes(bucket, dataset, mode, round_id, name)
    if data is None:
        return None

    tmp = "/tmp/%s_r%d.pth" % (role, round_id)
    with open(tmp, "wb") as f:
        f.write(data)

    return torch.load(tmp, map_location="cpu")


def load_round_metadata(bucket, dataset, mode, round_id):
    """
    Load metadata JSON files uploaded by clients.
    """
    meta = {}
    for role in ["roadside", "vehicle", "sensor", "camera", "bus"]:
        name = "%s_metadata.json" % role
        data = download_bytes(bucket, dataset, mode, round_id, name)
        if data is None:
            continue
        try:
            meta[role] = json.loads(data.decode("utf-8"))
        except Exception:
            continue
    return meta
