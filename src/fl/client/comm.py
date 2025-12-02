"""
Communication layer for clients.

Handles:
    - download of global model from S3
    - upload of processed local update
    - upload of metadata
"""

import time
import os
import json
import torch
import boto3

AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
s3 = boto3.client("s3", region_name=AWS_REGION)


def _key(dataset, mode, round_id, name):
    """
    Build S3 key for FL artifact.
    """
    return "experiments/%s/%s/round_%d/%s" % (dataset, mode, round_id, name)


def download_global_model(bucket, dataset, mode, round_id, timeout=600):
    """
    Poll S3 until the global model for this round is available.
    """
    name = "global_model"
    start = time.time()

    while True:
        try:
            obj = s3.get_object(
                Bucket=bucket,
                Key=_key(dataset, mode, round_id, name),
            )
            data = obj["Body"].read()

            tmp = "/tmp/global_%d.pth" % round_id
            with open(tmp, "wb") as f:
                f.write(data)

            return torch.load(tmp, map_location="cpu")

        except Exception:
            if time.time() - start > timeout:
                raise TimeoutError("Timeout waiting for global model r=%d" %
                                   round_id)
            time.sleep(2)


def upload_client_update(bucket, dataset, mode, round_id, role, state_dict):
    """
    Upload a serialized client model update to S3.
    """
    tmp = "/tmp/%s_r%d.pth" % (role, round_id)
    torch.save(state_dict, tmp)

    with open(tmp, "rb") as f:
        s3.put_object(
            Bucket=bucket,
            Key=_key(dataset, mode, round_id, "%s_update" % role),
            Body=f.read(),
        )


def upload_client_metadata(bucket, dataset, mode, round_id, role, metadata):
    """
    Upload per-round client metadata as JSON.
    """
    s3.put_object(
        Bucket=bucket,
        Key=_key(dataset, mode, round_id, "%s_metadata.json" % role),
        Body=json.dumps(metadata).encode("utf-8"),
    )
