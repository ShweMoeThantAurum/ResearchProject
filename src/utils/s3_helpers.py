"""
Small S3 helper utilities for experiment artefacts.

Used mainly by plotting scripts to upload generated PNGs and CSVs
into a central experiments/ prefix on S3.
"""

import os
from pathlib import Path

import boto3


def upload_to_s3(local_path, s3_bucket, s3_key):
    """
    Upload a local file to S3 at the given bucket/key.

    Prints a short confirmation message on success.
    """
    s3 = boto3.client("s3")
    local_path = Path(local_path)

    if not local_path.exists():
        raise FileNotFoundError(f"File not found: {local_path}")

    s3.upload_file(str(local_path), s3_bucket, s3_key)
    print(f"[S3] Uploaded {local_path} → s3://{s3_bucket}/{s3_key}")


def prefix_for_dataset(dataset, subfolder):
    """
    Build an S3 key prefix for experiment artefacts.

    Example:
      experiments/sz/plots/
    """
    dataset = str(dataset).strip().lower()
    subfolder = str(subfolder).strip().strip("/")
    return f"experiments/{dataset}/{subfolder}/"
