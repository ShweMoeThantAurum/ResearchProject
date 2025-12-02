"""
S3 helper functions for uploading/downloading model updates and metadata.

Used by:
- server_main.py
- client_main.py
"""

import os
import json
import boto3

s3 = boto3.client("s3")


def s3_key(dataset, mode, round_id, name):
    """Construct a consistent S3 key path."""
    return "%s/%s/round_%d/%s" % (dataset, mode.lower(), round_id, name)


def upload_json(bucket, dataset, mode, round_id, name, data):
    """Upload a JSON object to S3."""
    key = s3_key(dataset, mode, round_id, name)
    s3.put_object(Bucket=bucket, Key=key, Body=json.dumps(data))


def download_json(bucket, dataset, mode, round_id, name):
    """Download JSON object, returning None if missing."""
    key = s3_key(dataset, mode, round_id, name)
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
        return json.loads(obj["Body"].read())
    except Exception:
        return None


def upload_bytes(bucket, dataset, mode, round_id, name, data_bytes):
    """Upload bytes (model update) to S3."""
    key = s3_key(dataset, mode, round_id, name)
    s3.put_object(Bucket=bucket, Key=key, Body=data_bytes)


def download_bytes(bucket, dataset, mode, round_id, name):
    """Download raw byte data, returning None if not found."""
    key = s3_key(dataset, mode, round_id, name)
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
        return obj["Body"].read()
    except Exception:
        return None
