"""
Server-side shared helpers: roles, S3 paths, env helpers.
"""

import os

ROLES = ["roadside", "vehicle", "sensor", "camera", "bus"]


def get_s3_bucket():
    return os.environ.get("S3_BUCKET", "aefl")


def get_s3_prefix():
    dataset = os.environ.get("DATASET", "sz").lower()
    mode = os.environ.get("FL_MODE", "AEFL").upper()
    return f"fl/{dataset}/{mode}"
