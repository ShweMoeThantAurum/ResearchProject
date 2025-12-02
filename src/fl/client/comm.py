"""
Client-side communication (download global model + upload update).
"""

import time
import os
import torch
from src.fl.server.s3_io import download_bytes, upload_bytes


def download_global_model(bucket, dataset, mode, round_id):
    """
    Poll S3 until the global model for this round is available.
    """
    key_name = "global_model"
    while True:
        data = download_bytes(bucket, dataset, mode, round_id, key_name)
        if data is not None:
            tmp = os.path.join("/tmp", "%s_r%d.pth" % (key_name, round_id))
            with open(tmp, "wb") as f:
                f.write(data)
            return torch.load(tmp, map_location="cpu")
        time.sleep(1)


def upload_update(bucket, dataset, mode, round_id, role, update_dict):
    """
    Upload the local update to S3.
    """
    tmp = "/tmp/%s_update.pth" % role
    torch.save(update_dict, tmp)
    with open(tmp, "rb") as f:
        upload_bytes(bucket, dataset, mode, round_id, "%s_update" % role, f.read())
