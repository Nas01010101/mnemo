"""Alibaba Cloud OSS integration — durable snapshots of the Mnemo memory store.

This is the submission's "code file that demonstrates use of Alibaba Cloud services
and APIs" (Proof of Alibaba Cloud Deployment). It uses Alibaba Cloud Object Storage
Service (OSS) via the official `oss2` SDK to back up and restore the sqlite memory DB,
so a Mnemo backend running on Alibaba Cloud (ECS / Function Compute) can persist and
recover its memory across restarts.

Env (see .env.example):
  ALIBABA_CLOUD_ACCESS_KEY_ID / _SECRET, OSS_ENDPOINT, OSS_BUCKET
"""
from __future__ import annotations

from pathlib import Path

import config


def _bucket():
    import oss2  # imported lazily so the rest of the app runs without the SDK
    auth = oss2.Auth(
        config.require("ALIBABA_CLOUD_ACCESS_KEY_ID"),
        config.require("ALIBABA_CLOUD_ACCESS_KEY_SECRET"),
    )
    endpoint = config.require("OSS_ENDPOINT")   # e.g. https://oss-ap-southeast-1.aliyuncs.com
    bucket_name = config.require("OSS_BUCKET")
    return oss2.Bucket(auth, endpoint, bucket_name)


def snapshot(db_path: str | Path, object_key: str = "mnemo/snapshot.db") -> str:
    """Upload the memory DB to Alibaba Cloud OSS. Returns the object key."""
    b = _bucket()
    b.put_object_from_file(object_key, str(db_path))
    return object_key


def restore(db_path: str | Path, object_key: str = "mnemo/snapshot.db") -> bool:
    """Download the latest memory DB snapshot from OSS if it exists."""
    b = _bucket()
    import oss2
    if not b.object_exists(object_key):
        return False
    b.get_object_to_file(object_key, str(db_path))
    return True


if __name__ == "__main__":  # tiny CLI: python src/alicloud_oss.py snapshot <db>
    import sys
    action, path = (sys.argv[1], sys.argv[2]) if len(sys.argv) > 2 else ("snapshot", "data/mnemo.db")
    if action == "snapshot":
        print("uploaded ->", snapshot(path))
    elif action == "restore":
        print("restored:", restore(path))
