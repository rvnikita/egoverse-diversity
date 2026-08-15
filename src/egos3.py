"""Download EgoVerse objects from Cloudflare R2 using the read-only public creds.

Credentials come from ~/.egoverse_env (written by setup_secret.sh). This talks to R2
directly via boto3 — no `egomimic`, no s5cmd required.

    from egos3 import fetch
    fetch("s3://rldb/processed_v3/rl2/<hash>_video.mp4", "out/clip.mp4")
"""

from __future__ import annotations

import os
import pathlib

from egodb import load_env


def client():
    import boto3
    from botocore.config import Config

    load_env()
    # Two R2 quirks, both of which fail with an unhelpful 400/InvalidArgument:
    #  1. Do NOT pass R2_SESSION_TOKEN. setup_secret.sh writes one, but these are
    #     permanent R2 keys and R2 rejects X-Amz-Security-Token outright.
    #  2. boto3 >= 1.36 adds integrity checksums by default that R2 does not accept;
    #     pin them back to "when_required".
    return boto3.client(
        "s3",
        endpoint_url=os.environ["AWS_ENDPOINT_URL_S3"],
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        region_name="auto",
        config=Config(
            signature_version="s3v4",
            retries={"max_attempts": 3},
            request_checksum_calculation="when_required",
            response_checksum_validation="when_required",
        ),
    )


def split_uri(uri: str) -> tuple[str, str]:
    assert uri.startswith("s3://"), uri
    bucket, _, key = uri[5:].partition("/")
    return bucket, key


def fetch(uri: str, dest: str | pathlib.Path, cl=None) -> pathlib.Path:
    """Download one object. Skips work if the file already exists non-empty."""
    dest = pathlib.Path(dest)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    bucket, key = split_uri(uri)
    (cl or client()).download_file(bucket, key, str(dest))
    return dest


def fetch_many(uris: list[str], dest_dir: str | pathlib.Path,
               workers: int = 8) -> list[pathlib.Path]:
    """Parallel download; returns the paths that succeeded."""
    from concurrent.futures import ThreadPoolExecutor

    dest_dir = pathlib.Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    cl = client()  # botocore clients are thread-safe for reads

    def one(uri: str):
        try:
            return fetch(uri, dest_dir / pathlib.Path(split_uri(uri)[1]).name, cl=cl)
        except Exception as exc:  # noqa: BLE001 - report and continue
            print(f"  FAILED {uri}: {type(exc).__name__}: {str(exc)[:120]}")
            return None

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return [p for p in pool.map(one, uris) if p is not None]


def head(uri: str) -> dict:
    bucket, key = split_uri(uri)
    return client().head_object(Bucket=bucket, Key=key)
