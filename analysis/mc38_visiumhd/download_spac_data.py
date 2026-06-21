#!/usr/bin/env python3
"""Download SPAC-seq Visium HD subQ datasets from spac.pku-genomics.org."""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path


API = "https://spac.pku-genomics.org/spac/download/spatial/pageInfo"
ALIYUN_TOKEN = "https://bj21400.api.aliyunfile.com/v2/share_link/get_share_token"
ALIYUN_LIST = "https://bj21400.api.aliyunfile.com/v2/file/list"


def list_datasets(dataset_type: int = 2) -> list[dict]:
    data = urllib.parse.urlencode({"type": dataset_type, "page": 1, "limit": 50}).encode()
    req = urllib.request.Request(API, data=data, method="POST")
    with urllib.request.urlopen(req) as resp:
        payload = json.load(resp)
    return payload["data"]["records"]


def download_share(share_id: str, dest: Path) -> None:
    body = json.dumps({"share_id": share_id, "ignoreError": True}).encode()
    req = urllib.request.Request(
        ALIYUN_TOKEN, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req) as resp:
        token = json.load(resp)["share_token"]
    listing_body = json.dumps(
        {
            "limit": 100,
            "marker": "",
            "share_id": share_id,
            "parent_file_id": "root",
            "fields": "user_name,dir_size,url",
            "url_expire_sec": 7200,
        }
    ).encode()
    req2 = urllib.request.Request(
        ALIYUN_LIST,
        data=listing_body,
        headers={"Content-Type": "application/json", "x-share-token": token},
        method="POST",
    )
    with urllib.request.urlopen(req2) as resp:
        items = json.load(resp)["items"]
    for item in items:
        out = dest / item["name"]
        if out.exists():
            print(f"skip {out}")
            continue
        print(f"downloading {out.name} ...")
        urllib.request.urlretrieve(item["download_url"], out)
        print(f"saved {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="subQ-1", help="Dataset name from SPAC portal")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--components",
        nargs="+",
        default=["transcriptome", "perturbation", "segmentation", "raw"],
    )
    args = parser.parse_args()

    records = list_datasets(dataset_type=2)
    match = next((r for r in records if r["name"] == args.name), None)
    if match is None:
        raise SystemExit(f"Dataset {args.name!r} not found. Available: {[r['name'] for r in records]}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for key in args.components:
        share_id = match[key]
        download_share(share_id, args.out_dir)


if __name__ == "__main__":
    import urllib.parse

    main()
