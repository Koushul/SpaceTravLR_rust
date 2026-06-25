#!/usr/bin/env python3
"""Download SPAC-seq Visium HD datasets from spac.pku-genomics.org.

The ``raw`` component contains ``tissue_hires_image.png`` and
``scalefactors_json.json`` used by ``spatial_histology.attach_histology``.
"""

from __future__ import annotations

import argparse
import json
import shutil
import urllib.parse
import urllib.request
import zipfile
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


def _flatten_extracted(extract_dir: Path, inner_name: str) -> None:
    if not inner_name:
        return
    nested = extract_dir / inner_name
    if not nested.is_dir():
        return
    for item in nested.iterdir():
        dest = extract_dir / item.name
        if dest.exists():
            continue
        shutil.move(str(item), str(dest))
    shutil.rmtree(nested, ignore_errors=True)
    mac = extract_dir / "__MACOSX"
    if mac.exists():
        shutil.rmtree(mac, ignore_errors=True)


def extract_and_layout(out_dir: Path) -> None:
    """Normalize flat downloads into subQ-1-compatible directory layout."""
    guide_root = out_dir / "filtered_guide_bc_matrix.h5"
    guide_nested = out_dir / "perturbation" / "filtered_guide_bc_matrix.h5"
    if guide_root.exists() and not guide_nested.exists():
        guide_nested.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(guide_root), str(guide_nested))
        print(f"layout: {guide_nested}")

    for component, zip_name in [("segmentation", "segmentation.zip"), ("raw", "raw_output.zip")]:
        flat = out_dir / zip_name
        nested = out_dir / component / zip_name
        if flat.exists() and not nested.exists():
            nested.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(flat), str(nested))
        zip_path = nested if nested.exists() else flat
        extract_dir = out_dir / component / "extracted"
        marker = (
            extract_dir / "segmentation" / "filtered_feature_cell_matrix.h5"
            if component == "segmentation"
            else extract_dir / "tissue_positions.parquet"
        )
        if zip_path.exists() and not marker.exists():
            extract_dir.mkdir(parents=True, exist_ok=True)
            print(f"extracting {zip_path} -> {extract_dir}")
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(extract_dir)
            _flatten_extracted(extract_dir, "raw_output" if component == "raw" else "")
            print(f"extracted {component}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="subQ-1", help="Dataset name from SPAC portal")
    parser.add_argument(
        "--dataset-type",
        type=int,
        default=2,
        help="SPAC portal type: 1=lung metastasis, 2=subQ Visium HD, 3=timecourse Stereo-seq",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--components",
        nargs="+",
        default=["transcriptome", "perturbation", "segmentation", "raw"],
    )
    args = parser.parse_args()

    records = list_datasets(dataset_type=args.dataset_type)
    match = next((r for r in records if r["name"] == args.name), None)
    if match is None:
        raise SystemExit(f"Dataset {args.name!r} not found. Available: {[r['name'] for r in records]}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for key in args.components:
        share_id = match[key]
        download_share(share_id, args.out_dir)
    extract_and_layout(args.out_dir)


if __name__ == "__main__":
    main()
