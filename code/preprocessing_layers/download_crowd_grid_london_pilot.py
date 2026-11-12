"""Download a bounded Crowd-Grid / Recent Heat Pack UHI sensitivity set.

The script downloads daily maximum/minimum temperature CSVs for London's 33
local authorities and three rural-reference local authorities, plus the
official 2021 MSOA boundary files. A short-lived CEDA bearer token is read from
an owner-only file or ``CEDA_ACCESS_TOKEN`` and is never written to outputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


DATASET_DOI = "10.5285/9f45ae9217c44dea82be3ba4ef0fa30f"
CATALOGUE_URL = "https://catalogue.ceda.ac.uk/uuid/9f45ae9217c44dea82be3ba4ef0fa30f"
DATA_ROOT = "https://data.ceda.ac.uk/badc/crowd-grid/data/recent-heat-packs/v1.0"
DAP_ROOT = "https://dap.ceda.ac.uk/badc/crowd-grid/data/recent-heat-packs/v1.0"

LONDON_SLUGS = [
    "barking-and-dagenham", "barnet", "bexley", "brent", "bromley",
    "camden", "city-of-london", "croydon", "ealing", "enfield",
    "greenwich", "hackney", "hammersmith-and-fulham", "haringey", "harrow",
    "havering", "hillingdon", "hounslow", "islington",
    "kensington-and-chelsea", "kingston-upon-thames", "lambeth", "lewisham",
    "merton", "newham", "redbridge", "richmond-upon-thames", "southwark",
    "sutton", "tower-hamlets", "waltham-forest", "wandsworth", "westminster",
]

# These contain the three MIDAS sites retained as rural-background candidates:
# Wisley, Charlwood and Woburn. Exact station MSOAs are resolved spatially later.
# The supplied 2021 geography places Charlwood station in Crawley 001.
RURAL_REFERENCE_SLUGS = ["guildford", "crawley", "central-bedfordshire"]

SHAPE_COMPONENTS = [
    "00README-uk-msoa-iz-dez-boundaries.md",
    "uk-msoa-iz-dez-boundaries.cpg",
    "uk-msoa-iz-dez-boundaries.dbf",
    "uk-msoa-iz-dez-boundaries.prj",
    "uk-msoa-iz-dez-boundaries.shp",
    "uk-msoa-iz-dez-boundaries.shx",
]


def get_json(url: str) -> dict:
    with urllib.request.urlopen(f"{url}?json", timeout=120) as response:
        return json.load(response)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_token(token_file: Path | None) -> str:
    token = ""
    if token_file:
        mode = stat.S_IMODE(token_file.stat().st_mode)
        if mode & 0o077:
            raise PermissionError(f"Token file permissions are too broad ({mode:o}); use 600")
        token = token_file.read_text().strip()
    token = token or os.environ.get("CEDA_ACCESS_TOKEN", "")
    if not token:
        raise RuntimeError("Provide --token-file or set CEDA_ACCESS_TOKEN")
    return token


def download(url: str, destination: Path, token: str, expected_md5: str) -> dict:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        request = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
        with urllib.request.urlopen(request, timeout=240) as response, destination.open("wb") as target:
            while block := response.read(1024 * 1024):
                target.write(block)
    md5 = hashlib.md5(destination.read_bytes()).hexdigest()  # noqa: S324 - archive checksum
    if expected_md5 and md5 != expected_md5:
        raise ValueError(f"MD5 mismatch for {destination.name}: {md5} != {expected_md5}")
    return {
        "path": str(destination),
        "size_bytes": destination.stat().st_size,
        "md5": md5,
        "sha256": sha256(destination),
        "source_url": url,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--token-file", type=Path)
    args = parser.parse_args()

    token = read_token(args.token_file)
    slugs = LONDON_SLUGS + RURAL_REFERENCE_SLUGS
    daily_listing = get_json(f"{DATA_ROOT}/daily-csvs")
    daily_items = {item["name"]: item for item in daily_listing["items"]}
    shape_listing = get_json(
        f"{DATA_ROOT}/shapefiles/uk-msoa-iz-dez-boundaries"
    )
    shape_items = {item["name"]: item for item in shape_listing["items"]}

    records = []
    for slug in slugs:
        for long_name in ("maximum", "minimum"):
            filename = f"{slug}_recent-heat-pack_daily-{long_name}-temperature.csv"
            if filename not in daily_items:
                raise FileNotFoundError(f"CEDA listing does not contain {filename}")
            item = daily_items[filename]
            destination = args.output_dir / "daily_csvs" / filename
            records.append(download(item["download"], destination, token, item.get("md5", "")))

    for filename in SHAPE_COMPONENTS:
        item = shape_items[filename]
        destination = args.output_dir / "boundaries" / filename
        records.append(download(item["download"], destination, token, item.get("md5", "")))

    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "Recent Heat Packs for local areas in the UK (v1.0, prototype)",
        "dataset_doi": DATASET_DOI,
        "catalogue_url": CATALOGUE_URL,
        "london_local_authorities": LONDON_SLUGS,
        "rural_reference_local_authorities": RURAL_REFERENCE_SLUGS,
        "warning": (
            "Prototype Crowd-Grid-derived spatial averages; exact station MSOAs and "
            "interpretation are resolved in the analysis script"
        ),
        "downloads": records,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "crowd_grid_acquisition_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps({"files": len(records), "bytes": sum(r["size_bytes"] for r in records)}, indent=2))


if __name__ == "__main__":
    main()
