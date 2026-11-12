"""Resolve and download a bounded multi-year MIDAS UHI pilot.

The public CEDA archive listings are used to resolve exact v202607 files. Data
downloads require a short-lived bearer token in ``CEDA_ACCESS_TOKEN``. Tokens
are read from the environment and are never written to logs or manifests.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import stat
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


VERSION = "202607"
DATASET_DOI = "10.5285/d04207b551674c07801b7d4e6d883e50"
ARCHIVE_ROOT = (
    "https://data.ceda.ac.uk/badc/ukmo-midas-open/data/"
    f"uk-hourly-weather-obs/dataset-version-{VERSION}"
)
DAP_ROOT = (
    "https://dap.ceda.ac.uk/badc/ukmo-midas-open/data/"
    f"uk-hourly-weather-obs/dataset-version-{VERSION}"
)

# Roles are hypotheses for the acquisition pilot, not final classifications.
STATIONS = [
    ("00697", "greater-london", "urban_candidate"),
    ("62084", "greater-london", "urban_candidate"),
    ("00723", "greater-london", "urban_candidate"),
    ("62073", "middlesex", "urban_candidate"),
    ("00709", "greater-london", "urban_candidate"),
    ("00719", "surrey", "rural_candidate"),
    ("30620", "surrey", "rural_candidate"),
    ("00487", "essex", "rural_candidate"),
    ("00744", "kent", "rural_candidate"),
    ("00458", "bedfordshire", "rural_candidate"),
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def get_json(url: str) -> dict:
    with urllib.request.urlopen(f"{url}?json", timeout=60) as response:
        return json.load(response)


def resolve_files(years: list[int]) -> list[dict]:
    records = []
    county_cache = {}
    for station_id, county, role in STATIONS:
        county_listing = county_cache.setdefault(county, get_json(f"{ARCHIVE_ROOT}/{county}"))
        station_items = [item for item in county_listing["items"]
                         if item["type"] == "dir" and item["name"].startswith(f"{station_id}_")]
        if len(station_items) != 1:
            raise ValueError(f"Expected one station directory for {station_id}: {station_items}")
        station_dir = station_items[0]["name"]
        qc_url = f"{ARCHIVE_ROOT}/{county}/{station_dir}/qc-version-1"
        qc_listing = get_json(qc_url)
        available = {item["name"]: item for item in qc_listing["items"] if item["type"] == "file"}
        for year in years:
            suffix = f"_qcv-1_{year}.csv"
            matches = [name for name in available if name.endswith(suffix)]
            if len(matches) > 1:
                raise ValueError(f"Multiple files for {station_id}, {year}: {matches}")
            filename = matches[0] if matches else ""
            relative = f"{county}/{station_dir}/qc-version-1/{filename}" if filename else ""
            records.append({
                "station_id": station_id,
                "station_directory": station_dir,
                "county": county,
                "provisional_role": role,
                "year": year,
                "available": bool(filename),
                "filename": filename,
                "download_url": f"{DAP_ROOT}/{relative}?download=1" if filename else "",
                "classification_status": "requires exposure and land-cover review",
            })
    return records


def validate_badcsv(path: Path, station_id: str, year: int) -> None:
    text = path.read_text(errors="replace")
    if "<html" in text[:1000].lower() or "\ndata\n" not in text:
        raise ValueError(f"Downloaded content is not BADC-CSV: {path}")
    if station_id not in path.name or f"_{year}.csv" not in path.name:
        raise ValueError(f"Unexpected station/year filename: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--years", nargs="+", type=int, default=list(range(2018, 2024)))
    parser.add_argument("--manifest-only", action="store_true")
    parser.add_argument("--resume", action="store_true",
                        help="Resume an existing manifest after CEDA authentication.")
    parser.add_argument("--token-file", type=Path,
                        help="Owner-readable local token file; never stored in outputs.")
    args = parser.parse_args()
    if args.output_dir.exists() and not args.resume:
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")

    records = resolve_files(args.years)
    args.output_dir.mkdir(parents=True, exist_ok=args.resume)
    manifest_csv = args.output_dir / "midas_uhi_acquisition_manifest.csv"
    if manifest_csv.exists():
        previous = list(csv.DictReader(manifest_csv.open(newline="")))
        keys = ("station_id", "year", "filename", "download_url")
        old_signature = [tuple(row[key] for key in keys) for row in previous]
        new_signature = [tuple(str(row[key]) for key in keys) for row in records]
        if old_signature != new_signature:
            raise ValueError("Existing acquisition manifest does not match this request")
    else:
        with manifest_csv.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)

    downloads = []
    if not args.manifest_only:
        token = None
        if args.token_file:
            mode = stat.S_IMODE(args.token_file.stat().st_mode)
            if mode & 0o077:
                raise PermissionError(
                    f"Token file permissions are too broad ({mode:o}); run chmod 600."
                )
            token = args.token_file.read_text().strip()
        if not token:
            token = os.environ.get("CEDA_ACCESS_TOKEN")
        if not token:
            raise RuntimeError(
                "Provide --token-file or set CEDA_ACCESS_TOKEN to a short-lived "
                "CEDA archive token; the token will not be saved in outputs."
            )
        for record in records:
            if not record["available"]:
                continue
            destination = args.output_dir / "hourly" / record["station_id"] / record["filename"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                validate_badcsv(destination, record["station_id"], record["year"])
                downloads.append({"path": str(destination), "sha256": sha256(destination),
                                  "size_bytes": destination.stat().st_size})
                continue
            request = urllib.request.Request(
                record["download_url"], headers={"Authorization": f"Bearer {token}"})
            try:
                print(f"Downloading {record['station_id']} {record['year']} ...", flush=True)
                with (
                    urllib.request.urlopen(request, timeout=120) as response,
                    destination.open("wb") as target,
                ):
                    while block := response.read(1024 * 1024):
                        target.write(block)
            except urllib.error.HTTPError as error:
                raise RuntimeError(
                    f"CEDA download failed for {record['station_id']} {record['year']}: "
                    f"HTTP {error.code}") from error
            validate_badcsv(destination, record["station_id"], record["year"])
            downloads.append({"path": str(destination), "sha256": sha256(destination),
                              "size_bytes": destination.stat().st_size})

    summary = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_version": VERSION,
        "dataset_doi": DATASET_DOI,
        "years": args.years,
        "stations": len(STATIONS),
        "expected_files": len(records),
        "available_files": sum(record["available"] for record in records),
        "downloaded_files": len(downloads),
        "manifest_only": args.manifest_only,
        "warning": "Station roles are provisional pending exposure and land-cover review",
        "downloads": downloads,
    }
    (args.output_dir / "acquisition_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary | {"downloads": f"{len(downloads)} file records"}, indent=2))


if __name__ == "__main__":
    main()
