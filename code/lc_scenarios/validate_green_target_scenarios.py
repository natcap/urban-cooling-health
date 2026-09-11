"""Independently validate paired Green and Target LULC scenario rasters."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio


LABELS = ("10", "20", "30")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _same_grid(reference: rasterio.DatasetReader, other: rasterio.DatasetReader) -> bool:
    return (
        reference.shape == other.shape
        and reference.crs == other.crs
        and np.allclose(tuple(reference.transform), tuple(other.transform))
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--green", type=Path, nargs=3, metavar=("G10", "G20", "G30"), required=True)
    parser.add_argument("--target", type=Path, nargs=3, metavar=("T10", "T20", "T30"), required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--canopy-codes", type=int, nargs="+", default=[1, 2, 100])
    parser.add_argument("--source-codes", type=int, nargs="+", default=[4, 20, 21])
    parser.add_argument("--replacement-code", type=int, default=100)
    args = parser.parse_args()

    baseline_path = args.baseline.expanduser().resolve()
    green_paths = [path.expanduser().resolve() for path in args.green]
    target_paths = [path.expanduser().resolve() for path in args.target]
    for path in [baseline_path, *green_paths, *target_paths]:
        if not path.is_file():
            parser.error(f"Raster not found: {path}")
    canopy_codes = set(args.canopy_codes)
    source_codes = set(args.source_codes)
    if args.replacement_code not in canopy_codes:
        parser.error("Replacement code must be included in canopy codes")

    with rasterio.open(baseline_path) as baseline:
        base = baseline.read(1)
        base_valid = baseline.read_masks(1) > 0
        baseline_nodata = baseline.nodata
        pixel_area_m2 = abs(
            baseline.transform.a * baseline.transform.e
            - baseline.transform.b * baseline.transform.d
        )
        records = []
        addition_masks: dict[str, np.ndarray] = {}
        failed_checks: list[str] = []
        for family, paths in (("Green", green_paths), ("Target", target_paths)):
            for level, path in zip(LABELS, paths, strict=True):
                name = f"{family}{level}"
                with rasterio.open(path) as scenario:
                    if not _same_grid(baseline, scenario):
                        raise ValueError(f"Grid mismatch: {path}")
                    values = scenario.read(1)
                    valid = scenario.read_masks(1) > 0
                    nodata_match = bool(
                        baseline_nodata == scenario.nodata
                        and np.array_equal(base_valid, valid)
                    )
                changed = base_valid & (base != values)
                eligible_addition = (
                    changed
                    & np.isin(base, sorted(source_codes))
                    & (values == args.replacement_code)
                )
                ineligible_change = changed & ~eligible_addition
                lost_canopy = (
                    base_valid
                    & np.isin(base, sorted(canopy_codes))
                    & ~np.isin(values, sorted(canopy_codes))
                )
                transitions = Counter(
                    zip(map(int, base[changed]), map(int, values[changed]))
                )
                addition_masks[name] = eligible_addition
                record = {
                    "scenario": name,
                    "raster": str(path),
                    "sha256": _sha256(path),
                    "added_pixels": int(eligible_addition.sum()),
                    "added_area_km2": eligible_addition.sum()
                    * pixel_area_m2
                    / 1_000_000,
                    "all_changed_pixels": int(changed.sum()),
                    "ineligible_changed_pixels": int(ineligible_change.sum()),
                    "lost_canopy_pixels": int(lost_canopy.sum()),
                    "nodata_match": nodata_match,
                    "transitions": {
                        f"{source}->{target}": count
                        for (source, target), count in sorted(transitions.items())
                    },
                }
                records.append(record)
                if not nodata_match:
                    failed_checks.append(f"{name}: NoData mismatch")
                if record["ineligible_changed_pixels"]:
                    failed_checks.append(f"{name}: ineligible transitions")
                if record["lost_canopy_pixels"]:
                    failed_checks.append(f"{name}: canopy removed")

    pair_records = []
    for level in LABELS:
        green_mask = addition_masks[f"Green{level}"]
        target_mask = addition_masks[f"Target{level}"]
        green_count, target_count = int(green_mask.sum()), int(target_mask.sum())
        overlap = int(np.count_nonzero(green_mask & target_mask))
        union = int(np.count_nonzero(green_mask | target_mask))
        pair = {
            "level": level,
            "green_added_pixels": green_count,
            "target_added_pixels": target_count,
            "target_minus_green_pixels": target_count - green_count,
            "overlap_pixels": overlap,
            "overlap_pct_of_each_equal_budget": 100 * overlap / green_count,
            "jaccard_pct": 100 * overlap / union,
        }
        pair_records.append(pair)
        if target_count != green_count:
            failed_checks.append(f"Level {level}: unequal realized canopy")

    nesting = {}
    for family in ("Green", "Target"):
        first = bool(
            np.all(
                ~addition_masks[f"{family}10"] | addition_masks[f"{family}20"]
            )
        )
        second = bool(
            np.all(
                ~addition_masks[f"{family}20"] | addition_masks[f"{family}30"]
            )
        )
        nesting[family] = {"10_within_20": first, "20_within_30": second}
        if not first or not second:
            failed_checks.append(f"{family}: nesting failed")

    output_json = args.output_json.expanduser().resolve()
    output_csv = args.output_csv.expanduser().resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if output_json.exists() or output_csv.exists():
        raise FileExistsError("Refusing to overwrite an existing QA artifact")
    with output_csv.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(pair_records[0]))
        writer.writeheader()
        writer.writerows(pair_records)
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "baseline": str(baseline_path),
        "baseline_sha256": _sha256(baseline_path),
        "canopy_codes": sorted(canopy_codes),
        "source_codes": sorted(source_codes),
        "replacement_code": args.replacement_code,
        "pixel_area_m2": pixel_area_m2,
        "scenarios": records,
        "paired_comparison": pair_records,
        "nesting": nesting,
        "passed": not failed_checks,
        "failed_checks": failed_checks,
        "paired_summary_csv": str(output_csv),
    }
    output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Saved paired QA: {output_json}")
    print("PASS" if report["passed"] else f"FAIL: {failed_checks}")
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
