#!/usr/bin/env python3
"""Compare revised 25 C valuations with the manuscript-era outputs."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy
import pygeoprocessing

try:
    from .summarize_ucm_valuations import _sum_field
except ImportError:  # Direct command-line execution.
    from summarize_ucm_valuations import _sum_field


# These are the preserved manuscript-era outputs documented in this project.
ARCHIVE_PATHS = {
    "baseline": (
        "scenario0/work_and_energy_runs/buildings_with_stats_london_scenario_25.0deg_5.0uhi_45.0hum_energy_productivity.shp",
        "scenario0/work_and_energy_runs/intermediate/wbgt_london_scenario_25.0deg_5.0uhi_45.0hum_energy_productivity.tif",
    ),
    "green10": (
        "scenario41/work_and_energy_runs/tcc_10prc/buildings_with_stats_london_scenario4_10prc_25deg_5uhi_45hum_energy_productivity.shp",
        "scenario41/work_and_energy_runs/tcc_10prc/intermediate/wbgt_london_scenario4_10prc_25deg_5uhi_45hum_energy_productivity.tif",
    ),
    "green20": (
        "scenario42/work_and_energy_runs/tcc_20prc/buildings_with_stats_london_scenario4_20prc_25.0deg_5.0uhi_45.0hum_energy_productivity.shp",
        "scenario42/work_and_energy_runs/tcc_20prc/intermediate/wbgt_london_scenario4_20prc_25.0deg_5.0uhi_45.0hum_energy_productivity.tif",
    ),
    "green30": (
        "scenario43/work_and_energy_runs/tcc_30prc/buildings_with_stats_london_scenario4_30prc_25.0deg_5.0uhi_45.0hum_energy_productivity.shp",
        "scenario43/work_and_energy_runs/tcc_30prc/intermediate/wbgt_london_scenario4_30prc_25.0deg_5.0uhi_45.0hum_energy_productivity.tif",
    ),
    "target10": (
        "scenario510/work_and_energy_runs/buildings_with_stats_london_scenario510_25deg_5uhi_45hum_energy_productivity.shp",
        "scenario510/work_and_energy_runs/intermediate/wbgt_london_scenario510_25deg_5uhi_45hum_energy_productivity.tif",
    ),
    "target20": (
        "scenario520/work_and_energy_runs/buildings_with_stats_london_scenario520_25deg_5uhi_45hum_energy_productivity.shp",
        "scenario520/work_and_energy_runs/intermediate/wbgt_london_scenario520_25deg_5uhi_45hum_energy_productivity.tif",
    ),
    "target30": (
        "scenario530/work_and_energy_runs/buildings_with_stats_london_scenario530_25deg_5uhi_45hum_energy_productivity.shp",
        "scenario530/work_and_energy_runs/intermediate/wbgt_london_scenario530_25deg_5uhi_45hum_energy_productivity.tif",
    ),
}


def _hothaps_mean(wbgt_path: Path) -> tuple[float, int]:
    """Calculate mean workability directly from an archived WBGT raster."""
    nodata = pygeoprocessing.get_raster_info(str(wbgt_path))["nodata"][0]
    total = 0.0
    count = 0
    for _, block in pygeoprocessing.iterblocks((str(wbgt_path), 1)):
        valid = numpy.isfinite(block)
        if nodata is not None:
            valid &= ~numpy.isclose(block, nodata)
        values = block[valid]
        workability = 0.1 + 0.9 / (1.0 + numpy.power(values / 30.94, 16.64))
        total += float(workability.sum(dtype=numpy.float64))
        count += values.size
    if not count:
        raise ValueError(f"WBGT raster has no valid pixels: {wbgt_path}")
    return total / count, count


def _paired_hothaps_gain(
    scenario_path: Path, baseline_path: Path
) -> tuple[float, int]:
    """Return mean scenario-minus-baseline workability on common valid cells."""
    scenario_info = pygeoprocessing.get_raster_info(str(scenario_path))
    baseline_info = pygeoprocessing.get_raster_info(str(baseline_path))
    if scenario_info["raster_size"] != baseline_info["raster_size"]:
        raise ValueError("Paired WBGT rasters have different dimensions")
    scenario_nodata = scenario_info["nodata"][0]
    baseline_nodata = baseline_info["nodata"][0]
    total = 0.0
    count = 0
    scenario_blocks = pygeoprocessing.iterblocks((str(scenario_path), 1))
    baseline_blocks = pygeoprocessing.iterblocks((str(baseline_path), 1))
    for (scenario_offset, scenario), (baseline_offset, baseline) in zip(
        scenario_blocks, baseline_blocks, strict=True
    ):
        if scenario_offset != baseline_offset:
            raise ValueError("Paired WBGT rasters have different block grids")
        valid = numpy.isfinite(scenario) & numpy.isfinite(baseline)
        if scenario_nodata is not None:
            valid &= ~numpy.isclose(scenario, scenario_nodata)
        if baseline_nodata is not None:
            valid &= ~numpy.isclose(baseline, baseline_nodata)
        scenario_work = 0.1 + 0.9 / (
            1.0 + numpy.power(scenario[valid] / 30.94, 16.64)
        )
        baseline_work = 0.1 + 0.9 / (
            1.0 + numpy.power(baseline[valid] / 30.94, 16.64)
        )
        total += float((scenario_work - baseline_work).sum(dtype=numpy.float64))
        count += int(valid.sum())
    if not count:
        raise ValueError("Paired WBGT rasters have no common valid pixels")
    return total / count * 100, count


def _only_match(folder: Path, pattern: str) -> Path:
    matches = list(folder.glob(pattern))
    if len(matches) != 1:
        raise ValueError(f"Expected one {pattern} under {folder}; found {len(matches)}")
    return matches[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("original_ucm_root", type=Path)
    parser.add_argument("revised_summary_csv", type=Path)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument(
        "--reuse-energy-from-output",
        action="store_true",
        help="Reuse archived energy totals from an existing comparison CSV.",
    )
    args = parser.parse_args()

    original_root = args.original_ucm_root.expanduser().resolve()
    revised_path = args.revised_summary_csv.expanduser().resolve()
    output_path = (
        args.output_csv.expanduser().resolve()
        if args.output_csv
        else revised_path.parent / "revised_vs_original_energy_productivity.csv"
    )

    with revised_path.open(newline="", encoding="utf-8-sig") as table:
        revised = {row["scenario"]: row for row in csv.DictReader(table)}
    energy_cache = {}
    if args.reuse_energy_from_output and output_path.is_file():
        with output_path.open(newline="", encoding="utf-8-sig") as table:
            energy_cache = {row["scenario"]: row for row in csv.DictReader(table)}

    old_baseline_wbgt = original_root / ARCHIVE_PATHS["baseline"][1]
    revised_root = revised_path.parent.parent
    revised_baseline_wbgt = _only_match(
        revised_root / "baseline" / "intermediate", "wbgt_*.tif"
    )

    rows = []
    source_paths = []
    for scenario, (energy_relative, work_relative) in ARCHIVE_PATHS.items():
        energy_path = original_root / energy_relative
        work_path = original_root / work_relative
        for path in (energy_path, work_path):
            if not path.is_file():
                raise FileNotFoundError(path)
            source_paths.append(str(path))
        if scenario in energy_cache:
            cached = energy_cache[scenario]
            old_energy = float(cached["original_energy_savings"])
            old_valid = int(cached["original_buildings_with_energy"])
            old_missing = int(cached["original_buildings_missing_energy"])
        else:
            old_energy, old_valid, old_missing = _sum_field(energy_path, "energy_sav")
        # Recalculate from WBGT because one archived derived TIFF is corrupt.
        old_workability, old_pixels = _hothaps_mean(work_path)
        old_gain_pp, old_common_pixels = _paired_hothaps_gain(
            work_path, old_baseline_wbgt
        )
        revised_wbgt = _only_match(
            revised_root / scenario / "intermediate", "wbgt_*.tif"
        )
        new_gain_pp, new_common_pixels = _paired_hothaps_gain(
            revised_wbgt, revised_baseline_wbgt
        )
        new = revised[scenario]
        rows.append({
            "scenario": scenario,
            "original_energy_savings": old_energy,
            "revised_energy_savings": float(new["energy_savings_unique_buildings"]),
            "original_buildings_with_energy": old_valid,
            "original_buildings_missing_energy": old_missing,
            "original_workability_fraction": old_workability,
            "revised_workability_fraction": float(new["workability_fraction"]),
            "original_valid_productivity_pixels": old_pixels,
            "original_productivity_gain_vs_baseline_pp": old_gain_pp,
            "original_common_productivity_pixels": old_common_pixels,
            "revised_productivity_gain_vs_baseline_pp": new_gain_pp,
            "revised_common_productivity_pixels": new_common_pixels,
        })

    old_base = rows[0]["original_energy_savings"]
    new_base = rows[0]["revised_energy_savings"]
    for row in rows:
        old_gain = row["original_energy_savings"] - old_base
        new_gain = row["revised_energy_savings"] - new_base
        row["original_energy_gain_vs_baseline"] = old_gain
        row["revised_energy_gain_vs_baseline"] = new_gain
        row["energy_gain_revision_difference"] = new_gain - old_gain
        row["energy_gain_revision_difference_pct"] = (
            (new_gain / old_gain - 1) * 100 if old_gain else None
        )
        row["productivity_gain_revision_difference_pp"] = (
            row["revised_productivity_gain_vs_baseline_pp"]
            - row["original_productivity_gain_vs_baseline_pp"]
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as table:
        writer = csv.DictWriter(table, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "comparison": "25 C, 5 C UHI, 45% humidity; energy gains use each run's baseline and productivity gains use paired common valid cells",
        "original_sources": source_paths,
        "revised_summary": str(revised_path),
        "output": str(output_path),
    }
    output_path.with_suffix(".json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Created {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
