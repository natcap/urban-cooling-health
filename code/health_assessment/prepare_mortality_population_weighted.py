#!/usr/bin/env python3
"""Allocate borough mortality counts to pixels using 2021 population.

This is the revised alternative to the legacy area-weighted rasterization in
``health-model-01-prep-input-ONS-mortality-data.Rmd``. It preserves each
borough's observed annual death count while placing deaths in proportion to the
2021 population raster.

The mortality input must be the long CSV created by
``prepare_nomis_mortality_long.py``, with these columns:

    borough_id,year,cause,deaths

Example usage is documented in ``code/health_assessment/README.md``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import subprocess
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


NOMIS_SOURCE_URL = "https://www.nomisweb.co.uk/datasets/mortsa"
REGISTRATION_DEFINITION = (
    "Figures represent the number of deaths registered in the calendar year."
)


def allocate_deaths_by_population(
    population: np.ndarray,
    zone_index: np.ndarray,
    deaths_by_zone: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Allocate zone totals to pixels in proportion to pixel population.

    ``zone_index`` uses 0 for pixels outside all zones and positive consecutive
    integers for zones. ``deaths_by_zone`` must use the same integer indexing,
    with element 0 reserved for the outside-zone value.
    """
    if population.shape != zone_index.shape:
        raise ValueError("population and zone_index must have identical shapes")
    if zone_index.ndim != 2:
        raise ValueError("population and zone_index must be two-dimensional")
    if np.any(np.isfinite(population) & (population < 0)):
        raise ValueError("population contains negative values")
    if np.any(zone_index < 0):
        raise ValueError("zone_index contains negative values")

    max_zone = int(zone_index.max(initial=0))
    if len(deaths_by_zone) != max_zone + 1:
        raise ValueError("deaths_by_zone does not match the zone index range")

    valid = (zone_index > 0) & np.isfinite(population) & (population > 0)
    population_totals = np.bincount(
        zone_index[valid], weights=population[valid], minlength=max_zone + 1
    ).astype(np.float64)

    missing_population = np.flatnonzero(population_totals[1:] <= 0) + 1
    if missing_population.size:
        raise ValueError(
            "No positive 2021 population was found for zone index(es): "
            + ", ".join(map(str, missing_population))
        )
    if np.any(~np.isfinite(deaths_by_zone[1:])):
        raise ValueError("one or more zones have missing death counts")
    if np.any(deaths_by_zone[1:] < 0):
        raise ValueError("death counts must be non-negative")

    rates = np.zeros(max_zone + 1, dtype=np.float64)
    rates[1:] = deaths_by_zone[1:] / population_totals[1:]

    allocated = np.full(population.shape, np.nan, dtype=np.float64)
    allocated[valid] = population[valid] * rates[zone_index[valid]]
    allocated_totals = np.bincount(
        zone_index[valid], weights=allocated[valid], minlength=max_zone + 1
    ).astype(np.float64)
    return allocated, population_totals, allocated_totals


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for block in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _dataset_checksums(path: Path) -> dict[str, str]:
    """Hash a single-file dataset or every sidecar in an ESRI Shapefile."""
    paths = sorted(path.parent.glob(f"{path.stem}.*")) if path.suffix.lower() == ".shp" else [path]
    return {str(candidate): _sha256(candidate) for candidate in paths if candidate.is_file()}


def _safe_name(value: str) -> str:
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip()).strip("_").lower()
    if not name:
        raise ValueError(f"Cannot derive an output name from cause {value!r}")
    return name


def _git_commit(repository_root: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def _git_dirty(repository_root: Path) -> bool | None:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        )
        return bool(result.stdout.strip())
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def _require_columns(frame: pd.DataFrame, required: Iterable[str], source: Path) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{source} is missing required columns: {', '.join(missing)}")


def prepare(args: argparse.Namespace) -> None:
    # Heavy geospatial imports are local so the allocation function can be
    # unit-tested without loading the full GIS stack.
    import geopandas as gpd
    import rasterio
    from rasterio.features import rasterize

    borough_path = Path(args.boroughs).resolve()
    mortality_path = Path(args.mortality_csv).resolve()
    population_path = Path(args.population_2021).resolve()
    output_dir = Path(args.output_dir).resolve()

    for path in (borough_path, mortality_path, population_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required input does not exist: {path}")

    mortality = pd.read_csv(mortality_path, dtype={args.mortality_id_field: str})
    required = (args.mortality_id_field, "year", "cause", "deaths")
    _require_columns(mortality, required, mortality_path)
    mortality = mortality.loc[mortality["year"] == args.year, list(required)].copy()
    if mortality.empty:
        raise ValueError(f"No mortality rows were found for calendar year {args.year}")
    mortality[args.mortality_id_field] = (
        mortality[args.mortality_id_field].astype(str).str.strip()
    )
    mortality["cause"] = mortality["cause"].astype(str).str.strip()
    mortality["deaths"] = pd.to_numeric(mortality["deaths"], errors="raise")
    if mortality.duplicated([args.mortality_id_field, "cause"]).any():
        raise ValueError("Mortality CSV has duplicate borough_id/cause rows")
    if (mortality["deaths"] < 0).any() or mortality["deaths"].isna().any():
        raise ValueError("Mortality death counts must be complete and non-negative")

    boroughs = gpd.read_file(borough_path)
    if args.borough_field not in boroughs.columns:
        raise ValueError(f"Borough layer has no {args.borough_field!r} column")
    if boroughs.crs is None:
        raise ValueError("Borough layer has no CRS; assign it explicitly upstream")
    if boroughs.geometry.isna().any() or boroughs.geometry.is_empty.any():
        raise ValueError("Borough layer contains missing or empty geometries")
    if (~boroughs.geometry.is_valid).any():
        raise ValueError("Borough layer contains invalid geometries")
    boroughs[args.borough_field] = boroughs[args.borough_field].astype(str).str.strip()
    if boroughs[args.borough_field].duplicated().any():
        raise ValueError("Borough layer contains duplicate identifiers")

    with rasterio.open(population_path) as population_source:
        if population_source.crs is None:
            raise ValueError("2021 population raster has no CRS")
        population = (
            population_source.read(1, masked=True).filled(np.nan).astype(np.float64)
        )
        population_profile = population_source.profile.copy()
        population_transform = population_source.transform
        population_crs = population_source.crs

    boroughs = boroughs.to_crs(population_crs)
    borough_ids = boroughs[args.borough_field].tolist()
    zone_lookup = {borough_id: index for index, borough_id in enumerate(borough_ids, 1)}
    zone_index = rasterize(
        ((geometry, zone_lookup[borough_id]) for geometry, borough_id in zip(
            boroughs.geometry, borough_ids
        )),
        out_shape=population.shape,
        transform=population_transform,
        fill=0,
        all_touched=False,
        dtype="int32",
    )

    positive_population = np.isfinite(population) & (population > 0)
    total_positive_population = float(population[positive_population].sum())
    unassigned_population = float(population[positive_population & (zone_index == 0)].sum())
    unassigned_share = (
        unassigned_population / total_positive_population
        if total_positive_population > 0
        else 0.0
    )
    if unassigned_share > args.max_unassigned_population_share:
        raise ValueError(
            f"{unassigned_share:.3%} of positive population is outside borough polygons; "
            f"maximum allowed is {args.max_unassigned_population_share:.3%}"
        )

    missing_ids = sorted(set(borough_ids) - set(mortality[args.mortality_id_field]))
    extra_ids = sorted(set(mortality[args.mortality_id_field]) - set(borough_ids))
    if missing_ids or extra_ids:
        raise ValueError(
            "Borough identifiers do not match. "
            f"Missing mortality IDs={missing_ids}; extra mortality IDs={extra_ids}"
        )

    causes = sorted(mortality["cause"].unique())
    expected_ids = set(borough_ids)
    for cause in causes:
        cause_ids = set(
            mortality.loc[mortality["cause"] == cause, args.mortality_id_field]
        )
        if cause_ids != expected_ids:
            raise ValueError(
                f"Cause {cause!r} does not have exactly one row for every borough; "
                f"missing={sorted(expected_ids - cause_ids)}, "
                f"extra={sorted(cause_ids - expected_ids)}"
            )
    planned_outputs = {
        cause: output_dir
        / f"baseline_deaths_{_safe_name(cause)}_population_weighted_{args.year}.tif"
        for cause in causes
    }
    if len({path.name for path in planned_outputs.values()}) != len(planned_outputs):
        raise ValueError("Two cause names resolve to the same safe output filename")
    other_outputs = [output_dir / "allocation_qa.csv", output_dir / "manifest.json"]
    conflicts = [path for path in [*planned_outputs.values(), *other_outputs] if path.exists()]
    if conflicts and not args.force:
        raise FileExistsError(
            "Refusing to overwrite existing outputs; use --force after reviewing them: "
            + ", ".join(map(str, conflicts))
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    qa_frames: list[pd.DataFrame] = []
    output_records: dict[str, dict[str, object]] = {}

    for cause in causes:
        cause_rows = mortality.loc[mortality["cause"] == cause].set_index(
            args.mortality_id_field
        )
        deaths_by_zone = np.full(len(borough_ids) + 1, np.nan, dtype=np.float64)
        deaths_by_zone[0] = 0.0
        for borough_id, zone in zone_lookup.items():
            deaths_by_zone[zone] = float(cause_rows.at[borough_id, "deaths"])

        allocated, population_totals, allocated_totals = allocate_deaths_by_population(
            population, zone_index, deaths_by_zone
        )
        difference = allocated_totals - deaths_by_zone
        relative_error = np.divide(
            np.abs(difference),
            np.maximum(np.abs(deaths_by_zone), 1.0),
            out=np.zeros_like(difference),
            where=np.isfinite(deaths_by_zone),
        )
        if np.nanmax(relative_error[1:]) > args.conservation_tolerance:
            raise RuntimeError(
                f"Population allocation failed conservation for {cause}; maximum "
                f"relative error={np.nanmax(relative_error[1:]):.3g}"
            )

        output_path = planned_outputs[cause]
        profile = population_profile.copy()
        profile.update(driver="GTiff", dtype="float32", count=1, compress="lzw", nodata=-9999.0)
        raster_values = np.where(np.isfinite(allocated), allocated, -9999.0).astype(
            np.float32
        )
        with rasterio.open(output_path, "w", **profile) as target:
            target.write(raster_values, 1)

        qa_frames.append(
            pd.DataFrame(
                {
                    "borough_id": borough_ids,
                    "year": args.year,
                    "cause": cause,
                    "population_2021": population_totals[1:],
                    "observed_registered_deaths": deaths_by_zone[1:],
                    "allocated_deaths": allocated_totals[1:],
                    "difference": difference[1:],
                }
            )
        )
        output_records[cause] = {
            "path": str(output_path),
            "sha256": _sha256(output_path),
            "observed_total": float(deaths_by_zone[1:].sum()),
            "allocated_total": float(allocated_totals[1:].sum()),
        }

    qa = pd.concat(qa_frames, ignore_index=True)
    qa_path = output_dir / "allocation_qa.csv"
    qa.to_csv(qa_path, index=False)

    repository_root = Path(__file__).resolve().parents[2]
    manifest = {
        "schema_version": 1,
        "method": "borough annual registered deaths allocated by 2021 pixel population",
        "formula": "pixel_deaths = borough_deaths * pixel_population_2021 / borough_population_2021",
        "mortality_year": args.year,
        "population_year": 2021,
        "mortality_source": args.source_url,
        "mortality_definition": REGISTRATION_DEFINITION,
        "geography": {
            "borough_field": args.borough_field,
            "mortality_id_field": args.mortality_id_field,
            "borough_count": len(borough_ids),
        },
        "quality_control": {
            "allocation_conservation_tolerance": args.conservation_tolerance,
            "unassigned_population": unassigned_population,
            "unassigned_population_share": unassigned_share,
            "maximum_unassigned_population_share": args.max_unassigned_population_share,
        },
        "inputs": {
            "boroughs": {
                "path": str(borough_path),
                "checksums": _dataset_checksums(borough_path),
            },
            "mortality_csv": {
                "path": str(mortality_path),
                "sha256": _sha256(mortality_path),
            },
            "population_2021": {
                "path": str(population_path),
                "sha256": _sha256(population_path),
            },
        },
        "outputs": output_records,
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "geopandas": gpd.__version__,
            "rasterio": rasterio.__version__,
            "git_commit": _git_commit(repository_root),
            "git_worktree_dirty": _git_dirty(repository_root),
            "preparation_script_sha256": _sha256(Path(__file__).resolve()),
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"Created {len(planned_outputs)} population-weighted mortality rasters")
    print(f"Quality-control table: {qa_path}")
    print(f"Run manifest: {manifest_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--boroughs", required=True, help="Borough polygon dataset; GeoPackage recommended"
    )
    parser.add_argument("--borough-field", default="GSS_CODE", help="ID field in borough polygons")
    parser.add_argument(
        "--mortality-csv",
        required=True,
        help="Long mortality count CSV created by prepare_nomis_mortality_long.py",
    )
    parser.add_argument(
        "--mortality-id-field", default="borough_id", help="Borough ID column in mortality CSV"
    )
    parser.add_argument("--population-2021", required=True, help="2021 population raster")
    parser.add_argument("--year", type=int, default=2021, help="Calendar registration year")
    parser.add_argument("--output-dir", required=True, help="New output directory")
    parser.add_argument("--source-url", default=NOMIS_SOURCE_URL, help="Mortality source URL")
    parser.add_argument(
        "--conservation-tolerance",
        type=float,
        default=1e-8,
        help="Maximum relative allocation error per borough",
    )
    parser.add_argument(
        "--max-unassigned-population-share",
        type=float,
        default=0.001,
        help="Maximum share of positive raster population outside borough polygons",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite reviewed existing outputs")
    return parser


if __name__ == "__main__":
    prepare(build_parser().parse_args())
