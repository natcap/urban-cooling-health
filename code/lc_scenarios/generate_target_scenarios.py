"""Generate deterministic, equal-canopy Target10/20/30 LULC rasters.

Candidates must be the final screened Target point shapefile. The generator
explicitly sorts every record by increasing ``rank`` and then increasing source
``FID``. A candidate adds the one 10 m pixel whose centre lies strictly within
its assumed 5 m crown. Only baseline cells in the configured source-code set
may become code 100.

The historical Target rasters are never read or modified. Outputs are nested,
have exact canopy budgets, and include a manifest, transition table and map.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyogrio
import rasterio
from matplotlib.colors import ListedColormap

from build_tree_opportunity_mask import point_coordinate_views


CANOPY_CODES = {1, 2, 100}
DEFAULT_SOURCE_CODES = {4, 20, 21}
REPLACEMENT_CODE = 100
DEFAULT_BUDGETS = (307_768, 595_084, 894_249)
SCENARIO_LABELS = ("Target10", "Target20", "Target30")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _dbf_numeric_views(
    path: Path, requested_fields: tuple[str, ...]
) -> tuple[int, dict[str, np.ndarray]]:
    """Return strided byte views of selected numeric DBF fields."""
    with path.open("rb") as source:
        header = source.read(32)
        if len(header) != 32:
            raise ValueError(f"Invalid DBF header: {path}")
        record_count = int.from_bytes(header[4:8], "little")
        header_length = int.from_bytes(header[8:10], "little")
        record_length = int.from_bytes(header[10:12], "little")
        offset = 1  # Every DBF record begins with its deletion flag.
        fields: dict[str, tuple[int, int, str]] = {}
        while True:
            descriptor = source.read(32)
            if not descriptor or descriptor[0] == 13:
                break
            name = descriptor[:11].split(b"\0", 1)[0].decode("ascii")
            width = descriptor[16]
            fields[name] = (offset, width, chr(descriptor[11]))
            offset += width

    if offset != record_length:
        raise ValueError("DBF field widths do not match the declared record length")
    missing = set(requested_fields) - set(fields)
    if missing:
        raise ValueError(f"DBF is missing required fields: {sorted(missing)}")
    raw = np.memmap(path, mode="r", dtype="u1")
    views: dict[str, np.ndarray] = {}
    for name in requested_fields:
        field_offset, width, field_type = fields[name]
        if field_type not in {"N", "F"}:
            raise ValueError(f"DBF field {name} is not numeric")
        views[name] = np.ndarray(
            (record_count,),
            dtype=f"S{width}",
            buffer=raw,
            offset=header_length + field_offset,
            strides=(record_length,),
        )
    return record_count, views


def _integer_chunk(view: np.ndarray, start: int, stop: int, name: str) -> np.ndarray:
    """Parse a DBF numeric field chunk as int64 with a clear error."""
    try:
        return np.char.strip(view[start:stop]).astype(np.int64)
    except ValueError as error:
        raise ValueError(f"Field {name} contains a blank or non-integer value") from error


def _git_record(repo_root: Path | None) -> dict[str, object] | None:
    if repo_root is None:
        return None
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        return {"root": str(repo_root), "commit": commit, "dirty": bool(status)}
    except (OSError, subprocess.CalledProcessError):
        return {"root": str(repo_root), "commit": None, "dirty": None}


def _write_preview(
    output_path: Path,
    valid: np.ndarray,
    accepted_cells: np.ndarray,
    budgets: tuple[int, int, int],
    transform: rasterio.Affine,
) -> None:
    """Write a compact three-panel map of nested Target additions."""
    extent = (
        transform.c,
        transform.c + valid.shape[1] * transform.a,
        transform.f + valid.shape[0] * transform.e,
        transform.f,
    )
    background = np.where(valid, 1, np.nan)
    change_cmap = ListedColormap([(0, 0, 0, 0), "#16843d"])
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    for axis, label, budget in zip(axes, SCENARIO_LABELS, budgets, strict=True):
        additions = np.zeros(valid.size, dtype=np.uint8)
        additions[accepted_cells[:budget]] = 1
        axis.imshow(
            background,
            extent=extent,
            origin="upper",
            cmap=ListedColormap(["#ededed"]),
            interpolation="nearest",
        )
        axis.imshow(
            additions.reshape(valid.shape),
            extent=extent,
            origin="upper",
            cmap=change_cmap,
            interpolation="nearest",
        )
        axis.set_title(f"{label}: {budget:,} added cells")
        axis.set_axis_off()
    fig.suptitle("Revised vulnerability-targeted canopy additions")
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate-points", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--budgets",
        type=int,
        nargs=3,
        metavar=SCENARIO_LABELS,
        default=DEFAULT_BUDGETS,
    )
    parser.add_argument(
        "--source-codes",
        type=int,
        nargs="+",
        default=sorted(DEFAULT_SOURCE_CODES),
    )
    parser.add_argument("--replacement-code", type=int, default=REPLACEMENT_CODE)
    parser.add_argument("--radius-m", type=float, default=5.0)
    parser.add_argument("--chunk-size", type=int, default=250_000)
    parser.add_argument("--repo-root", type=Path)
    args = parser.parse_args()

    baseline_path = args.baseline.expanduser().resolve()
    points_path = args.candidate_points.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    repo_root = args.repo_root.expanduser().resolve() if args.repo_root else None
    budgets = tuple(args.budgets)
    source_codes = set(args.source_codes)
    for path in (baseline_path, points_path):
        if not path.is_file():
            parser.error(f"Input not found: {path}")
    if points_path.suffix.lower() != ".shp":
        parser.error("--candidate-points must be the final Point shapefile")
    dbf_path = points_path.with_suffix(".dbf")
    if not dbf_path.is_file():
        parser.error(f"Candidate DBF not found: {dbf_path}")
    if not (0 < budgets[0] < budgets[1] < budgets[2]):
        parser.error("Budgets must be positive and strictly increasing")
    if not source_codes or source_codes & CANOPY_CODES:
        parser.error("Source codes must be non-canopy classes")
    if args.replacement_code in source_codes:
        parser.error("The replacement code cannot also be a source code")
    if args.radius_m <= 0 or args.chunk_size <= 0:
        parser.error("Radius and chunk size must be positive")

    point_info = pyogrio.read_info(points_path)
    if point_info["geometry_type"] != "Point":
        raise ValueError("Candidate source must contain 2D Point geometries")
    if str(point_info["crs"]) != "EPSG:27700":
        raise ValueError(f"Expected candidate CRS EPSG:27700, got {point_info['crs']}")

    x, y = point_coordinate_views(points_path)
    record_count, dbf_views = _dbf_numeric_views(dbf_path, ("FID", "rank"))
    if len(x) != record_count:
        raise ValueError("SHP and DBF record counts differ")

    with rasterio.open(baseline_path) as baseline:
        if str(baseline.crs) != "EPSG:27700":
            raise ValueError(f"Expected baseline CRS EPSG:27700, got {baseline.crs}")
        transform = baseline.transform
        if transform.b != 0 or transform.d != 0:
            raise ValueError("A north-up baseline grid is required")
        pixel_width, pixel_height = transform.a, -transform.e
        if pixel_width <= 0 or pixel_height <= 0:
            raise ValueError("Invalid baseline pixel dimensions")
        base = baseline.read(1)
        valid = baseline.read_masks(1) > 0
        profile = baseline.profile.copy()

    allowed = valid & np.isin(base, sorted(source_codes))
    selected = np.zeros(base.size, dtype=bool)
    accepted_chunks: list[np.ndarray] = []
    cutoffs: dict[str, dict[str, int]] = {}
    accepted_count = 0
    radius_squared = args.radius_m**2
    flat_allowed = allowed.ravel()

    # Read only the two required DBF fields, then impose the documented stable
    # order. The source is rank-ordered but FIDs inside tied ranks are not.
    all_fids = _integer_chunk(dbf_views["FID"], 0, record_count, "FID")
    all_ranks = _integer_chunk(dbf_views["rank"], 0, record_count, "rank")
    source_is_rank_sorted = bool(np.all(all_ranks[1:] >= all_ranks[:-1]))
    source_is_rank_fid_sorted = bool(
        np.all(
            (all_ranks[1:] > all_ranks[:-1])
            | (
                (all_ranks[1:] == all_ranks[:-1])
                & (all_fids[1:] > all_fids[:-1])
            )
        )
    )
    candidate_order = np.lexsort((all_fids, all_ranks))
    ordered_ranks = all_ranks[candidate_order]
    ordered_fids = all_fids[candidate_order]
    duplicate_pair = (ordered_ranks[1:] == ordered_ranks[:-1]) & (
        ordered_fids[1:] == ordered_fids[:-1]
    )
    if np.any(duplicate_pair):
        raise ValueError("Candidate source contains a duplicate (rank, FID) pair")

    for start in range(0, record_count, args.chunk_size):
        stop = min(start + args.chunk_size, record_count)
        source_records = candidate_order[start:stop]
        ranks = ordered_ranks[start:stop]
        fids = ordered_fids[start:stop]

        xx, yy = x[source_records], y[source_records]
        columns = np.floor((xx - transform.c) / pixel_width).astype(np.int64)
        rows = np.floor((transform.f - yy) / pixel_height).astype(np.int64)
        in_bounds = (
            (rows >= 0)
            & (rows < base.shape[0])
            & (columns >= 0)
            & (columns < base.shape[1])
        )
        candidate_records = np.flatnonzero(in_bounds)
        if candidate_records.size:
            candidate_rows = rows[candidate_records]
            candidate_columns = columns[candidate_records]
            centre_x = transform.c + (candidate_columns + 0.5) * pixel_width
            centre_y = transform.f - (candidate_rows + 0.5) * pixel_height
            inside_crown = (
                (xx[candidate_records] - centre_x) ** 2
                + (yy[candidate_records] - centre_y) ** 2
                < radius_squared
            )
            candidate_records = candidate_records[inside_crown]

        if candidate_records.size:
            cells = rows[candidate_records] * base.shape[1] + columns[candidate_records]
            candidate_records = candidate_records[flat_allowed[cells]]
            cells = rows[candidate_records] * base.shape[1] + columns[candidate_records]
            # Retain the first ranked point for duplicate cells, then restore
            # candidate order because np.unique sorts by cell identifier.
            _, first_positions = np.unique(cells, return_index=True)
            first_positions.sort()
            candidate_records = candidate_records[first_positions]
            cells = rows[candidate_records] * base.shape[1] + columns[candidate_records]
            new = ~selected[cells]
            candidate_records, cells = candidate_records[new], cells[new]

        if candidate_records.size:
            remaining = budgets[-1] - accepted_count
            candidate_records, cells = candidate_records[:remaining], cells[:remaining]
            before = accepted_count
            accepted_count += len(cells)
            selected[cells] = True
            accepted_chunks.append(cells.astype(np.int64, copy=False))
            for label, budget in zip(SCENARIO_LABELS, budgets, strict=True):
                if label not in cutoffs and before < budget <= accepted_count:
                    local = budget - before - 1
                    cutoffs[label] = {
                        "accepted_pixel": budget,
                        "sorted_record_index_zero_based": int(
                            start + candidate_records[local]
                        ),
                        "source_record_index_zero_based": int(
                            source_records[candidate_records[local]]
                        ),
                        "source_record_number_one_based": int(
                            source_records[candidate_records[local]] + 1
                        ),
                        "rank": int(ranks[candidate_records[local]]),
                        "FID": int(fids[candidate_records[local]]),
                    }
        print(
            f"Checked {stop:,}/{record_count:,} candidates; "
            f"accepted {accepted_count:,}/{budgets[-1]:,}",
            flush=True,
        )
        if accepted_count == budgets[-1]:
            break

    if accepted_count < budgets[-1]:
        raise ValueError(
            f"Candidate source supplies only {accepted_count:,} eligible unique cells; "
            f"{budgets[-1]:,} required"
        )
    accepted_cells = np.concatenate(accepted_chunks)
    if len(accepted_cells) != budgets[-1] or len(np.unique(accepted_cells)) != len(
        accepted_cells
    ):
        raise AssertionError("Accepted-cell uniqueness/count invariant failed")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_names = {
        label: f"LULC_{label}_equal_area_eligible_v2.tif" for label in SCENARIO_LABELS
    }
    reserved = [output_dir / name for name in output_names.values()]
    reserved += [
        output_dir / "target_scenario_manifest_revised.json",
        output_dir / "target_transition_audit.csv",
        output_dir / "target_change_masks.png",
    ]
    existing = [path for path in reserved if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite existing outputs: {existing}")

    pixel_area_m2 = abs(
        transform.a * transform.e - transform.b * transform.d
    )
    profile.update(
        compress="lzw",
        tiled=True,
        blockxsize=256,
        blockysize=256,
        BIGTIFF="IF_SAFER",
    )
    scenario_records = []
    transition_rows = []
    base_flat = base.ravel()
    previous_cells: set[int] = set()
    for label, budget in zip(SCENARIO_LABELS, budgets, strict=True):
        cells = accepted_cells[:budget]
        cell_set = set(map(int, cells))
        if not previous_cells.issubset(cell_set):
            raise AssertionError(f"Nesting check failed for {label}")
        previous_cells = cell_set
        source_counts = Counter(map(int, base_flat[cells]))
        if set(source_counts) - source_codes:
            raise AssertionError(f"Ineligible source transition found for {label}")
        scenario = base.copy()
        scenario.ravel()[cells] = args.replacement_code
        output_path = output_dir / output_names[label]
        with rasterio.open(output_path, "w", **profile) as target:
            target.write(scenario, 1)
        for source_code in sorted(source_codes):
            transition_rows.append(
                {
                    "scenario": label,
                    "source_code": source_code,
                    "target_code": args.replacement_code,
                    "pixels": source_counts[source_code],
                    "area_km2": source_counts[source_code]
                    * pixel_area_m2
                    / 1_000_000,
                }
            )
        scenario_records.append(
            {
                "scenario": label,
                "output": str(output_path),
                "output_sha256": _sha256(output_path),
                "added_pixels": budget,
                "added_area_km2": budget * pixel_area_m2 / 1_000_000,
                "transitions": {
                    f"{code}->{args.replacement_code}": source_counts[code]
                    for code in sorted(source_codes)
                },
                "cutoff": cutoffs[label],
            }
        )

    transition_path = output_dir / "target_transition_audit.csv"
    with transition_path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(transition_rows[0]))
        writer.writeheader()
        writer.writerows(transition_rows)
    preview_path = output_dir / "target_change_masks.png"
    _write_preview(preview_path, valid, accepted_cells, budgets, transform)

    component_checksums = {}
    for suffix in (".shp", ".shx", ".dbf", ".prj", ".cpg"):
        component = points_path.with_suffix(suffix)
        if component.is_file():
            component_checksums[component.name] = _sha256(component)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "method": (
            "screened Target candidates in verified (rank, FID) order; pixel "
            "centre strictly within candidate radius; unique eligible cells"
        ),
        "baseline": str(baseline_path),
        "baseline_sha256": _sha256(baseline_path),
        "candidate_points": str(points_path),
        "candidate_components_sha256": component_checksums,
        "candidate_records_total": record_count,
        "source_is_rank_sorted": source_is_rank_sorted,
        "source_is_rank_fid_sorted": source_is_rank_fid_sorted,
        "applied_order": ["rank ascending", "FID ascending"],
        "candidate_crs": str(point_info["crs"]),
        "baseline_crs": str(profile["crs"]),
        "baseline_shape": list(base.shape),
        "baseline_transform": list(transform),
        "baseline_nodata": profile["nodata"],
        "pixel_area_m2": pixel_area_m2,
        "radius_m": args.radius_m,
        "source_codes": sorted(source_codes),
        "canopy_codes": sorted(CANOPY_CODES),
        "replacement_code": args.replacement_code,
        "budgets": dict(zip(SCENARIO_LABELS, budgets, strict=True)),
        "scenarios": scenario_records,
        "transition_audit": str(transition_path),
        "transition_audit_sha256": _sha256(transition_path),
        "change_mask_preview": str(preview_path),
        "change_mask_preview_sha256": _sha256(preview_path),
        "software": {
            "python": sys.version,
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "rasterio": rasterio.__version__,
            "gdal": rasterio.__gdal_version__,
            "pyogrio": pyogrio.__version__,
        },
        "git": _git_record(repo_root),
        "environment": {
            "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
        },
        "qa": {
            "exact_budgets": True,
            "allowed_transitions_only": True,
            "nested": True,
            "baseline_grid_and_nodata_preserved": True,
        },
    }
    manifest_path = output_dir / "target_scenario_manifest_revised.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Saved revised Target scenarios and audit: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
