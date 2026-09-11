#!/usr/bin/env python3
"""Convert InVEST WBGT rasters to Hothaps heavy-work productivity.

The output is a workability fraction from 0.1 to 1.0. Multiply a scenario
difference by 100 to report a percentage-point productivity change. This
script replaces the old machine-specific loop and deliberately does not use
InVEST's threshold-based ``heavy_work_loss_percent`` raster.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy
import pygeoprocessing


DEFAULT_ALPHA1 = 30.94
DEFAULT_ALPHA2 = 16.64
OUTPUT_NODATA = -1.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _workability(wbgt: numpy.ndarray, alpha1: float, alpha2: float) -> numpy.ndarray:
    """Return the Hothaps workability fraction for valid WBGT values."""
    return (
        0.1 + 0.9 / (1.0 + numpy.power(wbgt / alpha1, alpha2))
    ).astype(numpy.float32)


def _raster_summary(path: Path) -> dict[str, object]:
    info = pygeoprocessing.get_raster_info(str(path))
    stats = info.get("statistics")
    if stats and stats[0] is not None:
        statistics = list(stats[0])
    else:
        # Newly written rasters do not always have GDAL statistics metadata.
        nodata = info["nodata"][0]
        minimum = numpy.inf
        maximum = -numpy.inf
        for _, block in pygeoprocessing.iterblocks((str(path), 1)):
            valid = numpy.isfinite(block)
            if nodata is not None:
                valid &= ~numpy.isclose(block, nodata)
            if valid.any():
                minimum = min(minimum, float(block[valid].min()))
                maximum = max(maximum, float(block[valid].max()))
        if not numpy.isfinite(minimum):
            raise ValueError(f"Raster has no valid pixels: {path}")
        statistics = [minimum, maximum, None, None]
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
        "raster_size": list(info["raster_size"]),
        "pixel_size": list(info["pixel_size"]),
        "bounding_box": list(info["bounding_box"]),
        "projection_wkt": info["projection_wkt"],
        "nodata": list(info["nodata"]),
        "statistics": statistics,
    }


def _find_wbgt_rasters(root: Path, scenarios: set[str] | None) -> list[Path]:
    rasters = sorted(root.glob("*/intermediate/wbgt_*.tif"))
    if scenarios is not None:
        rasters = [path for path in rasters if path.parents[1].name in scenarios]
    return rasters


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ucm_output_root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--scenarios", nargs="+")
    parser.add_argument("--alpha1", type=float, default=DEFAULT_ALPHA1)
    parser.add_argument("--alpha2", type=float, default=DEFAULT_ALPHA2)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    input_root = args.ucm_output_root.expanduser().resolve()
    output_root = (
        args.output_root.expanduser().resolve()
        if args.output_root
        else input_root / "hothaps_work_intensity"
    )
    if args.alpha1 <= 0 or args.alpha2 <= 0:
        parser.error("alpha1 and alpha2 must be positive")
    scenario_filter = set(args.scenarios) if args.scenarios else None
    wbgt_rasters = _find_wbgt_rasters(input_root, scenario_filter)
    if not wbgt_rasters:
        parser.error(f"No matching WBGT rasters found under {input_root}")

    for wbgt_path in wbgt_rasters:
        scenario = wbgt_path.parents[1].name
        suffix = wbgt_path.stem.removeprefix("wbgt_")
        scenario_output = output_root / scenario
        target_path = scenario_output / f"high_work_productivity_{suffix}.tif"
        manifest_path = scenario_output / f"high_work_productivity_{suffix}.json"
        if args.validate_only:
            info = pygeoprocessing.get_raster_info(str(wbgt_path))
            if len(info["raster_size"]) != 2 or info["nodata"][0] is None:
                raise ValueError(f"WBGT raster lacks an explicit valid grid: {wbgt_path}")
            print(f"Validated {wbgt_path}")
            continue
        if not args.overwrite and (target_path.exists() or manifest_path.exists()):
            raise FileExistsError(f"Refusing to overwrite {target_path}")

        scenario_output.mkdir(parents=True, exist_ok=True)
        pygeoprocessing.raster_map(
            lambda wbgt: _workability(wbgt, args.alpha1, args.alpha2),
            [str(wbgt_path)],
            str(target_path),
            target_nodata=OUTPUT_NODATA,
            target_dtype=numpy.float32,
        )
        summary = _raster_summary(target_path)
        minimum, maximum = summary["statistics"][0], summary["statistics"][1]
        if minimum < 0.1 - 1e-6 or maximum > 1.0 + 1e-6:
            raise ValueError(
                f"Workability values outside [0.1, 1.0] for {target_path}: "
                f"{minimum}, {maximum}"
            )
        manifest = {
            "schema_version": 1,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "scenario": scenario,
            "method": "Hothaps workability fraction",
            "formula": "0.1 + 0.9 / (1 + (WBGT / alpha1)^alpha2)",
            "parameters": {"alpha1": args.alpha1, "alpha2": args.alpha2},
            "units": "fraction of full heavy-work productivity",
            "input": _raster_summary(wbgt_path),
            "output": summary,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(f"Created {target_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
