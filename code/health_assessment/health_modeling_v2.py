#!/usr/bin/env python3
"""Strict, windowed heat-mortality model driven by one JSON configuration.

Version 2 holds 2021 population and mortality fixed, requires all rasters to
share one grid, processes raster windows to bound memory use, and records a
machine-readable run manifest. The legacy ``health-modeling.py`` is retained
unchanged for comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import math
import subprocess
import sys
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Cause:
    name: str
    rr_1c: float
    rr_low_1c: float
    rr_high_1c: float
    mortality_raster: Path


def attributable_fraction(beta: float | np.ndarray, delta_t: np.ndarray) -> np.ndarray:
    """Return ``1 - exp(-beta * delta_t)`` for a log-linear RR model."""
    with np.errstate(over="ignore", invalid="ignore"):
        return 1.0 - np.exp(-np.asarray(beta) * delta_t)


def beta_standard_error(rr_low_1c: float, rr_high_1c: float) -> float:
    return float((np.log(rr_high_1c) - np.log(rr_low_1c)) / (2.0 * 1.96))


def cause_beta_draws(cause: Cause, n_draws: int, seed: int) -> np.ndarray:
    """Generate draws stable across scenarios and independent of cause order."""
    cause_key = int.from_bytes(
        hashlib.sha256(cause.name.encode("utf-8")).digest()[:8], "little"
    )
    rng = np.random.default_rng(np.random.SeedSequence([seed, cause_key]))
    return rng.normal(np.log(cause.rr_1c), beta_standard_error(
        cause.rr_low_1c, cause.rr_high_1c
    ), n_draws)


def monte_carlo_totals_from_moments(
    beta_draws: np.ndarray, weighted_delta_moments: np.ndarray
) -> np.ndarray:
    """Evaluate weighted ``1-exp(-beta*deltaT)`` totals from raster moments.

    The Taylor series is algebraically separable into beta powers and spatial
    sums of ``deaths * deltaT**k``. This avoids reevaluating the exponential at
    every raster cell for every draw while retaining effectively exact results
    for the small temperature-response products in this analysis.
    """
    totals = np.zeros(beta_draws.size, dtype=np.float64)
    for index, moment in enumerate(weighted_delta_moments, start=1):
        coefficient = (-1.0) ** (index + 1) / math.factorial(index)
        totals += coefficient * beta_draws**index * moment
    return totals


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for block in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_commit(repository_root: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repository_root, check=True,
            capture_output=True, text=True
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def _git_dirty(repository_root: Path) -> bool | None:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"], cwd=repository_root, check=True,
            capture_output=True, text=True
        )
        return bool(result.stdout.strip())
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def _expand_path(value: str, base: Path) -> Path:
    expanded = os.path.expandvars(os.path.expanduser(value))
    if "$" in expanded:
        raise ValueError(f"Path contains an unresolved environment variable: {value}")
    path = Path(expanded)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _load_config(path: Path, scenario_name: str) -> tuple[dict[str, Any], dict[str, Any], list[Cause]]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1:
        raise ValueError("health configuration must have schema_version 1")
    if config.get("design", {}).get("population_year") != 2021:
        raise ValueError("version 2 requires design.population_year = 2021")
    if config.get("design", {}).get("mortality_year") != 2021:
        raise ValueError("version 2 requires design.mortality_year = 2021")

    data_root = _expand_path(config["data_root"], path.parent)
    scenario = config.get("scenarios", {}).get(scenario_name)
    if scenario is None:
        raise ValueError(f"Unknown scenario {scenario_name!r}")
    if not scenario.get("enabled", False):
        raise ValueError(f"Scenario {scenario_name!r} is disabled pending review")

    causes: list[Cause] = []
    seen: set[str] = set()
    for row in config.get("causes", []):
        name = str(row["name"]).strip()
        if not name or name in seen:
            raise ValueError(f"Cause names must be non-empty and unique: {name!r}")
        rr, low, high = map(float, (row["rr_1c"], row["rr_low_1c"], row["rr_high_1c"]))
        if not (0 < low <= rr <= high):
            raise ValueError(f"Invalid RR interval for {name}: {low}, {rr}, {high}")
        causes.append(Cause(name, rr, low, high, _expand_path(row["mortality_raster"], data_root)))
        seen.add(name)
    if not causes:
        raise ValueError("At least one cause configuration is required")

    resolved = {
        "data_root": data_root,
        "population_2021": _expand_path(config["population_2021"], data_root),
        "output_root": _expand_path(config["output_root"], data_root),
        "t_baseline": _expand_path(scenario["t_baseline"], data_root),
        "t_scenario": _expand_path(scenario["t_scenario"], data_root),
    }
    return config, {**scenario, **resolved}, causes


def _same_grid(reference: rasterio.io.DatasetReader, other: rasterio.io.DatasetReader) -> bool:
    return (
        reference.crs == other.crs
        and reference.transform.almost_equals(other.transform)
        and reference.width == other.width
        and reference.height == other.height
    )


def _validate_inputs(paths: dict[str, Path], causes: list[Cause]) -> dict[str, dict[str, Any]]:
    import rasterio

    all_paths = {**paths, **{f"mortality_{c.name}": c.mortality_raster for c in causes}}
    for name, path in all_paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing {name} input: {path}")

    inventory: dict[str, dict[str, Any]] = {}
    with rasterio.open(paths["population_2021"]) as reference:
        if reference.crs is None:
            raise ValueError("The 2021 population reference raster has no CRS")
        for name, path in all_paths.items():
            with rasterio.open(path) as source:
                if source.count != 1:
                    raise ValueError(f"{name} must contain exactly one raster band")
                if source.crs is None:
                    raise ValueError(f"{name} has no CRS")
                if not _same_grid(reference, source):
                    raise ValueError(
                        f"{name} does not match the 2021 population grid. Align it in a "
                        "separate validated preprocessing step; version 2 never resamples silently."
                    )
                inventory[name] = {
                    "path": str(path), "sha256": _sha256(path),
                    "crs": source.crs.to_string(), "width": source.width,
                    "height": source.height, "transform": tuple(source.transform),
                    "nodata": source.nodata,
                }
    return inventory


def _profile(reference: rasterio.io.DatasetReader) -> dict[str, Any]:
    profile = reference.profile.copy()
    profile.update(driver="GTiff", dtype="float32", count=1, compress="lzw", nodata=-9999.0)
    return profile


def _read(source: rasterio.io.DatasetReader, window: Any) -> np.ndarray:
    return source.read(1, window=window, masked=True).filled(np.nan).astype(np.float64)


def _run_model(
    scenario_name: str,
    scenario: dict[str, Any],
    causes: list[Cause],
    out_dir: Path,
    n_draws: int,
    seed: int,
    series_order: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    import rasterio

    deterministic = {cause.name: 0.0 for cause in causes}
    baseline_totals = {cause.name: 0.0 for cause in causes}
    valid_counts = {cause.name: 0 for cause in causes}
    draws = {cause.name: cause_beta_draws(cause, n_draws, seed) for cause in causes}
    weighted_delta_moments = {
        cause.name: np.zeros(series_order, dtype=np.float64) for cause in causes
    }

    with ExitStack() as stack:
        pop_src = stack.enter_context(rasterio.open(scenario["population_2021"]))
        t0_src = stack.enter_context(rasterio.open(scenario["t_baseline"]))
        ts_src = stack.enter_context(rasterio.open(scenario["t_scenario"]))
        mortality_sources = {
            cause.name: stack.enter_context(rasterio.open(cause.mortality_raster))
            for cause in causes
        }
        profile = _profile(pop_src)
        delta_writer = stack.enter_context(rasterio.open(out_dir / "deltaT_degC.tif", "w", **profile))
        excess_writers = {
            cause.name: stack.enter_context(
                rasterio.open(out_dir / f"Excess_{cause.name}.tif", "w", **profile)
            ) for cause in causes
        }
        af_writers = {
            cause.name: stack.enter_context(
                rasterio.open(out_dir / f"AF_{cause.name}.tif", "w", **profile)
            ) for cause in causes
        }

        total_population = 0.0
        positive_population_pixels = 0
        common_temperature_pixels = 0
        for _, window in pop_src.block_windows(1):
            population = _read(pop_src, window)
            t0 = _read(t0_src, window)
            ts = _read(ts_src, window)
            positive_pop = np.isfinite(population) & (population > 0)
            if np.any(np.isfinite(population) & (population < 0)):
                raise ValueError("2021 population contains negative values")
            temperature_valid = np.isfinite(t0) & np.isfinite(ts)
            common = positive_pop & temperature_valid
            delta_t = np.full(population.shape, np.nan, dtype=np.float64)
            delta_t[common] = ts[common] - t0[common]
            delta_writer.write(np.where(np.isfinite(delta_t), delta_t, -9999).astype("float32"), 1, window=window)
            total_population += float(np.nansum(np.where(positive_pop, population, np.nan)))
            positive_population_pixels += int(positive_pop.sum())
            common_temperature_pixels += int(common.sum())

            for cause in causes:
                deaths = _read(mortality_sources[cause.name], window)
                if np.any(np.isfinite(deaths) & (deaths < 0)):
                    raise ValueError(f"Mortality raster for {cause.name} contains negative values")
                valid = common & np.isfinite(deaths)
                beta = float(np.log(cause.rr_1c))
                af = np.full(population.shape, np.nan, dtype=np.float64)
                excess = np.full(population.shape, np.nan, dtype=np.float64)
                af[valid] = attributable_fraction(beta, delta_t[valid])
                excess[valid] = af[valid] * deaths[valid]
                af_writers[cause.name].write(np.where(np.isfinite(af), af, -9999).astype("float32"), 1, window=window)
                excess_writers[cause.name].write(np.where(np.isfinite(excess), excess, -9999).astype("float32"), 1, window=window)
                deterministic[cause.name] += float(np.nansum(excess))
                baseline_totals[cause.name] += float(np.nansum(np.where(valid, deaths, np.nan)))
                valid_counts[cause.name] += int(valid.sum())

                if n_draws:
                    # Accumulate spatial moments once; after all windows are
                    # read, these evaluate every Monte Carlo draw in O(draws ×
                    # series_order), rather than O(draws × raster_cells).
                    dt = delta_t[valid]
                    death_values = deaths[valid]
                    delta_power = dt.copy()
                    for moment_index in range(series_order):
                        weighted_delta_moments[cause.name][moment_index] += float(
                            np.dot(death_values, delta_power)
                        )
                        delta_power *= dt

    deterministic_rows = []
    for cause in causes:
        excess = deterministic[cause.name]
        deterministic_rows.append({
            "scenario": scenario_name,
            "scenario_label": scenario.get("label", scenario_name),
            "cause": cause.name,
            "RR_1C": cause.rr_1c,
            "baseline_registered_deaths_2021": baseline_totals[cause.name],
            "excess_deaths": excess,
            "deaths_averted": -excess,
            "valid_pixel_count": valid_counts[cause.name],
        })
    deterministic_frame = pd.DataFrame(deterministic_rows)

    draws_frame = pd.DataFrame({"draw": np.arange(1, n_draws + 1, dtype=int)})
    for cause in causes:
        draws_frame[cause.name] = monte_carlo_totals_from_moments(
            draws[cause.name], weighted_delta_moments[cause.name]
        )

    coverage = {
        "population_total_2021": total_population,
        "positive_population_pixel_count": positive_population_pixels,
        "common_temperature_pixel_count": common_temperature_pixels,
        "temperature_coverage_of_populated_pixels": (
            common_temperature_pixels / positive_population_pixels
            if positive_population_pixels else 0.0
        ),
    }
    return deterministic_frame, draws_frame, coverage


def run(args: argparse.Namespace) -> None:
    import rasterio

    config_path = Path(args.config).resolve()
    config, scenario, causes = _load_config(config_path, args.scenario)
    configured_draws = int(config.get("monte_carlo", {}).get("n_draws", 0))
    n_draws = configured_draws if args.n_draws is None else args.n_draws
    seed = int(config.get("monte_carlo", {}).get("seed", 20260908))
    if n_draws < 0 or args.series_order <= 0:
        raise ValueError("n_draws must be non-negative and series-order positive")

    input_paths = {
        "population_2021": scenario["population_2021"],
        "t_baseline": scenario["t_baseline"],
        "t_scenario": scenario["t_scenario"],
    }
    inventory = _validate_inputs(input_paths, causes)
    print(f"Validated {len(inventory)} inputs on one common grid")
    if args.validate_only:
        return

    out_dir = scenario["output_root"] / args.scenario
    if out_dir.exists() and any(out_dir.iterdir()) and not args.force:
        raise FileExistsError(
            f"Refusing to overwrite non-empty output directory {out_dir}; review it or use --force"
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    deterministic, draws, coverage = _run_model(
        args.scenario, scenario, causes, out_dir, n_draws, seed, args.series_order
    )
    deterministic.to_csv(out_dir / "city_totals_deterministic.csv", index=False)
    if n_draws:
        draws.to_csv(out_dir / "city_total_draws_by_cause.csv", index=False)
        summary_rows = []
        for cause in causes:
            values = draws[cause.name].to_numpy()
            low, median, high = np.percentile(values, [2.5, 50, 97.5])
            summary_rows.append({
                "scenario": args.scenario, "cause": cause.name,
                "n_draws": n_draws, "seed": seed,
                "excess_p2p5": low, "excess_p50": median,
                "excess_p97p5": high, "excess_mean": values.mean(),
                "deaths_averted_mean": -values.mean(),
            })
        pd.DataFrame(summary_rows).to_csv(out_dir / "city_totals_monte_carlo.csv", index=False)

    repository_root = Path(__file__).resolve().parents[2]
    manifest = {
        "schema_version": 1,
        "scenario": args.scenario,
        "scenario_label": scenario.get("label", args.scenario),
        "design": config["design"],
        "model": "RR=exp(beta*deltaT); AF=(RR-1)/RR; excess=AF*baseline deaths",
        "sign_convention": "negative excess_deaths means deaths averted",
        "monte_carlo": {
            "n_draws": n_draws,
            "configured_n_draws": configured_draws,
            "command_line_override": args.n_draws is not None,
            "seed": seed,
            "pairing": "cause-stable draws",
            "city_total_method": "weighted delta-temperature Taylor moments",
            "series_order": args.series_order,
        },
        "coverage": coverage,
        "inputs": inventory,
        "causes": [
            {"name": c.name, "rr_1c": c.rr_1c, "rr_low_1c": c.rr_low_1c,
             "rr_high_1c": c.rr_high_1c, "mortality_raster": str(c.mortality_raster)}
            for c in causes
        ],
        "software": {
            "python": platform.python_version(), "numpy": np.__version__,
            "pandas": pd.__version__, "rasterio": rasterio.__version__,
            "git_commit": _git_commit(repository_root),
            "git_worktree_dirty": _git_dirty(repository_root),
            "model_script_sha256": _sha256(Path(__file__).resolve()),
        },
        "configuration": {"path": str(config_path), "sha256": _sha256(config_path)},
    }
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Completed {args.scenario}: {out_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Version 2 JSON configuration")
    parser.add_argument("--scenario", required=True, help="Scenario key in the configuration")
    parser.add_argument("--validate-only", action="store_true", help="Validate paths and grids only")
    parser.add_argument("--force", action="store_true", help="Write into a reviewed non-empty output directory")
    parser.add_argument(
        "--series-order", type=int, default=12,
        help="Taylor-series order for fast, effectively exact Monte Carlo city totals",
    )
    parser.add_argument(
        "--n-draws", type=int, default=None,
        help="Override configured Monte Carlo draws; use 0 for a deterministic QA run",
    )
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
