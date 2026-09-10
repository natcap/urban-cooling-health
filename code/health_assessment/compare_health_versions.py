#!/usr/bin/env python3
"""Compare legacy area-weighted and revised population-weighted health inputs.

The script uses already aligned legacy mortality rasters for spatial metrics
and the saved deterministic city summaries for health-impact differences.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio


CAUSE_FILES = {
    "all_cause": (
        "aligned_baseline_deaths_all_cause.tif",
        "baseline_deaths_all_cause_population_weighted_2021.tif",
    ),
    "mental_disorder": (
        "aligned_baseline_deaths_mental_disorder.tif",
        "baseline_deaths_mental_disorder_population_weighted_2021.tif",
    ),
    "cardiovascular": (
        "aligned_baseline_deaths_cardio.tif",
        "baseline_deaths_cardiovascular_population_weighted_2021.tif",
    ),
    "respiratory": (
        "aligned_baseline_deaths_resp.tif",
        "baseline_deaths_respiratory_population_weighted_2021.tif",
    ),
    "self_harm": (
        "aligned_baseline_deaths_self_harm.tif",
        "baseline_deaths_self_harm_population_weighted_2021.tif",
    ),
}


def _read(path: Path) -> tuple[np.ndarray, tuple]:
    with rasterio.open(path) as source:
        if source.crs is None:
            raise ValueError(f"Raster has no CRS: {path}")
        values = source.read(1, masked=True).filled(np.nan).astype(np.float64)
        grid = (source.crs, source.transform, source.width, source.height)
    return values, grid


def _same_grid(candidate: tuple, reference: tuple, *, allow_legacy_bng: bool = False) -> bool:
    candidate_crs, *candidate_geometry = candidate
    reference_crs, *reference_geometry = reference
    if candidate_geometry != reference_geometry:
        return False
    if candidate_crs == reference_crs:
        return True
    # Archived rasters use a local CRS label but the same British National Grid.
    return (
        allow_legacy_bng
        and reference_crs.to_epsg() == 27700
        and "British National Grid" in candidate_crs.to_wkt()
    )


def compare_allocations(
    legacy_dir: Path, revised_dir: Path, population_path: Path
) -> pd.DataFrame:
    population, population_grid = _read(population_path)
    populated = np.isfinite(population) & (population > 0)
    threshold = float(np.quantile(population[populated], 0.9))
    top_population_pixels = populated & (population >= threshold)
    rows = []

    for cause, (legacy_name, revised_name) in CAUSE_FILES.items():
        legacy, legacy_grid = _read(legacy_dir / legacy_name)
        revised, revised_grid = _read(revised_dir / revised_name)
        # The legacy BNG label is non-standard; geometry must still match exactly.
        if not _same_grid(legacy_grid, population_grid, allow_legacy_bng=True):
            raise ValueError(f"Legacy grid differs from the population grid for {cause}")
        if not _same_grid(revised_grid, population_grid):
            raise ValueError(f"CRS, transform or dimensions differ for {cause}")
        legacy_valid = np.isfinite(legacy) & (legacy >= 0)
        revised_valid = np.isfinite(revised) & (revised >= 0)
        legacy_total = float(legacy[legacy_valid].sum())
        revised_total = float(revised[revised_valid].sum())
        legacy_zero_pop = float(legacy[legacy_valid & ~populated].sum())
        revised_zero_pop = float(revised[revised_valid & ~populated].sum())
        common = populated & legacy_valid & revised_valid

        rows.append(
            {
                "cause": cause,
                "legacy_aligned_total": legacy_total,
                "revised_total": revised_total,
                "revised_minus_legacy_total": revised_total - legacy_total,
                "revised_minus_legacy_total_pct": 100 * (revised_total / legacy_total - 1),
                "legacy_deaths_on_zero_population_pixels": legacy_zero_pop,
                "legacy_zero_population_share_pct": 100 * legacy_zero_pop / legacy_total,
                "revised_deaths_on_zero_population_pixels": revised_zero_pop,
                "revised_zero_population_share_pct": 100 * revised_zero_pop / revised_total,
                "legacy_population_correlation": float(
                    np.corrcoef(population[common], legacy[common])[0, 1]
                ),
                "revised_population_correlation": float(
                    np.corrcoef(population[common], revised[common])[0, 1]
                ),
                "legacy_death_share_in_top_population_decile_pct": 100
                * float(legacy[top_population_pixels & legacy_valid].sum())
                / legacy_total,
                "revised_death_share_in_top_population_decile_pct": 100
                * float(revised[top_population_pixels & revised_valid].sum())
                / revised_total,
            }
        )
    return pd.DataFrame(rows)


def _legacy_summary(path: Path, scenario: str) -> pd.DataFrame:
    frame = pd.read_csv(path).rename(
        columns={
            "city_total_baseline_deaths": "legacy_baseline_deaths",
            "city_total_excess_deaths": "legacy_excess_deaths",
        }
    )
    frame["scenario"] = scenario
    frame["legacy_deaths_averted"] = -frame["legacy_excess_deaths"]
    return frame[["scenario", "cause", "legacy_baseline_deaths", "legacy_deaths_averted"]]


def _revised_summary(path: Path, scenario: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["scenario"] = scenario
    return frame[
        ["scenario", "cause", "baseline_registered_deaths_2021", "deaths_averted"]
    ].rename(columns={"baseline_registered_deaths_2021": "revised_baseline_deaths"})


def compare_health(args: argparse.Namespace) -> pd.DataFrame:
    legacy = pd.concat(
        [
            _legacy_summary(Path(args.legacy_green_csv), "Green30_25c"),
            _legacy_summary(Path(args.legacy_target_csv), "Target30_25c"),
        ],
        ignore_index=True,
    )
    revised = pd.concat(
        [
            _revised_summary(Path(args.revised_green_csv), "Green30_25c"),
            _revised_summary(Path(args.revised_target_csv), "Target30_25c"),
        ],
        ignore_index=True,
    )
    result = legacy.merge(revised, on=["scenario", "cause"], validate="one_to_one")
    result["baseline_deaths_change"] = (
        result["revised_baseline_deaths"] - result["legacy_baseline_deaths"]
    )
    result["baseline_deaths_change_pct"] = 100 * (
        result["revised_baseline_deaths"] / result["legacy_baseline_deaths"] - 1
    )
    result["deaths_averted_change"] = (
        result["deaths_averted"] - result["legacy_deaths_averted"]
    )
    result["deaths_averted_change_pct"] = 100 * (
        result["deaths_averted"] / result["legacy_deaths_averted"] - 1
    )
    return result


def main(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    allocation = compare_allocations(
        Path(args.legacy_aligned_mortality_dir),
        Path(args.revised_mortality_dir),
        Path(args.population_2021),
    )
    health = compare_health(args)
    allocation.to_csv(output_dir / "mortality_allocation_comparison.csv", index=False)
    health.to_csv(output_dir / "health_impact_comparison_25c.csv", index=False)

    all_cause = health.loc[health["cause"] == "all_cause"].set_index("scenario")
    summary = pd.DataFrame(
        [
            {
                "version": "legacy_area_weighted",
                "green30_deaths_averted": all_cause.at["Green30_25c", "legacy_deaths_averted"],
                "target30_deaths_averted": all_cause.at["Target30_25c", "legacy_deaths_averted"],
            },
            {
                "version": "revised_population_weighted",
                "green30_deaths_averted": all_cause.at["Green30_25c", "deaths_averted"],
                "target30_deaths_averted": all_cause.at["Target30_25c", "deaths_averted"],
            },
        ]
    )
    summary["target_minus_green_deaths_averted"] = (
        summary["target30_deaths_averted"] - summary["green30_deaths_averted"]
    )
    summary["target_vs_green_pct"] = 100 * (
        summary["target30_deaths_averted"] / summary["green30_deaths_averted"] - 1
    )
    summary.to_csv(output_dir / "all_cause_scenario_comparison_25c.csv", index=False)
    print(f"Saved comparison tables to {output_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "legacy_aligned_mortality_dir", "revised_mortality_dir", "population_2021",
        "legacy_green_csv", "legacy_target_csv", "revised_green_csv",
        "revised_target_csv", "output_dir",
    ):
        parser.add_argument(f"--{name.replace('_', '-')}", required=True)
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
