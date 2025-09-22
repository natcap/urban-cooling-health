
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
General template: Projecting summer (July) heat-attributable excess deaths from spatial dT

This script implements the grid-by-grid workflow described in the Methods docx:
 - Align July temperature rasters (baseline 2020 vs. future 2050)
 - Compute dT (degC) per pixel
 - Apply cause-specific temperature-mortality exposure-response (log-linear)
 - Compute AF and excess deaths using population and mortality-rate rasters
 - Aggregate to city totals
 - Optional Monte Carlo uncertainty
 - Save GeoTIFFs and CSV summaries

USAGE (example):
    python heat_mortality_template.py         --t2020 path/to/T_july_2020.tif         --t2050 path/to/T_july_2050.tif         --pop2050 path/to/pop_2050.tif         --mort_cardio path/to/mort_rate_cardio_2050.tif         --mort_resp path/to/mort_rate_resp_2050.tif         --mort_cere path/to/mort_rate_cere_2050.tif         --out_dir ./outputs         --n_draws 1000

NOTES:
 - All rasters should be in the SAME CRS and cover the same urban extent.
 - If rasters are at different resolutions/CRS, set --align_to to one of the inputs;
   the script will reproject/align the others to match.
 - Mortality rates must be deaths per person per YEAR.
 - Population is persons per pixel.
 - For city totals, the script sums per-pixel values, ignoring NaNs.
 - Causes: cardiovascular, respiratory, cerebrovascular (customizable).

Dependencies:
    rasterio, numpy, pandas
"""

import argparse
import json
import math
import sys
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject

# -----------------------------
# Configuration structures
# -----------------------------

@dataclass
class CauseConfig:
    name: str
    rr_1c: float           # Relative risk per +1 degC
    rr_low_1c: float       # 95% CI lower
    rr_high_1c: float      # 95% CI upper


DEFAULT_CAUSES = [
    CauseConfig("cardiovascular", 1.0344, 1.0310, 1.0378),
    CauseConfig("respiratory",    1.0360, 1.0318, 1.0402),
    CauseConfig("cerebrovascular",1.0140, 1.0006, 1.0275),
]

# -----------------------------
# Raster utilities
# -----------------------------

def _write_geotiff(path: str, array: np.ndarray, profile_like, nodata=np.nan):
    prof = profile_like.copy()
    prof.update(
        dtype=rasterio.float32,
        count=1,
        compress="lzw",
        nodata=nodata
    )
    if isinstance(nodata, float) and math.isnan(nodata):
        prof.pop("nodata", None)
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(array.astype(np.float32), 1)


def _align_to_template(src_path: str, template_src) -> np.ndarray:
    """Reproject/align a raster to match the template dataset's grid/CRS."""
    with rasterio.open(src_path) as src:
        dst_arr = np.empty((template_src.height, template_src.width), dtype=np.float32)
        dst_arr[:] = np.nan
        reproject(
            source=rasterio.band(src, 1),
            destination=dst_arr,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=template_src.transform,
            dst_crs=template_src.crs,
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )
    return dst_arr


def _ensure_aligned(paths, align_to=None):
    """
    Open rasters and align all to a template grid.
    Returns the template dataset and a dict of aligned arrays.
    """
    if align_to is None:
        first_key = next(iter(paths.keys()))
        template_path = paths[first_key]
    else:
        template_path = align_to

    template_src = rasterio.open(template_path)
    arrays = {}

    for key, p in paths.items():
        if p == template_path:
            arrays[key] = template_src.read(1, masked=True).filled(np.nan).astype(np.float32)
        else:
            arrays[key] = _align_to_template(p, template_src)

    return template_src, arrays

# -----------------------------
# Epidemiology helpers
# -----------------------------

def beta_from_rr(rr_1c: float) -> float:
    """
    Slope on log scale per +1 degC.
    """
    return float(np.log(rr_1c))


def se_from_ci(rr_low: float, rr_high: float) -> float:
    """
    Standard error of ln(RR) from 95% CI (assuming normality of log RR).
    """
    return (np.log(rr_high) - np.log(rr_low)) / (2.0 * 1.96)


def rr_from_deltaT(beta: float, dT: np.ndarray) -> np.ndarray:
    """
    RR(x,y) = exp(beta * dT).
    """
    with np.errstate(over='ignore', invalid='ignore'):
        return np.exp(beta * dT)


def af_from_rr(rr: np.ndarray) -> np.ndarray:
    """
    AF = (RR - 1) / RR; returns NaN where RR is NaN or <=0.
    """
    af = (rr - 1.0) / rr
    af[~np.isfinite(af)] = np.nan
    return af


def excess_deaths(af: np.ndarray, mort_rate: np.ndarray, population: np.ndarray) -> np.ndarray:
    """
    Excess = AF * (mort_rate * population).
    """
    baseline_deaths = mort_rate * population
    excess = af * baseline_deaths
    excess[~np.isfinite(excess)] = np.nan
    return excess

# -----------------------------
# Main pipeline
# -----------------------------

def main(args):
    os.makedirs(args.out_dir, exist_ok=True)

    # 1) Align all required rasters
    path_map = {
        "t2020": args.t2020,
        "t2050": args.t2050,
        "pop2050": args.pop2050,
        "mort_cardio": args.mort_cardio,
        "mort_resp": args.mort_resp,
        "mort_cere": args.mort_cere,
    }

    template_src, arrays = _ensure_aligned(path_map, align_to=args.align_to)

    T2020 = arrays["t2020"]
    T2050 = arrays["t2050"]
    dT = T2050 - T2020  # degC
    _write_geotiff(os.path.join(args.out_dir, "deltaT_degC.tif"), dT, template_src.profile)

    pop = arrays["pop2050"]
    mort_cardio = arrays["mort_cardio"]
    mort_resp   = arrays["mort_resp"]
    mort_cere   = arrays["mort_cere"]

    # 2) Deterministic estimates for each cause
    results = []
    for cause in DEFAULT_CAUSES:
        if cause.name == "cardiovascular":
            mort_rate = mort_cardio
        elif cause.name == "respiratory":
            mort_rate = mort_resp
        elif cause.name == "cerebrovascular":
            mort_rate = mort_cere
        else:
            continue

        beta = beta_from_rr(cause.rr_1c)
        rr = rr_from_deltaT(beta, dT)
        af = af_from_rr(rr)
        exc = excess_deaths(af, mort_rate, pop)

        # Save GeoTIFFs
        _write_geotiff(os.path.join(args.out_dir, f"AF_{cause.name}.tif"), af, template_src.profile)
        _write_geotiff(os.path.join(args.out_dir, f"Excess_{cause.name}.tif"), exc, template_src.profile)

        # City totals (ignore NaN)
        total_excess = float(np.nansum(exc))
        total_baseline = float(np.nansum(mort_rate * pop))

        results.append({
            "cause": cause.name,
            "RR_1C": cause.rr_1c,
            "city_total_baseline_deaths": total_baseline,
            "city_total_excess_deaths": total_excess,
        })

    pd.DataFrame(results).to_csv(os.path.join(args.out_dir, "city_totals_deterministic.csv"), index=False)

    # 3) Optional Monte Carlo
    if args.n_draws and args.n_draws > 0:
        rng = np.random.default_rng(args.seed)
        mc_rows = []
        for cause in DEFAULT_CAUSES:
            if cause.name == "cardiovascular":
                mort_rate = mort_cardio
            elif cause.name == "respiratory":
                mort_rate = mort_resp
            elif cause.name == "cerebrovascular":
                mort_rate = mort_cere
            else:
                continue

            beta_hat = beta_from_rr(cause.rr_1c)
            se = se_from_ci(cause.rr_low_1c, cause.rr_high_1c)

            betas = rng.normal(loc=beta_hat, scale=se, size=args.n_draws)

            totals = []
            for b in betas:
                rr = rr_from_deltaT(b, dT)
                af = af_from_rr(rr)
                exc = excess_deaths(af, mort_rate, pop)
                totals.append(float(np.nansum(exc)))

            totals = np.array(totals, dtype=float)
            low, med, high = np.percentile(totals, [2.5, 50, 97.5])
            mc_rows.append({
                "cause": cause.name,
                "n_draws": args.n_draws,
                "seed": args.seed,
                "excess_p2p5": float(low),
                "excess_p50": float(med),
                "excess_p97p5": float(high),
            })

        pd.DataFrame(mc_rows).to_csv(os.path.join(args.out_dir, "city_totals_monte_carlo.csv"), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Project heat-attributable excess deaths (July) from spatial dT")
    parser.add_argument("--t2020", required=True, help="Baseline July 2020 temperature raster (degC)")
    parser.add_argument("--t2050", required=True, help="Projected July 2050 temperature raster (degC)")
    parser.add_argument("--pop2050", required=True, help="Population raster for 2050 (persons/pixel)")
    parser.add_argument("--mort_cardio", required=True, help="Mortality rate raster 2050, cardiovascular (deaths/person/year)")
    parser.add_argument("--mort_resp", required=True, help="Mortality rate raster 2050, respiratory (deaths/person/year)")
    parser.add_argument("--mort_cere", required=True, help="Mortality rate raster 2050, cerebrovascular (deaths/person/year)")
    parser.add_argument("--out_dir", required=True, help="Output directory")
    parser.add_argument("--align_to", default=None, help="Optional path to raster used as alignment template")
    parser.add_argument("--n_draws", type=int, default=0, help="Monte Carlo draws for uncertainty (0 to skip)")
    parser.add_argument("--seed", type=int, default=2025, help="Random seed for Monte Carlo")

    args = parser.parse_args()
    main(args)
