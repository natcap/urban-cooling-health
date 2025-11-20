#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
General template: Projecting summer heat-attributable excess deaths from spatial dT

This script implements a grid-by-grid workflow for estimating heat-attributable mortality:
 - Align temperature rasters (baseline vs. scenario)
 - Compute dT (degC) per pixel
 - Apply cause-specific temperature-mortality exposure-response (log-linear)
 - Compute AF and excess deaths using population and mortality-rate rasters
 - Aggregate to city totals
 - Optional Monte Carlo uncertainty
 - Save GeoTIFFs and CSV summaries

USAGE (example):
    python heat_mortality_template.py \
        --t_baseline path/to/T_july_baseline.tif \
        --t_scenario path/to/T_july_scenario.tif \
        --pop_baseline path/to/pop_baseline.tif \
        --pop_scenario path/to/pop_scenario.tif \
        --baseline_deaths_cardio path/to/baseline_deaths_cardio.tif \
        --baseline_deaths_resp path/to/baseline_deaths_resp.tif \
        --baseline_deaths_self_harm path/to/baseline_deaths_self_harm.tif \
        --out_dir ./outputs \
        --n_draws 1000

NOTES:
 - All rasters should be in the SAME CRS and cover the same urban extent.
 - If rasters are at different resolutions/CRS, set --align_to to one of the inputs;
   the script will reproject/align the others to match.
 - Population is persons per pixel.
 - For city totals, the script sums per-pixel values, ignoring NaNs.
 - Causes: cardiovascular, respiratory, self_harm (customizable).

Dependencies:
    rasterio, numpy, pandas, os
"""

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject
from rasterio.crs import CRS
import subprocess

# -----------------------------
# Configuration structures
# -----------------------------

@dataclass
class CauseConfig:
    """Configuration for a specific cause of death"""
    name: str
    rr_1c: float           # Relative risk per +1 degC
    rr_low_1c: float       # 95% CI lower
    rr_high_1c: float      # 95% CI upper


# Default cause-specific relative risks (can be customized)
DEFAULT_CAUSES = [
    CauseConfig("cardiovascular", 1.0344, 1.0310, 1.0378),
    CauseConfig("respiratory",    1.0360, 1.0318, 1.0402),
    CauseConfig("self_harm",      1.0100, 1.0000, 1.0200),
]

# -----------------------------
# Raster utilities
# -----------------------------

def _write_geotiff(path: str, array: np.ndarray, profile_like, nodata=np.nan) -> None:
    """
    Write a numpy array to a GeoTIFF file using a reference profile.
    
    Args:
        path: Output file path
        array: Data array to write
        profile_like: Rasterio profile to use as template
        nodata: Value to use for nodata (default: np.nan)
    """
    prof = profile_like.copy()
    prof.update(
        dtype=rasterio.float32,
        count=1,
        compress="lzw",
        # nodata=nodata
        # nodata cannot be NaN in GeoTIFF metadata; keep unset if NaN
        # ensure we don't write NaN as nodata in metadata
    )
    if not (isinstance(nodata, float) and math.isnan(nodata)):
        prof["nodata"] = nodata
    else:
        prof.pop("nodata", None)
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(array.astype(np.float32), 1)





def read_aligned_arrays(aligned_files, reference_file=None):
    """Simply read pre-aligned files without additional processing"""
    arrays = {}
    template_profile = None
    
    for key, file_path in aligned_files.items():
        with rasterio.open(file_path) as src:
            arrays[key] = src.read(1, masked=True).filled(np.nan).astype(np.float32)
    # Pick template from aligned reference to guarantee transform/res/resolution
    if reference_file is not None:
        ref_aligned = os.path.join(os.path.dirname(list(aligned_files.values())[0]),
                                   f"aligned_{os.path.basename(reference_file)}") 
        with rasterio.open(ref_aligned if os.path.exists(ref_aligned) else reference_file) as src: 
            template_profile = src.profile.copy()
    else:
        # fallback: first item
        first_path = next(iter(aligned_files.values())) 
        with rasterio.open(first_path) as src:
            template_profile = src.profile.copy()
    return template_profile, arrays

# ---------- CRS NORMALIZATION HELPERS ----------



def _looks_like_bng(crs) -> bool:
    """Heuristic: detect BNG-like CRS definitions lacking EPSG authority."""
    if crs is None:
        return False
    s = crs.to_string()
    return (
        "OSGB36" in s
        or "Airy 1830" in s
        or "Transverse_Mercator" in s
        or "Longitude of natural origin" in s and "-2" in s
        or "False easting" in s and "400000" in s
        or "False northing" in s and "-100000" in s
    )  # <<< EDIT

def _coerce_to_ref_if_local(src_crs, ref_crs):
    """If source CRS is missing/LOCAL but looks like BNG, assume reference CRS."""
    if src_crs is None:
        return ref_crs  # <<< EDIT
    # Some drivers mark BNG as LOCAL/ENGCRS; if similar, reuse ref_crs
    try:
        if src_crs == ref_crs:
            return src_crs
    except Exception:
        pass
    return ref_crs if _looks_like_bng(src_crs) else src_crs 




# -----------------------------
def robust_gdal_alignment(input_files, output_dir, reference_file):
    """
    Align all rasters to the reference grid using GDAL (gdalwarp), with
    cautious CRS handling.

    - Uses the reference raster's bounds, resolution, and CRS as the target.
    - Applies -t_srs to reproject into the reference CRS.
    - Applies -s_srs ONLY when the source CRS is missing or looks like a
      local/BNG-like definition (see _looks_like_bng).
    - Uses bilinear resampling (suitable for continuous fields; counts will
      be handled in a density-preserving way later by force_align_with_reproject).
    - On failure, falls back to the original input file for that key.

    Args:
        input_files: dict[str, str]
            Mapping from logical key (e.g., "t_baseline") to input file path.
        output_dir: str
            Directory where aligned files will be written.
        reference_file: str
            Path to the reference raster that defines the target grid.

    Returns:
        aligned_files: dict[str, str]
            Mapping from key to aligned file path. If alignment fails for a
            given file, the original path is returned for that key.
    """
    import subprocess

    os.makedirs(output_dir, exist_ok=True)
    aligned_files = {}

    # Get precise reference properties
    with rasterio.open(reference_file) as ref:
        ref_bounds = ref.bounds
        ref_res = ref.res
        ref_crs_obj = ref.crs
        ref_shape = ref.shape

    # Prefer WKT for maximum robustness; None if CRS is missing
    ref_crs_str = ref_crs_obj.to_wkt() if ref_crs_obj else None

    print(f"Reference for GDAL alignment:")
    print(f"  file  : {reference_file}")
    print(f"  bounds: {ref_bounds}")
    print(f"  res   : {ref_res}")
    print(f"  crs   : {ref_crs_obj or 'None'}")
    print(f"  shape : {ref_shape}")
    print("---")

    for key, input_file in input_files.items():
        print(f"Aligning {key} from {input_file} ...")

        # Try to inspect the source CRS; if this fails, we treat as unknown.
        try:
            with rasterio.open(input_file) as src:
                src_crs = src.crs
        except Exception as e:
            print(f"  Warning: could not read CRS for {input_file}: {e}")
            src_crs = None

        output_file = os.path.join(
            output_dir, f"aligned_{key}.tif"
        )

        # Base gdalwarp command: clip to ref bounds + match resolution
        cmd = [
            "gdalwarp",
            "-te", str(ref_bounds.left), str(ref_bounds.bottom),
                   str(ref_bounds.right), str(ref_bounds.top),
            "-tr", str(ref_res[0]), str(ref_res[1]),
        ]

        # Target CRS: always warp into the reference CRS if present
        if ref_crs_str:
            cmd += ["-t_srs", ref_crs_str]

        # Resampling + grid alignment
        cmd += [
            "-r", "bilinear",   # continuous fields; counts handled later
            "-tap",             # align to reference grid
            "-overwrite",
            "-dstnodata", "-9999",
        ]

        # Only hint the source CRS if it is missing or looks like a local/BNG-like CRS.
        # We DO NOT override a well-defined CRS with the reference CRS.
        if ref_crs_str and (src_crs is None or _looks_like_bng(src_crs)):
            cmd += ["-s_srs", ref_crs_str]

        cmd += [input_file, output_file]

        try:
            result = subprocess.run(
                cmd, check=True, capture_output=True, text=True
            )
        except FileNotFoundError:
            # gdalwarp not installed / not on PATH
            print("  ✗ gdalwarp not found on PATH; using original file as fallback.")
            aligned_files[key] = input_file
            continue
        except subprocess.CalledProcessError as e:
            print("  ✗ gdalwarp failed with non-zero exit code.")
            print("    Command:", " ".join(cmd))
            print("    stderr :", e.stderr.strip())
            aligned_files[key] = input_file
            continue

        # Verify the output file
        try:
            with rasterio.open(output_file) as src:
                if src.shape != ref_shape:
                    print(
                        f"  WARNING: shape mismatch after GDAL for {key}: "
                        f"{src.shape} vs reference {ref_shape}"
                    )
            aligned_files[key] = output_file
            print(f"  ✓ Success: {key} → {output_file}")
        except Exception as e:
            print(f"  ✗ Could not open aligned file for {key}: {e}")
            print("    Using original file as fallback.")
            aligned_files[key] = input_file

    return aligned_files




#  correct, density-preserving final alignment 
def _pixel_area(transform) -> float:
    """Return pixel area in square meters for an affine transform."""
    return abs(transform.a * transform.e)  # area = |scale_x * scale_y|



def force_align_with_reproject(aligned_files: Dict[str, str], reference_file: str) -> Tuple[dict, dict]:
    """
    Reproject all rasters to the reference grid.
    - Continuous fields (temperature) use bilinear.
    - Counts per pixel (population, baseline deaths) are preserved by
      resampling *densities* with average and then multiplying by target pixel area.
    Returns (template_profile, arrays).
    """
    ref_aligned = os.path.join(os.path.dirname(next(iter(aligned_files.values()))),
                               f"aligned_{os.path.basename(reference_file)}") 
    ref_path = ref_aligned if os.path.exists(ref_aligned) else reference_file   
    arrays = {}
    with rasterio.open(ref_path) as ref:
        # Use the reference raster's CRS and transform as the destination grid
        ref_profile = ref.profile.copy()
        ref_shape = ref.shape
        ref_transform = ref.transform
        ref_crs = ref.crs
        ref_area = _pixel_area(ref_transform)   
        for key, fpath in aligned_files.items():
            with rasterio.open(fpath) as src:
                src_crs = _coerce_to_ref_if_local(src.crs, ref_crs)
                dst = np.full(ref_shape, np.nan, dtype=np.float32)
                # detect “counts” layers by key name
                is_count = key in ("pop_baseline", "pop_scenario",
                                   "baseline_deaths_cardio", "baseline_deaths_resp", "baseline_deaths_self_harm")  
                if is_count:
                    # Convert counts -> densities (per m²), resample densities with average, then back to counts
                    area_src = _pixel_area(src.transform)
                    src_arr = src.read(1, masked=True).filled(np.nan).astype(np.float32)
                    with np.errstate(divide='ignore', invalid='ignore'):
                        src_density = np.where(np.isfinite(src_arr), src_arr / area_src, np.nan) 
                    reproject(
                        source=src_density,
                        destination=dst,
                        src_transform=src.transform, src_crs=src_crs, 
                        dst_transform=ref_transform, dst_crs=ref_crs,
                        dst_nodata=np.nan,
                        resampling=Resampling.average,
                    )
                    dst = dst * ref_area     # EDIT back to counts
                else:
                    # Continuous value (e.g., temperature): bilinear is appropriate
                    reproject(
                        source=rasterio.band(src, 1),
                        destination=dst,
                        src_transform=src.transform, src_crs=src_crs, 
                        dst_transform=ref_transform, dst_crs=ref_crs,
                        dst_nodata=np.nan,
                        resampling=Resampling.bilinear,
                    )
                arrays[key] = dst
                print(f"Aligned {key} → {ref_shape} (counts={is_count})")
    return ref_profile, arrays






def debug_shape_issues(paths):
    """Check shapes of all input files"""
    print("Checking array shapes:")
    for key, path in paths.items():
        try:
            with rasterio.open(path) as src:
                print(f"  {key}: {path}")
                print(f"    Shape: {src.shape}")
                print(f"    CRS: {src.crs}")
                print(f"    Bounds: {src.bounds}")
                print("---")
        except Exception as e:
            print(f"  {key}: ERROR - {e}")


def debug_gdal_output(aligned_dir):
    """Check what GDAL actually produced"""
    print("Debugging GDAL output files:")
    for file in os.listdir(aligned_dir):
        if file.endswith('.tif'):
            file_path = os.path.join(aligned_dir, file)
            try:
                with rasterio.open(file_path) as src:
                    print(f"  {file}:")
                    print(f"    Shape: {src.shape}")
                    print(f"    Bounds: {src.bounds}")
                    print(f"    Resolution: {src.res}")
                    print(f"    CRS: {src.crs}")
                    print("---")
            except Exception as e:
                print(f"  {file}: ERROR - {e}")


def _choose_reference(paths: Dict[str, str], align_to: Optional[str] = None) -> str:
    # <<< EDIT: choose reference grid (prefer CLI, else densest raster)
    if align_to is not None:
        print(f"Using user-provided reference for alignment: {align_to}")
        return align_to
    best_key, best_path, best_pixels = None, None, -1
    for k, p in paths.items():
        try:
            with rasterio.open(p) as src:
                pixels = src.width * src.height
                if pixels > best_pixels:
                    best_key, best_path, best_pixels = k, p, pixels
        except Exception as e:
            print(f"Warning: could not inspect {k}: {e}")
    print(f"Auto-selected reference: {best_key} -> {best_path} (pixels={best_pixels})")
    return best_path




# -----------------------------
# Epidemiology helpers
# -----------------------------

def beta_from_rr(rr_1c: float) -> float:
    """
    Calculate slope on log scale per +1 degC.
    
    Args:
        rr_1c: Relative risk per 1°C increase
        
    Returns:
        Beta coefficient (log scale)
    """
    return float(np.log(rr_1c))


def se_from_ci(rr_low: float, rr_high: float) -> float:
    """
    Calculate standard error of ln(RR) from 95% CI.
    
    Args:
        rr_low: Lower 95% CI bound
        rr_high: Upper 95% CI bound
        
    Returns:
        Standard error of log relative risk
    """
    return (np.log(rr_high) - np.log(rr_low)) / (2.0 * 1.96)


def rr_from_deltaT(beta: float, dT: np.ndarray) -> np.ndarray:
    """
    Calculate relative risk from temperature change.
    
    Args:
        beta: Beta coefficient from exposure-response function
        dT: Temperature change array (degC)
        
    Returns:
        Relative risk array
    """
    with np.errstate(over='ignore', invalid='ignore'):
        return np.exp(beta * dT)


def af_from_rr(rr: np.ndarray) -> np.ndarray:
    """
    Calculate attributable fraction from relative risk.
    
    Args:
        rr: Relative risk array
        
    Returns:
        Attributable fraction array
    """
    af = (rr - 1.0) / rr
    af[~np.isfinite(af)] = np.nan
    return af


def excess_deaths(af: np.ndarray, baseline_deaths: np.ndarray) -> np.ndarray:
    """
    Calculate excess deaths from attributable fraction and baseline deaths.
    
    Args:
        af: Attributable fraction array
        baseline_deaths: Baseline deaths array (known deaths/pixel/year)
        
    Returns:
        Excess deaths array
    """
    excess = af * baseline_deaths
    excess[~np.isfinite(excess)] = np.nan
    return excess


def adjusted_baseline_deaths(baseline_deaths: np.ndarray, 
                             pop_baseline: np.ndarray, 
                             pop_scenario: np.ndarray) -> np.ndarray:
    """
    Adjust baseline deaths for scenario population changes.
    
    Args:
        baseline_deaths: Baseline deaths (deaths/pixel/year)
        pop_baseline: Baseline population (persons/pixel)
        pop_scenario: Scenario population (persons/pixel)
        
    Returns:
        Adjusted baseline deaths for scenario population
    """

    mask = (
        ~np.isfinite(baseline_deaths) |
        ~np.isfinite(pop_baseline) |
        ~np.isfinite(pop_scenario) |
        (pop_baseline <= 0)
    )

    # Calculate death rate per person in baseline
    with np.errstate(divide='ignore', invalid='ignore'):
        death_rate = baseline_deaths / pop_baseline
    
    death_rate[mask] = np.nan

    # Apply same death rate to scenario population
    adjusted = death_rate * pop_scenario
    adjusted[mask | ~np.isfinite(adjusted)] = np.nan
    return adjusted



# -----------------------------
# Main pipeline
# -----------------------------

def main(args):
    """
    Main processing pipeline for heat mortality projection.
    
    Args:
        args: Command line arguments
    """
    # Create output directory if it doesn't exist
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Save configuration for reproducibility
    config_path = os.path.join(args.out_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)

    # 1) Align all required rasters
    path_map = {
        "t_baseline": args.t_baseline,
        "t_scenario": args.t_scenario,
        "pop_baseline": args.pop_baseline,
        "pop_scenario": args.pop_scenario,
        "baseline_deaths_cardio": args.baseline_deaths_cardio,
        "baseline_deaths_resp": args.baseline_deaths_resp,
        "baseline_deaths_self_harm": args.baseline_deaths_self_harm,
    }


    # # REMOVE upsample step entirely — it duplicates people at high res.---- 
    # population_files = upsample_population_data(path_map)
    # path_map.update(population_files)  # Replace with upsampled files


    # Step 1: Check original files
    print("=== Checking original files ===")
    debug_shape_issues(path_map)
    
    # Step 2: GDAL alignment
    print("\n=== GDAL alignment ===")
    aligned_dir = os.path.join(args.out_dir, "aligned_files")
    reference_file = _choose_reference(path_map, args.align_to)
    print(f"Using reference file: {reference_file}")
    aligned_files = robust_gdal_alignment(path_map, aligned_dir, reference_file)
    
    # Step 3: Check GDAL output
    print("\n=== Checking GDAL output ===")
    debug_gdal_output(aligned_dir)
    
    
    # Step 4: Read & Finalize alignment correctly (reproject everything to reference grid)
    print("\n=== Reading & finalizing alignment ===")
    template_profile, arrays = force_align_with_reproject(aligned_files, reference_file)  # TRUE reprojection

    # Step 5: Profile shape must match the finalized arrays (reference grid)
    template_profile.update({
        'height': arrays["t_baseline"].shape[0],
        'width':  arrays["t_baseline"].shape[1],
        'dtype': 'float32',
        # keep nodata unset here; _write_geotiff handles it safely
    })
    
      

    # Step 7: Proceed with analysis -------------------------------------------
    T_baseline = arrays["t_baseline"]
    T_scenario = arrays["t_scenario"]
    dT = T_scenario - T_baseline  # degC temperature change
    # quick sanity check
    print("deltaT stats:",
      np.nanmin(dT), np.nanmax(dT), np.nanmean(dT))
    # Write temperature change raster
    _write_geotiff(os.path.join(args.out_dir, "deltaT_degC.tif"), dT, template_profile)

    # Continue with your processing...
    print("\n\n Analysis proceeding with aligned arrays...")

    pop_baseline = arrays["pop_baseline"]
    pop_scenario = arrays["pop_scenario"]

    # 2) Deterministic estimates for each cause
    results = []
    for cause in DEFAULT_CAUSES:
        # Select appropriate baseline deaths raster
        if cause.name == "cardiovascular":
            baseline_deaths = arrays["baseline_deaths_cardio"]
        elif cause.name == "respiratory":
            baseline_deaths = arrays["baseline_deaths_resp"]
        elif cause.name == "self_harm":
            baseline_deaths = arrays["baseline_deaths_self_harm"]
        else:
            print(f"Warning: Unknown cause '{cause.name}', skipping")
            continue

        # Adjust baseline deaths for scenario population
        adjusted_deaths = adjusted_baseline_deaths(
            baseline_deaths, pop_baseline, pop_scenario
        )

        # Calculate health impacts
        beta = beta_from_rr(cause.rr_1c)
        rr = rr_from_deltaT(beta, dT)
        af = af_from_rr(rr)
        exc = excess_deaths(af, adjusted_deaths)

        # Save GeoTIFFs
        _write_geotiff(os.path.join(args.out_dir, f"AF_{cause.name}.tif"), af, template_profile)
        _write_geotiff(os.path.join(args.out_dir, f"Excess_{cause.name}.tif"), exc, template_profile)

        # Calculate city totals (ignore NaN)
        total_excess = float(np.nansum(exc))
        total_baseline = float(np.nansum(adjusted_deaths))

        results.append({
            "cause": cause.name,
            "RR_1C": cause.rr_1c,
            "city_total_baseline_deaths": total_baseline,
            "city_total_excess_deaths": total_excess,
            "fraction_of_total_deaths_excess": (total_excess / (total_baseline + total_excess)) if (total_baseline + total_excess) > 0 else 0
        })

    # Save deterministic results
    pd.DataFrame(results).to_csv(os.path.join(args.out_dir, "city_totals_deterministic.csv"), index=False)
    print("✓ Analysis completed successfully!")


    # -----------------------------------------------------------------
    # 3) Optional Monte Carlo uncertainty analysis
    # -----------------------------------------------------------------
    if args.n_draws and args.n_draws > 0:
        print(f"\n\n Running Monte Carlo simulation with {args.n_draws} draws...")
        rng = np.random.default_rng(args.seed)
        
        # summary rows per cause
        mc_rows = []
        
        # container to store all draws per cause 
        cause_draws = {}  #  {cause_name: np.ndarray of shape (n_draws,)}

        for cause in DEFAULT_CAUSES:
            # Select appropriate baseline deaths raster
            if cause.name == "cardiovascular":
                baseline_deaths = arrays["baseline_deaths_cardio"]
            elif cause.name == "respiratory":
                baseline_deaths = arrays["baseline_deaths_resp"]
            elif cause.name == "self_harm":
                baseline_deaths = arrays["baseline_deaths_self_harm"]
            else:
                print(f"Warning: Unknown cause '{cause.name}', skipping Monte Carlo")
                continue

            # Adjust baseline deaths for scenario population
            adjusted_deaths = adjusted_baseline_deaths(
                baseline_deaths, pop_baseline, pop_scenario
            )

            beta_hat = beta_from_rr(cause.rr_1c)
            se = se_from_ci(cause.rr_low_1c, cause.rr_high_1c)

            # Generate random beta values from normal distribution
            betas = rng.normal(loc=beta_hat, scale=se, size=args.n_draws)

            # Calculate city total excess deaths for each draw (scalar per draw)
            totals = np.empty(args.n_draws, dtype=np.float64)  # preallocate
            for j, b in enumerate(betas):
                rr = rr_from_deltaT(b, dT)
                af = af_from_rr(rr)
                exc = excess_deaths(af, adjusted_deaths)
                totals[j] = float(np.nansum(exc))


            # 1/2. Stash full draw vector
            cause_draws[cause.name] = totals 


            # 2/2. Calculate summary statistics
            totals = np.array(totals, dtype=float)
            low, med, high = np.percentile(totals, [2.5, 50, 97.5])
            mc_rows.append({
                "cause": cause.name,
                "n_draws": args.n_draws,
                "seed": args.seed,
                "excess_p2p5": float(low),
                "excess_p50": float(med),
                "excess_p97p5": float(high),
                "excess_mean": float(np.mean(totals)),
                "excess_std": float(np.std(totals)),
                "excess_min": float(np.min(totals)),
                "excess_max": float(np.max(totals)),
            })
            
            print(f"  {cause.name}: {np.mean(totals):.1f} ± {np.std(totals):.1f} excess deaths")


        # ------------------------------------------------------
        # Save outputs
        # ------------------------------------------------------
        # 1/2. Save Monte Carlo results
        mc_df = pd.DataFrame(mc_rows)
        mc_output_path = os.path.join(args.out_dir, "city_totals_monte_carlo.csv")
        mc_df.to_csv(mc_output_path, index=False)
        print(f"Monte Carlo results saved to: {mc_output_path}")

        # 2/2. save full draws matrix (n_draws x causes)
        # Columns are per cause; add a 1-based draw index column for readability.
        draws_df = pd.DataFrame({name: vals for name, vals in cause_draws.items()}) 
        draws_df.insert(0, "draw", np.arange(1, args.n_draws + 1))
        draws_path = os.path.join(args.out_dir, "city_total_draws_by_cause.csv") 
        draws_df.to_csv(draws_path, index=False) 
        print(f"Full draw matrix saved to: {draws_path}")


        
        # Also save full distribution for advanced analysis
        if args.n_draws <= 1000:  # Only save if not too large
            flat_rows = []
            for cause_name, totals in cause_draws.items():
                for draw_id, total in enumerate(totals, start=1):
                    flat_rows.append({
                        "cause": cause_name,
                        "draw": draw_id,
                        "excess": float(total),
                    })
            pd.DataFrame(flat_rows).to_csv(
                os.path.join(args.out_dir, "monte_carlo_distributions.csv"),
                index=False
            )
        print("Monte Carlo simulation completed.")


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Project heat-attributable excess deaths from spatial temperature change"
    )
    # Required arguments
    parser.add_argument("--t_baseline", required=True, help="Baseline temperature raster (degC)")
    parser.add_argument("--t_scenario", required=True, help="Scenario temperature raster (degC)")
    parser.add_argument("--pop_baseline", required=True, help="Baseline population raster (persons/pixel)")
    parser.add_argument("--pop_scenario", required=True, help="Scenario population raster (persons/pixel)")
    parser.add_argument("--baseline_deaths_cardio", required=True, help="Baseline deaths raster, cardiovascular (deaths/pixel/year)")
    parser.add_argument("--baseline_deaths_resp", required=True, help="Baseline deaths raster, respiratory (deaths/pixel/year)")
    parser.add_argument("--baseline_deaths_self_harm", required=True, help="Baseline deaths raster, self_harm (deaths/pixel/year)")
    parser.add_argument("--out_dir", required=True, help="Output directory")
    # Optional arguments
    parser.add_argument("--align_to", default=None, help="Optional path to raster used as alignment template")
    parser.add_argument("--n_draws", type=int, default=0, help="Monte Carlo draws for uncertainty (0 to skip)")
    parser.add_argument("--seed", type=int, default=2025, help="Random seed for Monte Carlo")
    parser.add_argument("--causes", default=None, help="JSON file with custom cause configurations (overrides defaults)")
    args = parser.parse_args()

    # Load custom causes if provided
    if args.causes:
        try:
            with open(args.causes, 'r') as f:
                custom_causes = json.load(f)
            DEFAULT_CAUSES = [CauseConfig(**c) for c in custom_causes]  # type: ignore
            print(f"Loaded {len(DEFAULT_CAUSES)} custom causes from {args.causes}")
        except Exception as e:
            print(f"Error loading custom causes: {e}. Using defaults.")

    # Run main pipeline
    main(args)