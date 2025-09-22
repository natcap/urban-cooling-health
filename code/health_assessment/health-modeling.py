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
        --baseline_deaths_cere path/to/baseline_deaths_cere.tif \
        --out_dir ./outputs \
        --n_draws 1000

NOTES:
 - All rasters should be in the SAME CRS and cover the same urban extent.
 - If rasters are at different resolutions/CRS, set --align_to to one of the inputs;
   the script will reproject/align the others to match.
 - Mortality rates must be deaths per person per YEAR.
 - Population is persons per pixel.
 - For city totals, the script sums per-pixel values, ignoring NaNs.
 - Causes: cardiovascular, respiratory, cerebrovascular (customizable).

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
from pandas.core import arrays
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject

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
    CauseConfig("cerebrovascular",1.0140, 1.0006, 1.0275),
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
        nodata=nodata
    )
    if isinstance(nodata, float) and math.isnan(nodata):
        prof.pop("nodata", None)
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(array.astype(np.float32), 1)


def _align_to_template(src_path: str, template_src) -> np.ndarray:
    """
    Reproject/align a raster to match the template dataset's grid/CRS.
    
    Args:
        src_path: Path to source raster
        template_src: Rasterio dataset to use as alignment template
        
    Returns:
        Aligned numpy array
    """
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



def _ensure_aligned(paths: Dict[str, str], align_to: Optional[str] = None) -> Tuple:
    """
    Open rasters and align all to a template grid with shape validation.
    """
    if align_to is None:
        first_key = next(iter(paths.keys()))
        template_path = paths[first_key]
    else:
        template_path = align_to

    with rasterio.open(template_path) as template_src:
        template_profile = template_src.profile.copy()
        arrays = {}
        
        print(f"Template shape: {template_src.shape}")
        
        for key, p in paths.items():
            print(f"Processing: {key}")
            
            if p == template_path:
                arrays[key] = template_src.read(1, masked=True).filled(np.nan).astype(np.float32)
                print(f"  Shape: {arrays[key].shape}")
            else:
                try:
                    arrays[key] = _align_to_template(p, template_src)
                    print(f"  Shape after alignment: {arrays[key].shape}")
                    
                    # Verify shape matches template
                    if arrays[key].shape != template_src.shape:
                        print(f"  WARNING: Shape mismatch! Expected {template_src.shape}, got {arrays[key].shape}")
                        # Force resize to template shape
                        arrays[key] = _force_resize(arrays[key], template_src.shape)
                        
                except Exception as e:
                    print(f"  ERROR aligning {key}: {e}")
                    # Fallback: read directly and hope for the best
                    with rasterio.open(p) as src:
                        arrays[key] = src.read(1, masked=True).filled(np.nan).astype(np.float32)
                        print(f"  Fallback shape: {arrays[key].shape}")

    return template_profile, arrays


def _force_resize(array, target_shape):
    """Force resize an array to target shape using padding/cropping"""
    result = np.full(target_shape, np.nan, dtype=array.dtype)
    
    # Copy the overlapping region
    min_rows = min(array.shape[0], target_shape[0])
    min_cols = min(array.shape[1], target_shape[1])
    
    result[:min_rows, :min_cols] = array[:min_rows, :min_cols]
    return result




def preprocess_with_gdal(input_files, output_dir, reference_file):
    """Pre-align all files using GDAL before main processing"""
    import subprocess
    import os
    
    os.makedirs(output_dir, exist_ok=True)
    aligned_files = {}
    
    # Get reference file properties
    with rasterio.open(reference_file) as ref:
        ref_bounds = ref.bounds
        ref_res = ref.res
    
    for key, input_file in input_files.items():
        output_file = os.path.join(output_dir, f"aligned_{os.path.basename(input_file)}")
        
        cmd = [
            'gdalwarp', 
            '-te', str(ref_bounds.left), str(ref_bounds.bottom), 
                   str(ref_bounds.right), str(ref_bounds.top),
            '-tr', str(ref_res[0]), str(ref_res[1]),
            '-t_srs', 'EPSG:27700',  # Force British National Grid
            '-r', 'bilinear',        # Bilinear resampling for continuous data
            '-overwrite',
            input_file, 
            output_file
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            aligned_files[key] = output_file
            print(f"✓ Successfully aligned: {key}")
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to align {key}: {e.stderr.decode()}")
            # Fallback: use original file
            aligned_files[key] = input_file
    
    return aligned_files

def read_aligned_arrays(aligned_files):
    """Simply read pre-aligned files without additional processing"""
    arrays = {}
    template_profile = None
    
    for key, file_path in aligned_files.items():
        with rasterio.open(file_path) as src:
            arrays[key] = src.read(1, masked=True).filled(np.nan).astype(np.float32)
            if template_profile is None:
                template_profile = src.profile.copy()
    
    return template_profile, arrays


def robust_gdal_alignment(input_files, output_dir, reference_file):
    """GDAL alignment with thorough verification"""
    import subprocess
    
    os.makedirs(output_dir, exist_ok=True)
    aligned_files = {}
    
    # Get precise reference properties
    with rasterio.open(reference_file) as ref:
        ref_bounds = ref.bounds
        ref_res = ref.res
        ref_crs = ref.crs.to_string() if ref.crs else 'EPSG:27700'
    
    print(f"Reference: bounds={ref_bounds}, res={ref_res}, crs={ref_crs}")
    
    for key, input_file in input_files.items():
        output_file = os.path.join(output_dir, f"aligned_{os.path.basename(input_file)}")
        
        cmd = [
            'gdalwarp', 
            '-te', str(ref_bounds.left), str(ref_bounds.bottom), 
                   str(ref_bounds.right), str(ref_bounds.top),
            '-tr', str(ref_res[0]), str(ref_res[1]),
            '-t_srs', ref_crs,
            '-r', 'bilinear',
            '-overwrite',
            '-dstnodata', '-9999',
            input_file, 
            output_file
        ]
        
        try:
            print(f"Aligning {key}...")
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            
            # Verify the output file
            with rasterio.open(output_file) as src:
                if src.shape != (ref.height, ref.width):
                    print(f"WARNING: {key} shape mismatch after GDAL: {src.shape} vs {ref.shape}")
                
            aligned_files[key] = output_file
            print(f"✓ Success: {key}")
            
        except Exception as e:
            print(f"✗ Failed {key}: {e}")
            print(f"Using original file as fallback")
            aligned_files[key] = input_file
    
    return aligned_files


def force_align_arrays(arrays, reference_key="pop_baseline"):
    """Force all arrays to match the reference array shape"""
    if reference_key not in arrays:
        print(f"Reference key {reference_key} not found, using first array")
        reference_key = list(arrays.keys())[0]
    
    reference_shape = arrays[reference_key].shape
    print(f"Reference shape: {reference_shape} from {reference_key}")
    
    aligned_arrays = {}
    
    for key, array in arrays.items():
        if array.shape == reference_shape:
            aligned_arrays[key] = array
            print(f"✓ {key}: already matches reference shape")
        else:
            print(f"⚠ {key}: {array.shape} -> resizing to {reference_shape}")
            aligned_arrays[key] = resize_array(array, reference_shape)
    
    return aligned_arrays

def resize_array(array, target_shape, fill_value=np.nan):
    """Resize array to target shape with padding/cropping"""
    result = np.full(target_shape, fill_value, dtype=array.dtype)
    
    # Copy overlapping region
    rows = min(array.shape[0], target_shape[0])
    cols = min(array.shape[1], target_shape[1])
    
    result[:rows, :cols] = array[:rows, :cols]
    return result


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
    # Calculate death rate per person in baseline
    death_rate = baseline_deaths / pop_baseline
    death_rate[~np.isfinite(death_rate)] = 0
    
    # Apply same death rate to scenario population
    adjusted_deaths = death_rate * pop_scenario
    return adjusted_deaths



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
        "baseline_deaths_cere": args.baseline_deaths_cere,
    }


    # Step 1: Pre-align with Python and rasterio - not always reliable
    # template_src, arrays = _ensure_aligned(path_map, align_to=args.align_to)

    # Step 1: Check original files
    print("=== Checking original files ===")
    debug_shape_issues(path_map)
    
    # Step 2: GDAL alignment
    print("\n=== GDAL alignment ===")
    aligned_dir = os.path.join(args.out_dir, "aligned_files")
    aligned_files = robust_gdal_alignment(path_map, aligned_dir, args.pop_baseline)
    
    # Step 3: Check GDAL output
    print("\n=== Checking GDAL output ===")
    debug_gdal_output(aligned_dir)
    
    # Step 4: Read files
    print("\n=== Reading files ===")
    template_profile, arrays = read_aligned_arrays(aligned_files)
    
    # Step 5: Force alignment if still needed
    print("\n=== Final alignment check ===")
    shapes_before = {k: v.shape for k, v in arrays.items()}
    print("Shapes before force alignment:", shapes_before)
    
    if len(set(shapes_before.values())) > 1:
        print("WARNING: Shapes still don't match, forcing alignment...")
        arrays = force_align_arrays(arrays, "pop_baseline")
        
        shapes_after = {k: v.shape for k, v in arrays.items()}
        print("Shapes after force alignment:", shapes_after)
        
        if len(set(shapes_after.values())) > 1:
            print("CRITICAL ERROR: Could not align arrays!")
            return
        else:
            print("✓ Successfully force-aligned all arrays")
    else:
        print("✓ All arrays already aligned")
    
    # Step 6: Update template profile to match actual array size
    template_profile.update({
        'height': arrays["pop_baseline"].shape[0],
        'width': arrays["pop_baseline"].shape[1],
        'dtype': 'float32',
        'nodata': np.nan
    })
    

    # Step 7: Proceed with analysis -------------------------------------------
    T_baseline = arrays["t_baseline"]
    T_scenario = arrays["t_scenario"]
    dT = T_scenario - T_baseline  # degC temperature change
    # Write temperature change raster
    _write_geotiff(os.path.join(args.out_dir, "deltaT_degC.tif"), dT, template_profile)

    # Continue with your processing...
    print("Analysis proceeding with aligned arrays...")

    pop_baseline = arrays["pop_baseline"]
    pop_scenario = arrays["pop_scenario"]
    baseline_deaths_cardio = arrays["baseline_deaths_cardio"]
    baseline_deaths_resp   = arrays["baseline_deaths_resp"]  
    baseline_deaths_cere   = arrays["baseline_deaths_cere"]

    # 2) Deterministic estimates for each cause
    results = []
    for cause in DEFAULT_CAUSES:
        # Select appropriate baseline deaths raster
        if cause.name == "cardiovascular":
            baseline_deaths = arrays["baseline_deaths_cardio"]
        elif cause.name == "respiratory":
            baseline_deaths = arrays["baseline_deaths_resp"]
        elif cause.name == "cerebrovascular":
            baseline_deaths = arrays["baseline_deaths_cere"]
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
            "attributable_fraction": total_excess / (total_baseline + total_excess) if (total_baseline + total_excess) > 0 else 0
        })

    # Save deterministic results
    pd.DataFrame(results).to_csv(os.path.join(args.out_dir, "city_totals_deterministic.csv"), index=False)
    print("✓ Analysis completed successfully!")

    # 3) Optional Monte Carlo uncertainty analysis
    if args.n_draws and args.n_draws > 0:
        print(f"\n\n Running Monte Carlo simulation with {args.n_draws} draws...")
        rng = np.random.default_rng(args.seed)
        mc_rows = []
        
        for cause in DEFAULT_CAUSES:
            # Select appropriate baseline deaths raster
            if cause.name == "cardiovascular":
                baseline_deaths = arrays["baseline_deaths_cardio"]
            elif cause.name == "respiratory":
                baseline_deaths = arrays["baseline_deaths_resp"]
            elif cause.name == "cerebrovascular":
                baseline_deaths = arrays["baseline_deaths_cere"]
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

            # Calculate excess deaths for each beta
            totals = []
            for b in betas:
                rr = rr_from_deltaT(b, dT)
                af = af_from_rr(rr)
                exc = excess_deaths(af, adjusted_deaths)  # Use adjusted_deaths instead of mort_rate
                totals.append(float(np.nansum(exc)))

            # Calculate summary statistics
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

        # Save Monte Carlo results
        mc_df = pd.DataFrame(mc_rows)
        mc_output_path = os.path.join(args.out_dir, "city_totals_monte_carlo.csv")
        mc_df.to_csv(mc_output_path, index=False)
        print(f"Monte Carlo results saved to: {mc_output_path}")
        
        # Also save full distribution for advanced analysis
        if args.n_draws <= 1000:  # Only save if not too large
            distribution_path = os.path.join(args.out_dir, "monte_carlo_distributions.csv")
            distribution_data = []
            for i, cause in enumerate(DEFAULT_CAUSES):
                if i < len(mc_rows):
                    for j, total in enumerate(totals if i == 0 else []):
                        # This would need to be adjusted to store all draws for all causes
                        pass
            # For simplicity, we're just saving summary statistics
        print("Monte Carlo simulation completed.")


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Project heat-attributable excess deaths from spatial temperature change"
    )
    
    # Required arguments
    parser.add_argument(
        "--t_baseline", 
        required=True, 
        help="Baseline temperature raster (degC)"
    )
    parser.add_argument(
        "--t_scenario", 
        required=True, 
        help="Scenario temperature raster (degC)"
    )
    
    
    parser.add_argument(
        "--pop_baseline", 
        required=True, 
        help="Baseline population raster (persons/pixel)"
    )
    parser.add_argument(
        "--pop_scenario", 
        required=True, 
        help="Scenario population raster (persons/pixel)"
    )
    parser.add_argument(
        "--baseline_deaths_cardio", 
        required=True, 
        help="Baseline deaths raster, cardiovascular (deaths/pixel/year)"
    )
    parser.add_argument(
        "--baseline_deaths_resp", 
        required=True, 
        help="Baseline deaths raster, respiratory (deaths/pixel/year)"
    )
    parser.add_argument(
        "--baseline_deaths_cere", 
        required=True, 
        help="Baseline deaths raster, cerebrovascular (deaths/pixel/year)"
    )

    parser.add_argument(
        "--out_dir", 
        required=True, 
        help="Output directory"
    )
    
    # Optional arguments
    parser.add_argument(
        "--align_to", 
        default=None, 
        help="Optional path to raster used as alignment template"
    )
    parser.add_argument(
        "--n_draws", 
        type=int, 
        default=0, 
        help="Monte Carlo draws for uncertainty (0 to skip)"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=2025, 
        help="Random seed for Monte Carlo"
    )
    
    # Additional optional arguments for flexibility
    parser.add_argument(
        "--causes", 
        default=None,
        help="JSON file with custom cause configurations (overrides defaults)"
    )
    
    args = parser.parse_args()
    
    # Load custom causes if provided
    if args.causes:
        try:
            with open(args.causes, 'r') as f:
                custom_causes = json.load(f)
            DEFAULT_CAUSES = [CauseConfig(**c) for c in custom_causes]
            print(f"Loaded {len(DEFAULT_CAUSES)} custom causes from {args.causes}")
        except Exception as e:
            print(f"Error loading custom causes: {e}. Using defaults.")
    
    # Run main pipeline
    main(args)