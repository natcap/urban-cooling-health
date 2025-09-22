import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.errors import RasterioError

def _pixel_area(transform):
    return abs(transform.a * transform.e)  # m² per pixel (north-up)

def _looks_like_bng(crs_obj) -> bool:
    if crs_obj is None:
        return False
    s = crs_obj.to_string()
    return (
        "OSGB36" in s
        or "Airy 1830" in s
        or "Transverse" in s
        or "Longitude of natural origin" in s
        or "False easting" in s
        or "False northing" in s
    )

def _coerce_to_ref_if_local(src_crs, ref_crs):
    """If src has LOCAL/unknown-but-BNG, assume the reference CRS."""
    if ref_crs is None:
        return src_crs
    if src_crs is None:
        return ref_crs
    try:
        # If exactly same, keep as-is
        if src_crs == ref_crs or src_crs.to_wkt() == ref_crs.to_wkt():
            return src_crs
    except Exception:
        pass
    return ref_crs if _looks_like_bng(src_crs) else src_crs

def resample_counts_areal(src_counts_path, dst_ref_path, out_path):
    """
    Disaggregate counts (e.g., pop per 100m pixel) to finer grid (e.g., 10m)
    by resampling densities with area-weighted average. Preserves totals.
    Robust to 'LOCAL/unknown' BNG CRSs by coercing to reference CRS.
    """
    with rasterio.open(src_counts_path) as src, rasterio.open(dst_ref_path) as ref:
        src_arr = src.read(1, masked=True).filled(np.nan).astype(np.float32)
        dst_density = np.full((ref.height, ref.width), np.nan, dtype=np.float32)

        A_src = _pixel_area(src.transform)
        A_dst = _pixel_area(ref.transform)

        with np.errstate(divide='ignore', invalid='ignore'):
            src_density = np.where(np.isfinite(src_arr), src_arr / A_src, np.nan).astype(np.float32)

        src_crs_use = _coerce_to_ref_if_local(src.crs, ref.crs)

        try:
            reproject(
                source=src_density,
                destination=dst_density,
                src_transform=src.transform, src_crs=src_crs_use,
                dst_transform=ref.transform, dst_crs=ref.crs,
                dst_nodata=np.nan,
                resampling=Resampling.average
            )
        except RasterioError as e:
            # Last-resort retry: force both CRSs to the reference CRS
            print(f"  [warn] reproject error ({e}). Retrying with src_crs=dst_crs=ref.crs …")
            reproject(
                source=src_density,
                destination=dst_density,
                src_transform=src.transform, src_crs=ref.crs,
                dst_transform=ref.transform, dst_crs=ref.crs,
                dst_nodata=np.nan,
                resampling=Resampling.average
            )

        dst_counts = dst_density * A_dst

        profile = ref.profile.copy()
        profile.update(dtype=rasterio.float32, count=1)
        profile.pop("nodata", None)  # keep NaNs in data, not tag
        with rasterio.open(out_path, "w", **profile) as dst_ds:
            dst_ds.write(dst_counts.astype(np.float32), 1)

        # QA: totals should match closely
        src_total = float(np.nansum(src_arr))
        dst_total = float(np.nansum(dst_counts))
        print(f"  [areal] \n total src={src_total:.3f}, \n dst={dst_total:.3f}, \n Δ={dst_total - src_total:.3f}")




# -------------------------------------------------------------------------------
# application 
# -------------------------------------------------------------------------------
import os
import glob


dir_main = r"G:/Shared drives/Wellcome Trust Project Data" 
dir_pop = os.path.join(dir_main, "1_preprocess/UrbanCoolingModel/OfficialWorkingInputs/pop_raster" )

# Reference 10m raster (defines target resolution, extent, CRS)
reference_file = os.path.join(dir_main, "2_postprocess_intermediate/UCM_official_runs/current_lulc/current_climate/intermediate/T_air_london_current_scenario_20deg_2uhi.tif")


# Output folder for resampled rasters
out_dir = os.path.join(dir_pop, "resampled_10m")
os.makedirs(out_dir, exist_ok=True)

# Areal-weighted:
# Loop over all .tif files in the folder
for src_path in glob.glob(os.path.join(dir_pop, "gbr_pop_*100m*.tif")):
    fname = os.path.basename(src_path)
    out_path = os.path.join(out_dir, fname.replace("_CN_100m_R2025A_v1_EPSG27700_clip2aoi.tif", "_10m_areal.tif"))
    print(f"Resampling {fname} -> {out_path}")

    try:
        resample_counts_areal(src_counts_path=src_path, dst_ref_path=reference_file, out_path=out_path)
        print("  ✓ done")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
