"""
lulc_stats.py
Calculates area (km²) and % land cover per LULC class from a raster.
Saves results to a formatted .xlsx file.
"""

import os
import sys
import sqlite3
import subprocess
import shutil
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter

# ============================================================
# STEP 1: FIX PROJ BEFORE RASTERIO IMPORT
# ============================================================

def _configure_proj():
    candidates = [
        Path(sys.prefix) / "Library" / "share" / "proj",
        Path(sys.prefix) / "share" / "proj",
    ]
    for proj_dir in candidates:
        db = proj_dir / "proj.db"
        if not db.exists():
            continue
        try:
            con = sqlite3.connect(str(db))
            minor = int(con.execute(
                "SELECT value FROM metadata WHERE key='DATABASE.LAYOUT.VERSION.MINOR'"
            ).fetchone()[0])
            con.close()
            if minor < 5:
                continue
        except Exception:
            continue
        os.environ["PROJ_DATA"]    = str(proj_dir)
        os.environ["PROJ_LIB"]     = str(proj_dir)
        os.environ["PROJ_NETWORK"] = "OFF"
        print(f"[proj] PROJ_DATA → {proj_dir} (v{minor} ✓)")
        return True
    print("[proj] WARNING: valid proj.db not found")
    return False

_configure_proj()

import rasterio  # safe after PROJ configured

# ============================================================
# LULC CLASS LOOKUP — LCM2023
# ============================================================

LCM2023_CLASSES = {
    1:   "Broadleaved woodland",
    2:   "Coniferous Woodland",
    3:   "Arable and Horticulture",
    4:   "Improved Grassland",
    5:   "Neutral Grassland",
    6:   "Calcareous Grassland",
    7:   "Acid Grassland",
    8:   "Fen, Marsh and Swamp",
    9:   "Heather",
    10:  "Heather Grassland",
    11:  "Bog",
    12:  "Inland Rock",
    13:  "Saltwater",
    14:  "Freshwater",
    15:  "Supralittoral Rock",
    16:  "Supralittoral Sediment",
    17:  "Littoral Rock",
    18:  "Littoral Sediment",
    19:  "Saltmarsh",
    20:  "Urban",
    21:  "Suburban",
    100: "Tree Canopy",
}

# ============================================================
# CORE FUNCTION
# ============================================================

def calculate_lulc_stats(raster_path: str) -> pd.DataFrame:
    """
    Reads a LULC raster and calculates area (km²) and % per class.

    Args:
        raster_path (str): Path to the LULC .tif file.

    Returns:
        DataFrame with columns: Value, Label, Pixel_Count, Area_km2, Area_pct
    """
    with rasterio.open(raster_path) as src:
        data      = src.read(1)
        nodata    = src.nodata
        transform = src.transform
        crs       = src.crs

        # Pixel area in m² (assumes projected CRS in metres)
        pixel_width  = abs(transform.a)
        pixel_height = abs(transform.e)
        pixel_area_m2 = pixel_width * pixel_height

        print(f"Raster CRS        : {crs}")
        print(f"Pixel size        : {pixel_width:.1f} x {pixel_height:.1f} m")
        print(f"Pixel area        : {pixel_area_m2:.1f} m²")
        print(f"Raster shape      : {src.shape}")
        print(f"NoData value      : {nodata}")

    # Mask nodata
    if nodata is not None:
        valid_mask = data != nodata
    else:
        valid_mask = np.ones(data.shape, dtype=bool)

    valid_pixels = data[valid_mask]
    total_valid_pixels = len(valid_pixels)
    total_area_km2 = (total_valid_pixels * pixel_area_m2) / 1e6

    print(f"Total valid pixels: {total_valid_pixels:,}")
    print(f"Total valid area  : {total_area_km2:,.4f} km²\n")

    # Count pixels per class
    pixel_counts = Counter(valid_pixels.tolist())

    # Build DataFrame
    rows = []
    for value, label in LCM2023_CLASSES.items():
        count = pixel_counts.get(value, 0)
        area_km2 = (count * pixel_area_m2) / 1e6
        area_pct = (count / total_valid_pixels * 100) if total_valid_pixels > 0 else 0
        rows.append({
            "Value":       value,
            "Label":       label,
            "Pixel_Count": count,
            "Area_km2":    round(area_km2, 4),
            "Area_pct":    round(area_pct, 4),
        })

    # Include any unrecognised classes present in data
    known_values = set(LCM2023_CLASSES.keys())
    for value, count in pixel_counts.items():
        if value not in known_values:
            area_km2 = (count * pixel_area_m2) / 1e6
            area_pct = (count / total_valid_pixels * 100)
            rows.append({
                "Value":       value,
                "Label":       f"Unknown (class {value})",
                "Pixel_Count": count,
                "Area_km2":    round(area_km2, 4),
                "Area_pct":    round(area_pct, 4),
            })

    df = (pd.DataFrame(rows)
          .sort_values("Value")
          .reset_index(drop=True))

    return df, total_area_km2


# ============================================================
# XLSX EXPORT
# ============================================================
def save_stats_csv(df: pd.DataFrame, output_path: str,
                   raster_name: str, total_area_km2: float):
    """Saves LULC stats to CSV with a totals row."""

    # Add totals row
    totals = pd.DataFrame([{
        "Value":       "TOTAL",
        "Label":       "All Classes",
        "Pixel_Count": df["Pixel_Count"].sum(),
        "Area_km2":    round(df["Area_km2"].sum(), 4),
        "Area_pct":    round(df["Area_pct"].sum(), 4),
    }])

    df_out = pd.concat([df, totals], ignore_index=True)

    # Write metadata header lines then data
    with open(output_path, "w") as f:
        f.write(f"# Source: {raster_name}\n")
        f.write(f"# Total valid area: {total_area_km2:,.4f} km2\n")

    df_out.to_csv(output_path, index=False, mode="a")
    print(f"✓ Saved: {output_path}")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    # --- SINGLE RASTER ---
    # RASTER_PATH = r"G:\Shared drives\Wellcome Trust Project Data\1_preprocess\UrbanCoolingModel\OfficialWorkingInputs\LULC\LCM2023_London_10m_clip2aoi_tcc24.tif"
    # OUTPUT_DIR  = Path(RASTER_PATH).parent / "lc_stats_output"
    # RASTER_PATHS = [RASTER_PATH]  # swap with list above for batch mode
    

    # --- OR: BATCH MULTIPLE RASTERS (uncomment to use) ---
    LULC_PATH = r"G:\Shared drives\Wellcome Trust Project Data\1_preprocess\UrbanCoolingModel\OfficialWorkingInputs\LULC"
    RASTER_PATHS = [
        # Path(LULC_PATH, "LCM2023_London_10m_clip2aoi_tcc24.tif"),
        # Path(LULC_PATH, "LCM2023_London_10m_clip2aoi_tcc24_scenario4_nearest_to_edge_10prc_canopy_increase.tif"),
        # Path(LULC_PATH, "LCM2023_London_10m_clip2aoi_tcc24_scenario4_nearest_to_edge_20prc_canopy_increase.tif"),
        # Path(LULC_PATH, "LCM2023_London_10m_clip2aoi_tcc24_scenario4_nearest_to_edge_30prc_canopy_increase.tif"),
        # Path(LULC_PATH, "lc_tree_equity_scenarios_output", "LULC_Scenario710.tif"),
        # Path(LULC_PATH, "lc_tree_equity_scenarios_output", "LULC_Scenario715.tif"),
        # Path(LULC_PATH, "lc_tree_equity_scenarios_output", "LULC_Scenario720.tif"),
        # Path(LULC_PATH, "lc_tree_equity_scenarios_output", "LULC_Scenario730.tif"),

        # Path(LULC_PATH, "lc_tree_equity_scenarios_output", "LULC_Scenario710v2.tif"),
        # Path(LULC_PATH, "lc_tree_equity_scenarios_output", "LULC_Scenario720v2.tif"),
        # Path(LULC_PATH, "lc_tree_equity_scenarios_output", "LULC_Scenario730v2.tif"),

        # Path(LULC_PATH, "lc_tree_equity_scenarios_output", "LULC_Scenario710v3.tif"),
        # Path(LULC_PATH, "lc_tree_equity_scenarios_output", "LULC_Scenario720v3.tif"),
        # Path(LULC_PATH, "lc_tree_equity_scenarios_output", "LULC_Scenario730v3.tif"),

        Path(LULC_PATH, "lc_tree_equity_scenarios_output", "LULC_Scenario510.tif"),
        Path(LULC_PATH, "lc_tree_equity_scenarios_output", "LULC_Scenario520.tif"),
        Path(LULC_PATH, "lc_tree_equity_scenarios_output", "LULC_Scenario530.tif"),
        
    ]
    OUTPUT_DIR  = Path(LULC_PATH, "lc_tree_equity_stats_output")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Process each raster
    for raster_path in RASTER_PATHS:
        raster_path = Path(raster_path)

        if not raster_path.exists():
            print(f"ERROR: File not found — {raster_path}")
            continue

        print(f"\n====== Processing: {raster_path.name} ======")

        df, total_area_km2 = calculate_lulc_stats(str(raster_path))

        print(df[["Value", "Label", "Area_km2", "Area_pct"]].to_string(index=False))

        out_csv = OUTPUT_DIR / f"{raster_path.stem}_lulc_stats.csv"
        save_stats_csv(
            df=df,
            output_path=str(out_csv),
            raster_name=raster_path.name,
            total_area_km2=total_area_km2
        )
    print("\nDone.")