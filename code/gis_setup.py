# gis_setup.py
import os
import sys
from pathlib import Path
import pyproj
import rasterio
from rasterio.crs import CRS

def configure_proj_environment():
    """
    Automatically finds proj.db and sets the PROJ_LIB environment variable.
    Must be run BEFORE importing geopandas or running rasterio operations.
    """
    try:
        # Ask pyproj where it keeps the database
        proj_dir = pyproj.datadir.get_data_dir()
        proj_db = os.path.join(proj_dir, 'proj.db')

        if os.path.exists(proj_db):
            os.environ['PROJ_LIB'] = proj_dir
            # print(f"--- GIS SETUP: PROJ_LIB set to {proj_dir} ---")
            return True
        else:
            print("--- GIS SETUP WARNING: proj.db not found in pyproj directory. ---")
            return False
            
    except Exception as e:
        print(f"--- GIS SETUP ERROR: Could not configure PROJ environment: {e} ---")
        return False

def ensure_raster_crs(raster_path: str, target_epsg: int, overwrite: bool = True) -> str:
    import subprocess
    import shutil
    import rasterio

    raster_path = Path(raster_path)
    if not raster_path.exists():
        raise FileNotFoundError(f"Raster not found: {raster_path}")

    # --- Find gdalwarp — check PATH first, then common conda locations ---
    gdalwarp = shutil.which("gdalwarp")
    if gdalwarp is None:
        # Common conda-forge location on Windows
        candidates = [
            Path(sys.prefix) / "Library" / "bin" / "gdalwarp.exe",
            Path(sys.prefix) / "bin" / "gdalwarp",
        ]
        for c in candidates:
            if c.exists():
                gdalwarp = str(c)
                break

    if gdalwarp is None:
        raise FileNotFoundError(
            "gdalwarp not found. Run: conda install -c conda-forge gdal"
        )

    print(f"[gis_setup] Using gdalwarp: {gdalwarp}")

    with rasterio.open(raster_path) as src:
        current_epsg = src.crs.to_epsg() if src.crs else None

    if current_epsg == target_epsg:
        print(f"[gis_setup] Already EPSG:{target_epsg} — no action needed.")
        return str(raster_path)

    print(f"[gis_setup] Reprojecting {raster_path.name}: EPSG:{current_epsg} → EPSG:{target_epsg}")

    if overwrite:
        out_path = raster_path.with_suffix(".tmp.tif")
    else:
        out_path = raster_path.with_name(f"{raster_path.stem}_EPSG{target_epsg}.tif")

    cmd = [
        gdalwarp,              # ✅ full path instead of bare "gdalwarp"
        "-t_srs",  f"EPSG:{target_epsg}",
        "-r",      "bilinear",
        "-co",     "COMPRESS=LZW",
        "-co",     "TILED=YES",
        "-overwrite",
        str(raster_path),
        str(out_path)
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"gdalwarp failed:\n{result.stderr}")

    if overwrite:
        raster_path.unlink()
        out_path.rename(raster_path)
        print(f"[gis_setup] Saved (overwritten): {raster_path}")
        return str(raster_path)
    else:
        print(f"[gis_setup] Saved alongside original: {out_path}")
        return str(out_path)
