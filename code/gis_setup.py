# gis_setup.py
import os
import sys
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

def ensure_raster_crs(raster_path, target_epsg=27700):
    """
    Checks if a raster has a valid EPSG CRS. 
    If it is 'LOCAL_CS' or missing, it overwrites it with the target EPSG.
    """
    try:
        with rasterio.open(raster_path, 'r+') as src:
            # Check if CRS is valid and matches target
            if src.crs is None or not src.crs.is_valid or src.crs.to_epsg() != target_epsg:
                print(f"Fixing CRS for: {os.path.basename(raster_path)}")
                print(f"   Was: {src.crs.wkt[:50]}...")
                
                # Define the correct CRS
                new_crs = CRS.from_epsg(target_epsg)
                src.crs = new_crs
                print(f"   Now: EPSG:{target_epsg}")
                return True
    except Exception as e:
        print(f"Error checking/fixing raster CRS: {e}")
        return False