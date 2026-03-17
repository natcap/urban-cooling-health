
import rasterio
import rasterio.mask
from rasterio.crs import CRS
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patches as mpatches
from pathlib import Path


import os
import sys

# ------------------------------------------------------------------
# FIX FOR "Cannot find proj.db" ERROR
# ------------------------------------------------------------------
try:
    import pyproj
    # Get the path to the PROJ database from pyproj
    proj_path = pyproj.datadir.get_data_dir()
    
    # Force the environment variable to point to this path
    os.environ['PROJ_LIB'] = proj_path
    print(f"DEBUG: PROJ_LIB set to {proj_path}")
except ImportError:
    print("Warning: pyproj not found. If you get a CRSError, install pyproj.")
except Exception as e:
    print(f"Warning: Could not auto-fix PROJ_LIB: {e}")
# ------------------------------------------------------------------

# ================= CONFIGURATION =================
lc_path = Path(r"G:\Shared drives\Wellcome Trust Project Data\1_preprocess\UrbanCoolingModel\OfficialWorkingInputs\LULC")
sf_path = Path(r'G:\Shared drives\Wellcome Trust Project Data\1_preprocess\UrbanCoolingModel\OfficialWorkingInputs\AOIs')

# aoi_path = sf_path / "London_Borough_aoi.shp"
# full_aoi = gpd.read_file(aoi_path)
# aoi = full_aoi.loc[:, ["NAME", "GSS_CODE", "geometry"]].query("NAME == 'Camden'")
aoi_path = r'E:\London\jlg_tree_planting\aoi_Camden.gpkg'

input_raster = lc_path / 'change_detection' / 'change_map_green30.tif'         # Where to save the visual map
output_png_simple   = lc_path / 'change_detection' / 'map_change_green30_simple.png'
output_png_detailed = lc_path / 'change_detection' / 'map_change_green30_detailed.png'


# *** THE FIX: MANUALLY SET THE EPSG CODE ***
# Common UK Codes: 
# 27700 (British National Grid) - Most likely for UK government data/OS
# 32630 (UTM Zone 30N) - Common for Satellite imagery (Sentinel-2)
# 4326  (Lat/Lon) - Standard GPS
FORCE_RASTER_EPSG = 27700 

MULTIPLIER = 1000 
# =================================================

def visualize_change(tif_path, shp_path=None):
    print("Reading data...")
    
    with rasterio.open(tif_path) as src:
        # 1. DEFINE SOURCE CRS (The Fix)
        # We ignore what the file 'says' it is and force a known standard
        if FORCE_RASTER_EPSG:
            print(f"Forcing Raster CRS to EPSG:{FORCE_RASTER_EPSG}")
            raster_crs = CRS.from_epsg(FORCE_RASTER_EPSG)
        else:
            raster_crs = src.crs

        # ---------------------------------------------------------
        # 1. CLIP TO AOI LOGIC
        # ---------------------------------------------------------
        if shp_path:
            print(f"Clipping to AOI: {shp_path}")
            # Read the shapefile
            gdf = gpd.read_file(shp_path)
            
            # --- DEBUGGING CRS ---
            print(f"Raster CRS: {src.crs}")
            print(f"AOI CRS:    {gdf.crs}")

            # Reproject AOI to match the forced Raster CRS
            if gdf.crs != raster_crs:
                print(f"Reprojecting AOI from {gdf.crs} to EPSG:{FORCE_RASTER_EPSG}...")
                gdf = gdf.to_crs(epsg=FORCE_RASTER_EPSG)
            
            # Convert geometry for masking
            shapes = gdf.geometry.values
            
            # Clip
            try:
                out_image, out_transform = rasterio.mask.mask(src, shapes, crop=True)
                data = out_image[0]
            except ValueError:
                print("\nCRITICAL ERROR: The AOI and Raster do not overlap.")
                print(f"1. Check if EPSG:{FORCE_RASTER_EPSG} is correct for your raster.")
                print("2. Try changing FORCE_RASTER_EPSG to 32630 (Sentinel-2) or 4326 (Lat/Lon).")
                return
        else:
            data = src.read(1)

    # ---------------------------------------------------------
    # 2. DECODE DATA
    # ---------------------------------------------------------

    if data is None: return

    print("Generating Map...")

    # Everything outside the AOI will be 0 (NoData)
        
    # 1. Decode the data
    # Calculate 'From' and 'To' classes from the integer code
    from_class = data // MULTIPLIER
    to_class = data % MULTIPLIER
    
    # 2. Create Masks
    # Ignore background (0) or the specific artifact (0->255) seen in your stats
    valid_mask = (data != 0) & (data != 255) 
    
    # Identify pixels where From != To (Change) vs From == To (Stable)
    change_mask = (from_class != to_class) & valid_mask
    stable_mask = (from_class == to_class) & valid_mask



    # ---------------------------------------------------------
    # 3. VISUALIZE
    # ---------------------------------------------------------

    # ======================================================
    # PLOT 1: SIMPLE (Red = Change, Grey = Stable)
    # ======================================================
    print("Generating Simple Change Map...")
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # Create an empty RGB array (White background)
    # Shape: (Height, Width, 3 for RGB) | Init with 1.0 (White)
    vis_image = np.ones((data.shape[0], data.shape[1], 3))

        
    # Paint Stable areas Light Grey
    vis_image[stable_mask] = [0.85, 0.85, 0.85] 
    
    # Paint Changed areas Red
    vis_image[change_mask] = [1.0, 0.0, 0.0]    
    
    # Display
    ax.imshow(vis_image)
    ax.set_title("Change Detection: Stable (Grey) vs Change (Red)")
    ax.axis('off') # Hide coordinates
    
    # Add simple Legend
    legend_elements = [
        mpatches.Patch(color='#d9d9d9', label='Stable'),
        mpatches.Patch(color='#ff0000', label='Change')
    ]
    ax.legend(handles=legend_elements, loc='lower left')
    
    plt.savefig(output_png_simple, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_png_simple}")
    plt.close()

    # ======================================================
    # PLOT 2: DETAILED (Color by "Target" Class)
    # ======================================================
    print("Generating Detailed Change Map...")
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # Reset image to white
    vis_image = np.ones((data.shape[0], data.shape[1], 3))
    
    # Paint Stable areas Very Light Grey (Background)
    vis_image[stable_mask] = [0.9, 0.9, 0.9]

    # --- DEFINE COLORS FOR SPECIFIC TARGETS ---
    # Based on your CSV, valid target classes (To_Class_ID) include:
    # 100 (TCC), 20 (Urban), 21 (Suburban), 4 (Grass), etc.
    
    # Get the "To" class ONLY where change occurred
    changed_targets = to_class * change_mask
    unique_targets = np.unique(changed_targets)

    # Define colors for the 'To' classes (R, G, B) 0.0-1.0
    # You can customize these colors
    colors = {
        100: [0.0, 0.6, 0.0],  # TCC (Green)
        20:  [0.8, 0.0, 0.0],  # Urban (Red)
        21:  [1.0, 0.6, 0.6],  # Suburban (Pink)
        4:   [0.6, 0.8, 0.2],  # Grass (Lime)
        1:   [0.1, 0.4, 0.1],  # Woodland (Dark Green)
        # Add a default for others:
        'default': [0.0, 0.0, 1.0] # Blue for anything else
    }

    # Apply colors
    # We loop through unique target classes found in the change mask
    
    for target in unique_targets:
        if target == 0: continue # Skip background
        
        # Determine color
        c = colors.get(target, colors['default'])
        
        # Apply color to pixels where (Change occurred) AND (Target == this class)
        target_mask = (changed_targets == target)
        vis_image[target_mask] = c

    ax.imshow(vis_image)
    ax.set_title("Change Detection: Colored by New Land Use")
    ax.axis('off')
    
    # Create Dynamic Legend
    patches = [mpatches.Patch(color='#e6e6e6', label='Stable')]
    for target in unique_targets:
        if target == 0: continue
        c = colors.get(target, colors['default'])
        label = f"Change to Class {target}"
        
        # Give nicer names if known
        if target == 100: label = "Change to TCC"
        if target == 20: label = "Change to Urban"
        if target == 21: label = "Change to Suburban"
        if target == 4: label = "Change to Grass"
        
        patches.append(mpatches.Patch(color=c, label=label))

    ax.legend(handles=patches, loc='lower left')

    plt.savefig(output_png_detailed, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_png_detailed}")
    plt.close()

if __name__ == "__main__":
    visualize_change(input_raster, aoi_path)