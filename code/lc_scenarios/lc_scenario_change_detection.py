import rasterio
import numpy as np
import pandas as pd
import os
from pathlib import Path


# ================= CONFIGURATION =================
# 1. Input/Output Paths
lc_path = Path(r"G:\Shared drives\Wellcome Trust Project Data\1_preprocess\UrbanCoolingModel\OfficialWorkingInputs\LULC")

image_t1_path = lc_path / 'LCM2023_London_10m_clip2aoi_tcc24.tif'  # The "Before" image
image_t2_path = lc_path / 'LCM2023_London_10m_clip2aoi_tcc24_scenario4_nearest_to_edge_30prc_canopy_increase.tif'  # The "After" image
output_raster = lc_path / 'change_detection' / 'change_map_green30.tif'         # Where to save the visual map
output_stats  = lc_path / 'change_detection' / 'change_statistics_green30.csv'  # Where to save the area report

# 2. Updated Class Definitions based on your screenshots
class_dict = {
    1: 'Broadleaved woodland',
    2: 'Coniferous Woodland',
    3: 'Arable and Horticulture',
    4: 'Improved Grassland',
    5: 'Neutral Grassland',
    6: 'Calcareous Grassland',
    7: 'Acid grassland',
    8: 'Fen, Marsh and Swamp',
    9: 'Heather',
    10: 'Heather Grassland',
    11: 'Bog',
    12: 'Inland Rock',
    13: 'Saltwater',
    14: 'Freshwater',
    15: 'Supralittoral rock',
    16: 'Supralittoral sediment',
    17: 'Littoral rock',
    18: 'Littoral sediment',
    19: 'Saltmarsh',
    20: 'Urban',
    21: 'Suburban',
    100: 'TCC' # Likely Unclassified/Background
}

# 3. Encoding Multiplier
# CHANGED TO 1000 because you have a class ID of 100. 
# Logic: Code = (T1_Class * 1000) + T2_Class
# Example: Class 21 to Class 20 = 21020
MULTIPLIER = 1000 
# =================================================

def calculate_changes(t1_path, t2_path, out_tif, out_csv):
    print("Loading rasters...")
    
    with rasterio.open(t1_path) as src_t1:
        t1 = src_t1.read(1)
        meta = src_t1.meta.copy()
        res = src_t1.res # (x_resolution, y_resolution)
        
    with rasterio.open(t2_path) as src_t2:
        t2 = src_t2.read(1)
        
        # Validation checks
        if t1.shape != t2.shape:
            raise ValueError("Error: Images have different dimensions.")
        if src_t1.crs != src_t2.crs:
            print("Warning: CRS does not match. Ensure images are aligned.")

    print("Calculating transitions...")
    
    # Ensure data types handle the new values (uint32 is safest for multiplier 1000)
    t1 = t1.astype(np.uint32)
    t2 = t2.astype(np.uint32)
    
    # Calculate transition array
    change_array = (t1 * MULTIPLIER) + t2
    
    print("Generating statistics...")
    
    # Count pixels for each unique transition
    unique_codes, counts = np.unique(change_array, return_counts=True)
    
    # Calculate Area
    pixel_area_sqm = res[0] * res[1]
    
    stats_list = []
    
    for code, count in zip(unique_codes, counts):
        # Decode the transition
        class_t1 = code // MULTIPLIER
        class_t2 = code % MULTIPLIER
        
        # Get Names
        name_t1 = class_dict.get(class_t1, f"Unknown ({class_t1})")
        name_t2 = class_dict.get(class_t2, f"Unknown ({class_t2})")
        
        # Calculate Areas
        area_sqm = count * pixel_area_sqm
        area_km2 = area_sqm / 1_000_000
        
        # Determine status
        status = "Stable" if class_t1 == class_t2 else "Change"
        
        stats_list.append({
            'Transition_Code': code,
            'From_Class_ID': class_t1,
            'To_Class_ID': class_t2,
            'From_Label': name_t1,
            'To_Label': name_t2,
            'Status': status,
            'Pixel_Count': count,
            'Area_SqM': area_sqm,
            'Area_Km2': area_km2
        })
        
    # Convert to DataFrame and sort
    df = pd.DataFrame(stats_list)
    df = df.sort_values(by=['Status', 'Area_Km2'], ascending=[True, False])
    
    # Save CSV
    df.to_csv(out_csv, index=False)
    print(f"Statistics saved to {out_csv}")

    # Save Change Map Raster
    # We update metadata to uint32 to hold the larger transition codes
    meta.update(dtype=rasterio.uint32, count=1, nodata=0)
    
    with rasterio.open(out_tif, 'w', **meta) as dst:
        dst.write(change_array.astype(rasterio.uint32), 1)
    
    print(f"Change map saved to {out_tif}")

if __name__ == "__main__":
    try:
        calculate_changes(image_t1_path, image_t2_path, output_raster, output_stats)
    except Exception as e:
        print(f"An error occurred: {e}")