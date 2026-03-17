import os
import glob
import numpy as np
import rasterio
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

# ==========================================
# 1. DEFINE COLOR PALETTE (The "Symbology")
# ==========================================
# Mapping: Raster Value -> (Hex Color, Label)
# Based on UKCEH Land Cover classes + your new class 100

LC_SYMBOLOGY = {
    1:  ('#009E73', 'Broadleaved woodland'),    # Teal/Green
    2:  ('#004D00', 'Coniferous Woodland'),     # Dark Forest Green
    3:  ('#FFFF66', 'Arable and Horticulture'), # Bright Yellow
    4:  ('#A3FF73', 'Improved Grassland'),      # Light Lime Green
    5:  ('#7fe57f', 'Neutral Grassland'),       
    6:  ('#70a800', 'Calcareous Grassland'),    # Olive Green
    7:  ('#998100', 'Acid Grassland'),          # Pale Green
    8:  ('#ffff00', 'Fin, March and Swamp'),       
    9:  ('#730073', 'Heather'),                 # Purple
    10: ('#e68ca6', 'Heather Grassland'),       # Rose/Dark Pink
    11: ('#008073', 'Bog'),     
    12: ('#d2d2ff', 'Inland Rock'),             # Light Periwinkle
    13: ('#000080', 'Saltwater'),               # Navy Blue
    14: ('#0000FF', 'Freshwater'),              # Blue
    15: ('#ccb300', 'Supralittoral Rock'),      # Light Lavender
    16: ('#ccb300', 'Supralittoral Sediment'),  #
    17: ('#ffff80', 'Littoral Rock'),           # Medium Lavender
    18: ('#ffff80', 'Littoral Sediment'),
    19: ('#8080ff', 'Saltmarsh'),               # Light Orange
    20: ('#ff7f7f', 'Urban'),                   # Salmon/Light Red
    21: ('#ffbebe', 'Suburban'),                # Pale Pink
    100:('#006349', 'Tree cover'),              # Dark Jade Green (The new Scenario Class)
    0:  ('#FFFFFF', 'NA')                       # White
}

def array_to_rgb(data_array, symbology_dict):
    """
    Converts a 2D single-band raster array into a 3D RGB array
    based on the dictionary mapping.
    """
    # Get dimensions
    h, w = data_array.shape
    
    # Initialize empty 3D array (Height, Width, 3 channels for R,G,B)
    # Fill with white (1.0, 1.0, 1.0) by default
    rgb_image = np.ones((h, w, 3)) 

    for value, (hex_color, label) in symbology_dict.items():
        if value == 0: continue # Skip NoData (leave as white)
        
        # Convert Hex to RGB (0-1 range)
        rgb_color = [int(hex_color.lstrip('#')[i:i+2], 16)/255.0 for i in (0, 2, 4)]
        
        # Create a boolean mask for pixels matching this value
        mask = (data_array == value)
        
        # Apply color to those pixels
        rgb_image[mask] = rgb_color
        
    return rgb_image

def plot_single_scenario(raster_path, output_png_path):
    """
    Reads a raster, styles it, and saves a PNG plot.
    """
    with rasterio.open(raster_path) as src:
        data = src.read(1)
        
    # Check if there is actual data
    unique_vals = np.unique(data)
    if len(unique_vals) == 1 and unique_vals[0] == 0:
        print(f"Skipping empty raster: {os.path.basename(raster_path)}")
        return

    # Convert to RGB image
    img_rgb = array_to_rgb(data, LC_SYMBOLOGY)

    # --- PLOTTING ---
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # Remove axis ticks/labels for a cleaner map look
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(os.path.basename(raster_path), fontsize=14, fontweight='bold')
    
    # Show image
    ax.imshow(img_rgb)

    # --- CREATE LEGEND ---
    # Only create legend entries for values that actually exist in this raster
    legend_patches = []
    for val in unique_vals:
        if val in LC_SYMBOLOGY and val != 0:
            color, label = LC_SYMBOLOGY[val]
            patch = mpatches.Patch(color=color, label=label)
            legend_patches.append(patch)
    
    # Add legend to side
    if legend_patches:
        ax.legend(handles=legend_patches, loc='center left', bbox_to_anchor=(0, 0.15), title="Land Cover")

    # Save
    plt.tight_layout()
    plt.savefig(output_png_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved: {output_png_path}")


# ==========================================
# 2. BATCH EXECUTION
# ==========================================
if __name__ == "__main__":
    
    # Define directories
    SCENARIO_DIR = r"E:/London/jlg_tree_planting/lc_scenarios_output"
    PLOT_DIR = os.path.join(SCENARIO_DIR, "Plots")
    
    if not os.path.exists(PLOT_DIR):
        os.makedirs(PLOT_DIR)

    # Find all TIFs
    tif_files = glob.glob(os.path.join(SCENARIO_DIR, "*.tif"))
    
    print(f"Found {len(tif_files)} rasters to plot.")

    for tif in tif_files:
        filename = os.path.basename(tif)
        # Create output filename (e.g., LULC_Scenario_A.png)
        out_name = filename.replace(".tif", ".png")
        out_path = os.path.join(PLOT_DIR, out_name)
        
        plot_single_scenario(tif, out_path)