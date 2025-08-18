import os
import numpy as np
import rasterio
import geopandas as gpd
from rasterio.mask import mask

def clip_raster_to_aoi(
    raster_path: str,
    aoi,                          # path to vector OR a GeoDataFrame
    out_path: str = None,
    crop: bool = True,
    all_touched: bool = False,
    compress: str = "lzw",
    replace_nodata_with: int = 0,
    keep_nodata_tag: bool = False  # set True ONLY if you really want nodata=0 in metadata
):
    """
    Clip a raster to an AOI and replace masked/nodata cells with 0 (or another integer).
    Returns an integer array; no NaNs are introduced.

    - Uses rasterio.mask(..., filled=False) to get a MaskedArray, then fills masked cells.
    - By default, clears the nodata tag so GIS won't hide your 0 class.
    """
    # Load AOI
    aoi_gdf = gpd.read_file(aoi) if isinstance(aoi, str) else aoi.copy()

    with rasterio.open(raster_path) as src:
        # CRS align
        if aoi_gdf.crs != src.crs:
            aoi_gdf = aoi_gdf.to_crs(src.crs)

        # Dissolve to single geometry (no explicit shapely import needed)
        geoms = [aoi_gdf.unary_union.__geo_interface__]

        # Get masked array so we can precisely fill outside-AOI/nodata
        data_ma, transform = mask(
            src, geoms, crop=crop, all_touched=all_touched, filled=False
        )

        # Prepare profile
        profile = src.profile.copy()
        profile.update({
            "height": data_ma.shape[1],
            "width":  data_ma.shape[2],
            "transform": transform,
        })
        if profile.get("driver", "GTiff") == "GTiff":
            profile.update({"compress": compress})

        # Fill masked cells (outside AOI AND original nodata) with desired value (default 0)
        # Keep original dtype (e.g., uint8)
        dtype = np.dtype(profile["dtype"])
        filled = np.ma.filled(data_ma, fill_value=replace_nodata_with).astype(dtype, copy=False)

        # Metadata: by default, remove nodata tag so class 0 is not hidden by GIS
        if keep_nodata_tag:
            profile["nodata"] = replace_nodata_with
        else:
            profile["nodata"] = None  # write without a nodata tag

    # Optional write
    if out_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(filled)

    return filled, transform, profile
