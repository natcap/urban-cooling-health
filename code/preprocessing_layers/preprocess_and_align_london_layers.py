### This script will be used to preprocess and align rasters
## for input into the Urban Cooling Model.

import geopandas as gpd
import os
import rasterio
from rasterio.mask import mask

# set working directory and project CRS
wd = "G:/Shared drives/Wellcome Trust Project Data/1_preprocess/UrbanCoolingModel"
target_crs = "EPSG:27700"

#import AOI
aoi_path = os.path.join(wd, "EP_preliminary_tests/buffered_aois/London_Ward_aoi_5km_buffer.shp")
aoi_borough_path = os.path.join(wd, "London_Borough_aoi.shp")

aoi = gpd.read_file(aoi_path).to_crs(target_crs)
aoi_borough = gpd.read_file(aoi_borough_path).to_crs(target_crs)

# subset boroughs of interest
# according to this: https://trustforlondon.org.uk/news/borough-level-poverty-2025/
# some areas with high poverty and deprivation are Camden and Enfield
camden = aoi_borough[aoi_borough['NAME'] == 'Camden']
enfield = aoi_borough[aoi_borough['NAME'] == 'Enfield']

# create buffer of 1000m
camden_buffer = camden.buffer(500)
enfield_buffer = enfield.buffer(500)

# export buffered layers for future use
camden_buffer.to_file(os.path.join(wd, "EP_preliminary_tests/buffered_aois/camden_borough_500m_buffer.shp"))
enfield_buffer.to_file(os.path.join(wd, "EP_preliminary_tests/buffered_aois/enfield_borough_500m_buffer.shp"))

## Clip all LULC layers with buffers:
# GLC LULC - files were reprojected and clipped to London using QGIS due to the large size of the files
glc_path = os.path.join(wd, "EP_preliminary_tests/clipped_lulc/GLC")

# clip glc layers to borough level
for filename in os.listdir(glc_path):
    if filename.endswith('.tif'):
        print(filename)
        file_path = os.path.join(glc_path, filename)
        print(file_path)
        # clip to camden
        with rasterio.open(file_path, crs=target_crs) as glc_lc:
            print(glc_lc.crs)
            print(glc_lc.meta)
        # Clip the raster using the shapefile
            landcover_clipped, clipped_transform = mask(glc_lc, camden_buffer.geometry, crop=True)
            # Update metadata for the clipped raster
            clipped_meta = glc_lc.meta.copy()
            clipped_meta.update({
                "driver": "GTiff",
                "height": landcover_clipped.shape[1],
                "width": landcover_clipped.shape[2],
                "transform": clipped_transform,
                "count": 1,
                "dtype": "float32",
                "nodata": 255
            })
            # Save the clipped raster
            landcover_clipped_path = file_path.replace('_london_reprojected.tif', '_camden.tif')
            print(f"Clipped raster saved to: {landcover_clipped_path}")
            with rasterio.open(landcover_clipped_path, "w", **clipped_meta) as dest:
                dest.write(landcover_clipped[0].astype("float32"), 1)


# UKECH LULC
ukech2000_path = "G:/Shared drives/Wellcome Trust Project Data/0_source_data/UKCEH Land Cover 2000/data/LCM2000.tif"
ukech2021_path = "G:/Shared drives/Wellcome Trust Project Data/0_source_data/UKCEH Land Cover 2021/data/LCM.tif" # need band 1

with rasterio.open(ukech2000_path, crs=target_crs) as ukech2000:
    print(ukech2000.crs)
    print(ukech2000.meta)
    # Clip the raster using the shapefile
    landcover_clipped, clipped_transform = mask(ukech2000, aoi.geometry, crop=True)
    # Update metadata for the clipped raster
    clipped_meta = ukech2000.meta.copy()
    clipped_meta.update({
        "driver": "GTiff",
        "height": landcover_clipped.shape[1],
        "width": landcover_clipped.shape[2],
        "transform": clipped_transform,
        "count": 1,
        "dtype": "float32",
        "nodata": 255
    })
    filename = os.path.basename(ukech2000_path)
    file_path = os.path.join(wd, "EP_preliminary_tests/clipped_lulc/UKECH", filename)
    # Save the clipped raster
    landcover_clipped_path = file_path.replace('.tif', '_london.tif')
    print(f"Clipped raster saved to: {landcover_clipped_path}")
    with rasterio.open(landcover_clipped_path, "w", **clipped_meta) as dest:
        dest.write(landcover_clipped[0].astype("float32"), 1)
        # Clip the raster using the shapefile
    landcover_clipped, clipped_transform = mask(ukech2000, camden_buffer.geometry, crop=True)
    # Update metadata for the clipped raster
    clipped_meta = ukech2000.meta.copy()
    clipped_meta.update({
        "driver": "GTiff",
        "height": landcover_clipped.shape[1],
        "width": landcover_clipped.shape[2],
        "transform": clipped_transform,
        "count": 1,
        "dtype": "float32",
        "nodata": 255
    })
    filename = os.path.basename(ukech2000_path)
    file_path = os.path.join(wd, "EP_preliminary_tests/clipped_lulc/UKECH", filename)
    # Save the clipped raster
    landcover_clipped_path = file_path.replace('.tif', '_camden.tif')
    print(f"Clipped raster saved to: {landcover_clipped_path}")
    with rasterio.open(landcover_clipped_path, "w", **clipped_meta) as dest:
        dest.write(landcover_clipped[0].astype("float32"), 1)

with rasterio.open(ukech2021_path, crs=target_crs) as ukech2021:
    print(ukech2021.crs)
    print(ukech2021.meta)
    # Clip the raster using the shapefile
    landcover_clipped, clipped_transform = mask(ukech2021, aoi.geometry, crop=True)
    # Update metadata for the clipped raster
    clipped_meta = ukech2021.meta.copy()
    clipped_meta.update({
        "driver": "GTiff",
        "height": landcover_clipped.shape[1],
        "width": landcover_clipped.shape[2],
        "transform": clipped_transform,
        "count": 1,
        "dtype": "float32",
        "nodata": 255
    })
    filename = os.path.basename(ukech2021_path)
    file_path = os.path.join(wd, "EP_preliminary_tests/clipped_lulc/UKECH", filename)
    # Save the clipped raster
    landcover_clipped_path = file_path.replace('.tif', '2021_london.tif')
    print(f"Clipped raster saved to: {landcover_clipped_path}")
    with rasterio.open(landcover_clipped_path, "w", **clipped_meta) as dest:
        dest.write(landcover_clipped[0].astype("float32"), 1)
        # Clip the raster using the shapefile
    landcover_clipped, clipped_transform = mask(ukech2021, camden_buffer.geometry, crop=True)
    # Update metadata for the clipped raster
    clipped_meta = ukech2021.meta.copy()
    clipped_meta.update({
        "driver": "GTiff",
        "height": landcover_clipped.shape[1],
        "width": landcover_clipped.shape[2],
        "transform": clipped_transform,
        "count": 1,
        "dtype": "float32",
        "nodata": 255
    })
    filename = os.path.basename(ukech2021_path)
    file_path = os.path.join(wd, "EP_preliminary_tests/clipped_lulc/UKECH", filename)
    # Save the clipped raster
    landcover_clipped_path = file_path.replace('.tif', '2021_camden.tif')
    print(f"Clipped raster saved to: {landcover_clipped_path}")
    with rasterio.open(landcover_clipped_path, "w", **clipped_meta) as dest:
        dest.write(landcover_clipped[0].astype("float32"), 1)


### ESA
esa_path = "G:/Shared drives/Wellcome Trust Project Data/1_preprocess/UrbanCoolingModel/ESA_WorldCover_10m_2021_v200_Mosaic_Mask_proj_clipped.tif"

with rasterio.open(esa_path, crs=target_crs) as esa2021:
    print(esa2021.crs)
    print(esa2021.meta)
    # Clip the raster using the shapefile
    landcover_clipped, clipped_transform = mask(esa2021, aoi.geometry, crop=True)
    # Update metadata for the clipped raster
    clipped_meta = esa2021.meta.copy()
    clipped_meta.update({
        "driver": "GTiff",
        "height": landcover_clipped.shape[1],
        "width": landcover_clipped.shape[2],
        "transform": clipped_transform,
        "count": 1,
        "dtype": "float32",
        "nodata": 255
    })
    filename = os.path.basename(esa_path)
    file_path = os.path.join(wd, "EP_preliminary_tests/clipped_lulc/ESA", filename)
    # Save the clipped raster
    landcover_clipped_path = file_path.replace('.tif', '_london.tif')
    print(f"Clipped raster saved to: {landcover_clipped_path}")
    with rasterio.open(landcover_clipped_path, "w", **clipped_meta) as dest:
        dest.write(landcover_clipped[0].astype("float32"), 1)
        # Clip the raster using the shapefile
    landcover_clipped, clipped_transform = mask(esa2021, camden_buffer.geometry, crop=True)
    # Update metadata for the clipped raster
    clipped_meta = esa2021.meta.copy()
    clipped_meta.update({
        "driver": "GTiff",
        "height": landcover_clipped.shape[1],
        "width": landcover_clipped.shape[2],
        "transform": clipped_transform,
        "count": 1,
        "dtype": "float32",
        "nodata": 255
    })
    filename = os.path.basename(esa_path)
    file_path = os.path.join(wd, "EP_preliminary_tests/clipped_lulc/ESA", filename)
    # Save the clipped raster
    landcover_clipped_path = file_path.replace('.tif', '_camden.tif')
    print(f"Clipped raster saved to: {landcover_clipped_path}")
    with rasterio.open(landcover_clipped_path, "w", **clipped_meta) as dest:
        dest.write(landcover_clipped[0].astype("float32"), 1)