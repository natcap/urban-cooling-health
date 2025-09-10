#!/bin/bash
#
# A script to download, average, and stitch
# ECOSTRESS albedo tiles for London for UCM model run.
# Data downloaded from: https://www.earthdata.nasa.gov/data/catalog/lpcloud-eco-l2t-stars-002
# Read: README.txt for more information

ROOT_DIR="G:\Shared drives\Wellcome Trust Project Data\1_preprocess\UrbanCoolingModel\EP_preliminary_tests\albedo\ECOSTRESS_ECO_L2T_STARS"
URLS=ECOSTRESS_DOWNLOADS.txt
FILES=downloaded-files.txt
# .env file is where nasa login credentials live
ENV_FILE="C:\Users\epavia\Desktop\natcap\repo\urban-cooling-health\.env"

declare -a TILE_LIST=("30UXB" "30UXC" "30UYB" "30UYC")

# import NASA credentials
if [ -f "$ENV_FILE" ]; then
        # Source the .env file to load variables
        set -o allexport # Automatically export all variables defined after this point
        source "$ENV_FILE"
        set +o allexport # Turn off automatic exporting
    else
        echo "Error: .env file not found"
        exit 1
fi

IFS=$'\r\n'       
set -f          # disable globbing
# Download only the albedo geoTIFF files
for t in $(cat $URLS); do
   if echo "$t" | grep "_albedo.tif$"; then
       echo "downloading $t"
       wget $t \	        
       --no-verbose --no-clobber --tries=20 --random-wait --retry-connrefused \		    
       --user=$NASA_EARTHDATA_USERNAME --password=$NASA_EARTHDATA_PASSWORD \		    
       --directory-prefix=$ROOT_DIR\\downloads
   else
       : # do nothing
   fi
done

# list files to feed into gdal
ls -R $ROOT_DIR/downloads > downloaded-files.txt

#create raster stack for each tile and find averages
for t in "${TILE_LIST[@]}"; do
   echo $t
   grep "$t" $FILES | while IFS= read -r line;
   do
       tilepath="downloads\\"$line
       echo $tilepath >> $ROOT_DIR/$t.txt
   done
done 
    
cd $ROOT_DIR
for t in "${TILE_LIST[@]}"; do
    echo $t
    gdal raster stack --optfile intermediate/$t.txt intermediate/$t'_raster_stack.tif'
    gdal_calc.py -A intermediate/$t'_raster_stack.tif' --outfile intermediate/$t'avg_albedo.tif' --calc="numpy.average(A,axis=0)"
done

# join all new tiffs to one geotiff
gdalbuildvrt albedo_file.vrt intermediate/*avg_albedo.tif
GDAL_CACHEMAX=2048 gdal_translate -of GTiff intermediate/albedo_file.vrt ecostress_avg_albedo_07_08.tif
