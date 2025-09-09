#!/bin/bash
#
# A script to download, average, and stitch
# ECOSTRESS albedo tiles for London for UCM model run.
# Data downloaded from: https://www.earthdata.nasa.gov/data/catalog/lpcloud-eco-l2t-stars-002
# Read: README.txt for more information

ROOT_DIR="G:\Shared drives\Wellcome Trust Project Data\1_preprocess\UrbanCoolingModel\EP_preliminary_tests\albedo\ECOSTRESS_ECO_L2T_STARS"
URLS=ECOSTRESS_DOWNLOADS.txt
ENV="C:\Users\epavia\Desktop\natcap\repo\urban-cooling-health\.env"

# import NASA credentials
if [ -f "$ENV" ]; then
        # Source the .env file to load variables
        set -o allexport # Automatically export all variables defined after this point
        source "$ENV"
        set +o allexport # Turn off automatic exporting
    else
        echo "Error: .env file not found"
        exit 1
fi

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

#cat $(URLS) | 
#
#srtm.vrt:
#	gdalbuildvrt $@ tiles/*.hgt{,.zip}