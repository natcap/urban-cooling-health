# Run before running on Sherlock: module load python/3.9.0

import geopandas as gpd
import os
import pandas as pd

# aggregate the building shapefile data outputs
# to compare to already aggregated results
path = "/oak/stanford/groups/gdaily/users/epavia/WellcomeTrustProjectData/"
boroughs = gpd.read_file(os.path.join(path, "1_preprocess/UrbanCoolingModel/OfficialWorkingInputs/AOIs/London_Borough_aoi.shp"))
boroughs = boroughs[['NAME', 'geometry']]
building = gpd.read_file(os.path.join(path, "2_postprocess_intermediate/UCM_official_runs/current_lulc/work_and_energy_runs/buildings_with_stats_london_scenario_25.0deg_5.0uhi_45.0hum_energy_productivity.shp")) # 25 deg 5 uhi data


# Intersect to add boroughs to building data
intersection = gpd.overlay(boroughs, building, how='intersection')
intersection_nogeom = intersection.drop(columns=['geometry'])

intersection_nogeom.to_csv(os.path.join(path, '2_postprocess_intermediate/UCM_official_runs/cleaned-data/aggregated_building_data/buildings_by_borough_25deg_5uhi.csv'), index=False)
