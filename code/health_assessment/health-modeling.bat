@echo off
echo Starting heat mortality analysis...
echo.

python D:/natcap/urban-cooling-health/code/health_assessment/health-modeling.py ^
    --t_baseline "G:/Shared drives/Wellcome Trust Project Data/2_postprocess_intermediate/UCM_official_runs/current_lulc/current_climate/intermediate/T_air_london_current_scenario_20deg_2uhi.tif" ^
    --t_scenario "G:/Shared drives/Wellcome Trust Project Data/2_postprocess_intermediate/UCM_official_runs/current_lulc/current_climate/intermediate/T_air_london_current_scenario_25deg_2uhi.tif" ^
    --align_to "G:/Shared drives/Wellcome Trust Project Data/1_preprocess/UrbanCoolingModel/OfficialWorkingInputs/pop_raster/gbr_pop_2021_CN_100m_R2025A_v1_EPSG27700_clip2aoi.tif" ^
	--pop_baseline "G:/Shared drives/Wellcome Trust Project Data/1_preprocess/UrbanCoolingModel/OfficialWorkingInputs/pop_raster/gbr_pop_2021_CN_100m_R2025A_v1_EPSG27700_clip2aoi.tif" ^
    --pop_scenario "G:/Shared drives/Wellcome Trust Project Data/1_preprocess/UrbanCoolingModel/OfficialWorkingInputs/pop_raster/gbr_pop_2030_CN_100m_R2025A_v1_EPSG27700_clip2aoi.tif" ^
    --baseline_deaths_cardio "G:/Shared drives/Wellcome Trust Project Data/1_preprocess/UrbanCoolingModel/OfficialWorkingInputs/health_rasters_100m_bng/i00_i99_ix_diseases_of_the_circulatory_system_bng100m_2021.tif" ^
    --baseline_deaths_resp   "G:/Shared drives/Wellcome Trust Project Data/1_preprocess/UrbanCoolingModel/OfficialWorkingInputs/health_rasters_100m_bng/j00_j99_x_diseases_of_the_respiratory_system_bng100m_2021.tif" ^
	--baseline_deaths_cere   "G:/Shared drives/Wellcome Trust Project Data/1_preprocess/UrbanCoolingModel/OfficialWorkingInputs/health_rasters_100m_bng/j00_j99_x_diseases_of_the_respiratory_system_bng100m_2021.tif" ^
    --out_dir "G:/Shared drives/Wellcome Trust Project Data/2_postprocess_intermediate/UCM_official_runs/health_output" ^
    --n_draws 1000

echo.
echo Analysis complete! Check the output directory for results.
pause