@echo off
echo Starting heat mortality analysis...
echo.

python D:/natcap/urban-cooling-health/code/health_assessment/health-modeling.py ^
    --t_baseline "G:/Shared drives/Wellcome Trust Project Data/2_postprocess_intermediate/UCM_official_runs/scenario0/work_and_energy_runs/intermediate/T_air_london_scenario_20.0deg_2.0uhi_66.9hum_energy_productivity.tif" ^
    --t_scenario "G:/Shared drives/Wellcome Trust Project Data/2_postprocess_intermediate/UCM_official_runs/scenario3/work_and_energy_runs/intermediate/T_air_london_scenario3_25deg_5uhi_45hum_energy_productivity.tif" ^
    --align_to   "G:/Shared drives/Wellcome Trust Project Data/2_postprocess_intermediate/UCM_official_runs/scenario0/work_and_energy_runs/intermediate/T_air_london_scenario_20.0deg_2.0uhi_66.9hum_energy_productivity.tif" ^
	--pop_baseline "G:/Shared drives/Wellcome Trust Project Data/1_preprocess/UrbanCoolingModel/OfficialWorkingInputs/pop_raster/resampled_10m/gbr_pop_2021_10m_areal.tif" ^
    --pop_scenario "G:/Shared drives/Wellcome Trust Project Data/1_preprocess/UrbanCoolingModel/OfficialWorkingInputs/pop_raster/resampled_10m/gbr_pop_2021_10m_areal.tif" ^
    --baseline_deaths_cardio "G:/Shared drives/Wellcome Trust Project Data/1_preprocess/UrbanCoolingModel/OfficialWorkingInputs/health_rasters_10m_bng/i00_i99_ix_diseases_of_the_circulatory_system_bng10m_2021.tif" ^
    --baseline_deaths_resp   "G:/Shared drives/Wellcome Trust Project Data/1_preprocess/UrbanCoolingModel/OfficialWorkingInputs/health_rasters_10m_bng/j00_j99_x_diseases_of_the_respiratory_system_bng10m_2021.tif" ^
	--baseline_deaths_cere   "G:/Shared drives/Wellcome Trust Project Data/1_preprocess/UrbanCoolingModel/OfficialWorkingInputs/health_rasters_10m_bng/x60_x84_intentional_self_harm_bng10m_2021.tif" ^
    --out_dir "G:/Shared drives/Wellcome Trust Project Data/2_postprocess_intermediate/UCM_official_runs/health_output_s0_s3" ^
    --n_draws 100

echo.
echo Analysis complete! Check the output directory for results.
pause