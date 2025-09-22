#!/bin/bash
#
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --mail-type=ALL
#SBATCH --partition=hns,normal,serc
#SBATCH --job-name="UCM-run"

module load physics gdal/3.5.2
module load gcc/14.2.0
module load python/3.9.0


TARGET_DIR='/oak/stanford/groups/gdaily/users/epavia/WellcomeTrustProjectData'

# Run with correct path prefix
#python3 execute_invest_urban_cooling_model_current_lulc.py $TARGET_DIR --eap &
python3 execute_invest_urban_cooling_model_scenario1.py $TARGET_DIR --eap
#python3 execute_invest_urban_cooling_model_scenario2.py $TARGET_DIR --eap &
#python3 execute_invest_urban_cooling_model_scenario3.py $TARGET_DIR --eap 
#python3 execute_invest_urban_cooling_model_scenario4_20prc_TCC.py $TARGET_DIR --eap &
#python3 execute_invest_urban_cooling_model_scenario4_30prc_TCC.py $TARGET_DIR --eap 