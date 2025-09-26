#!/bin/bash
#
#SBATCH --time=16:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --mail-type=ALL
#SBATCH --partition=hns,normal,serc
#SBATCH --job-name="UCM-run"

# This script is used to run batches
# on Sherlock. If you are running locally,
# please use invest_setup.sh

module load physics gdal/3.5.2
module load gcc/14.2.0
module load python/3.9.0


TARGET_DIR='/oak/stanford/groups/gdaily/users/epavia/WellcomeTrustProjectData'

# Run with correct path prefix
# Note: scenario4_10prc was removed as it was close to opportunity trees
python3 execute_invest_urban_cooling_model_current_lulc.py $TARGET_DIR --eap
python3 execute_invest_urban_cooling_model_scenario1.py $TARGET_DIR --eap
python3 execute_invest_urban_cooling_model_scenario2.py $TARGET_DIR --eap
python3 execute_invest_urban_cooling_model_scenario3.py $TARGET_DIR --eap 
python3 execute_invest_urban_cooling_model_scenario4_20prc_TCC.py $TARGET_DIR --eap
python3 execute_invest_urban_cooling_model_scenario4_30prc_TCC.py $TARGET_DIR --eap 