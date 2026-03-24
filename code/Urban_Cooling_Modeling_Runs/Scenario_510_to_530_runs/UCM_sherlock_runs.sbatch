#!/bin/bash
#
#SBATCH --time=6:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --mail-type=ALL
#SBATCH --partition=hns,normal,serc
#SBATCH --job-name="UCM-run"

# This script is used to run batches
# on Sherlock. If you are running locally,
# please use invest_setup.sh

module load python/3.9.0
module load physics gdal/3.5.2 py-gdal-utils
module load gcc/14.2.0

TARGET_DIR='/oak/stanford/projects/natcap/users/epavia/WellcomeTrustProjectData'

# Run with correct path prefix
#python3 execute_invest_urban_cooling_model_scenario510.py $TARGET_DIR
#python3 execute_invest_urban_cooling_model_scenario520.py $TARGET_DIR
#python3 execute_invest_urban_cooling_model_scenario530.py $TARGET_DIR