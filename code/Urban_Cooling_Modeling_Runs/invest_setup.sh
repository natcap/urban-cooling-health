# This script will allow users to run
# various scenarios of the Urban Cooling Model
# for the Wellcome Trust Project.
# to start, make sure you have an environment created with
# the package by running: mamba create -n invest-env -c conda-forge natcap.invest
# once complete, you will run: conda activate invest-env
# next you can manually edit the model parameter files then run this
# make sure you have your correct prefixes
# as a first argument
# script in full: sh invest_setup.sh or the
# code for each scenario below:

path_prefix = 'G:\Shared drives\Wellcome Trust Project Data'
# Make sure to cd into the following directory:
cd ~/code/Urban_Cooling_Modeling_Runs/

# runs all temp and uhi combos for current and future conditions
# for the current lulc with tcc (--eap flag will run work productivity and energy)
python execute_invest_urban_cooling_model_current_lulc.py $path_prefix --eap

# runs all temp and uhi combos for current and future conditions
# for scenario 1 (--eap flag will run work productivity and energy)
python execute_invest_urban_cooling_model_scenario1.py --eap

# runs all temp and uhi combos for current and future conditions
# for scenario 2 (--eap flag will run work productivity and energy)
python execute_invest_urban_cooling_model_scenario2.py --eap

# runs all temp and uhi combos for current and future conditions
# for scenario 3 (--eap flag will run work productivity and energy)
python execute_invest_urban_cooling_model_scenario3.py --eap