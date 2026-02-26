# This script will allow users to run
# various scenarios of the Urban Cooling Model
# for the Wellcome Trust Project.
# to start, make sure you have an environment created with
# the package by running: mamba create -n invest-env -c conda-forge natcap.invest
# once complete, you will run: conda activate invest-env
# next you can manually edit the model parameter files then run this
# make sure you have your correct prefixes
# as a first argument
# script in full: sh invest_setup_camden.sh or the
# code for each scenario below:

# Run the python script to run the model.
# To change which scenarios are run, you will need to edit the python script directly
python execute_invest_urban_cooling_model_camden.py 'G:\Shared drives\Wellcome Trust Project Data'
