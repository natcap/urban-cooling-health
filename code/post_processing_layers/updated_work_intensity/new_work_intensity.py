"""Calculate work intensity with the Hothaps function.

Written by James Douglass, per the request of @empavia to try out as a new work function for the InVEST Urban Cooling Model.

To install dependencies, `mamba install pygeoprocessing`.
"""
import pygeoprocessing
import sys
import os
import numpy as np

def calculate_work_intensity(target_raster, wbgt_raster, alpha1, alpha2):
  """Calculate work intensity with the Hothaps function.
  
  Args:
      target_raster (str): The path to the new work intensity raster being created.
      wbgt_raster (str): The path to an already-calulated Wet Bulb Globe Temp raster.
      alpha1 (float): The alpha1 parameter.
      alpha2 (float): The alpha2 paramter.
  
  Returns:
      ``None``.
  """
  def _calculate_workability(wbgt):
    return 0.1 + (0.9/(1+((wbgt/alpha1)**alpha2)))
  
  pygeoprocessing.raster_map(_calculate_workability, [wbgt_raster], target_raster)



# To loop over datasets
path = "G:\\Shared drives\\Wellcome Trust Project Data\\2_postprocess_intermediate\\UCM_official_runs"

# Set the standard variables
alpha1 = 30.94
alpha2 = 16.64
variables = np.array([[22.0, 2.0, 55.0], [22.0, 5.0, 55.0], [25.0, 2.0, 45.0],
                      [25.0, 5.0, 45.0]])
#variables = np.array([[28, 2, 45], [28, 5, 45]]) # these are separate as they don't have decimals in the main files
for temp, uhi, hum in variables:
  # set the temperatures and UHI values in the inputs and outputs
  target_raster = os.path.join(path, f'new_work_intensity/scenario0/high_work_productivity_{temp}deg_{uhi}uhi_{hum}hum.tif')
  wbgt_raster = os.path.join(path, f'scenario0/work_and_energy_runs/intermediate/wbgt_london_scenario_{temp}deg_{uhi}uhi_{hum}hum_energy_productivity.tif')
  # run the new calculation in a loop
  calculate_work_intensity(target_raster, wbgt_raster, alpha1, alpha2)
