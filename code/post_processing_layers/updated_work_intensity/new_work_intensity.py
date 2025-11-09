"""Calculate work intensity with the Hothaps function.

Written by James Douglass, per the request of @empavia to try out as a new work function for the InVEST Urban Cooling Model.

To install dependencies, `mamba install pygeoprocessing`.
"""
import pygeoprocessing
import sys
import os

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


if __name__ == '__main__':
  calculate_work_intensity(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])


# To test
path = "G:\\Shared drives\\Wellcome Trust Project Data\\2_postprocess_intermediate\\UCM_official_runs"

target_raster = os.path.join(path, 'new_work_intensity/scenario1/high_work_productivity_25deg_5uhi_45hum.tif')
wbgt_raster = os.path.join(path, 'scenario1/work_and_energy_runs/intermediate/wbgt_london_scenario_25.0deg_5.0uhi_45.0hum_energy_productivity.tif')
alpha1 = 30.94
alpha2 = 16.64

# run
calculate_work_intensity(target_raster, wbgt_raster, alpha1, alpha2)