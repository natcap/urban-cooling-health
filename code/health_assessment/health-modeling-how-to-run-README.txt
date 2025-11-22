
Heat-Attributable Excess Deaths (July)
========================================================

This package includes:
- health-modeling.py : main analysis script
- (You can add your own data paths and run from the command line.)

Quick start:
1) Prepare rasters on the same urban grid (or use --align_to to force alignment):
   - T_baseline.tif (degC)
   - T_scenario.tif (degC)
   - pop_baseline.tif (persons per pixel)
   - pop_scenario.tif (persons per pixel)
   - mort_case_baseline_cardio.tif (deaths/person/year)
   - mort_case_baseline_resp.tif (deaths/person/year)
   - mort_case_baseline_cere.tif (deaths/person/year)

2) Run:

In `VS Code` / Terminal

```

conda activate geo_env

cd D:\natcap\urban-cooling-health\code\health_assessment

.\health-modeling_s0_s1_2050_2050.bat

```

Outputs:
- deltaT_degC.tif
- AF_{cause}.tif
- Excess_{cause}.tif
- city_totals_deterministic.csv
- city_totals_monte_carlo.csv (if --n_draws > 0)

Notes:
- Mortality rates should be annual (deaths/person/year). If you use July-only rates,
  scale consistently across all inputs.
- If your exposure-response is non-linear or thresholded, replace the log-linear
  RR(x,y) = exp(beta * dT) with your function.
