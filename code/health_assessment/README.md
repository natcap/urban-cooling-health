
Heat-Attributable Excess Deaths (July)
========================================================

Study-design note
-----------------

The project intentionally uses 2021 population and 2021 baseline mortality for
all analyses, including simulations using 2050 temperature fields. This holds
demography constant so the comparison isolates climate and land-cover effects.
Describe these results as "2050 temperature scenarios under 2021 population
and mortality," not as demographic projections for 2050.

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

.\health-modeling_s0_s530_2050_2050.bat

```

Outputs:
- deltaT_degC.tif
- AF_{cause}.tif
- Excess_{cause}.tif
- city_totals_deterministic.csv
- city_totals_monte_carlo.csv (if --n_draws > 0)
- city_total_draws_by_cause.csv (paired scenario comparison input)

For Green30 versus Target30, use at least 2,000 draws and the same explicit
random seed in both batch files. Matching draw numbers can then be compared
pairwise in `code/equity-health-fig7-production.Rmd`.

Notes:
- Mortality rates should be annual (deaths/person/year). If you use July-only rates,
  scale consistently across all inputs.
- If your exposure-response is non-linear or thresholded, replace the log-linear
  RR(x,y) = exp(beta * dT) with your function.
