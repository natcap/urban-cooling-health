# Legacy versus revised health assessment

Compared: 9 September 2026

## Scope

This comparison holds the 25°C baseline, Green30 and Target30 temperature
rasters and the cause-specific relative-risk parameters constant. It compares:

- **legacy:** borough deaths distributed uniformly by land area, followed by
  alignment to the UCM grid; and
- **revised:** borough deaths distributed by count-preserved 2021 WorldPop
  weights directly on the UCM grid.

The first comparison below records the deterministic QA stage. A final section
now reports the approved equal-budget Target30 v4 rerun and paired Monte Carlo
analysis.

## Population preprocessing finding

The raw WorldPop raster totals 67.403 million people across Great Britain. The
legacy preprocessing notebook reprojected pixel counts with bilinear
resampling; the resulting GB raster totals only 35.218 million and its London
clip totals 4.743 million. Population is an extensive quantity, so direct
bilinear interpolation did not conserve counts.

The corrected workflow uses GDAL sum resampling from the raw geographic raster
to the 10 m UCM grid and then applies the London borough mask. It gives
8,834,023 people, 0.387% above the
[2021 Census London benchmark of 8.8 million](https://data.london.gov.uk/download/24fd8ec4-c65a-4c17-947b-e6f8c40d2c11/18e4a444-11a7-4777-b383-8cf199176157/2021%20census%20first%20release.pdf).
The previous processed raster and the first provisional mortality allocation
are retained for audit but must not be used in production.

## Mortality-allocation comparison

Across the five causes:

- the legacy aligned rasters place 6.27% to 8.05% of deaths on pixels with zero
  population; the revised rasters place 0%;
- correlation between pixel population and allocated deaths increases from
  0.316–0.454 to 0.815–0.968; and
- the share of deaths located in the most-populated pixel decile increases from
  11.45–16.73% to 25.20–34.09%.

The legacy alignment also loses 1.29–1.80% of the cause totals before the
temperature-coverage mask is applied. The revised rasters preserve every
borough/cause total to numerical precision.

## Deterministic 25°C health comparison

| Scenario | Cause | Legacy deaths averted | Revised deaths averted | Change |
|---|---|---:|---:|---:|
| Green30 | All cause | 379.37 | 391.82 | +3.28% |
| Green30 | Cardiovascular | 148.11 | 152.77 | +3.15% |
| Green30 | Respiratory | 56.94 | 58.77 | +3.21% |
| Target30 | All cause | 386.36 | 480.91 | +24.47% |
| Target30 | Cardiovascular | 151.34 | 188.10 | +24.30% |
| Target30 | Respiratory | 57.54 | 71.99 | +25.10% |

All-cause Target30 minus Green30 increases from 6.99 deaths averted under the
legacy workflow to 89.09 under the revised workflow. Expressed relative to
Green30, the Target30 advantage changes from 1.84% to 22.74%.

## Interpretation

Green30 changes relatively little because its cooling is broadly distributed.
Target30 changes substantially because the population-weighted mortality burden
is more concentrated in inhabited pixels, which overlap more strongly with the
targeted cooling pattern. This result is consistent with the intended equity
mechanism. The later equal-budget, paired-uncertainty and official-code LSOA
checks described below supersede the provisional v3 comparison.

The legacy canopy-budget audit failed because Target30 v3 adds 6.511% less
realized canopy than Green30. Target30 v4 was calibrated to the exact Green30
budget and passes the raster audit. The calculations shown above are retained
only as a v3 diagnostic; the final v4 results in the next section replace them.
A clean InVEST 3.20.2 publication rerun reproduced the Target30 v4 temperature
rasters byte for byte, so another health rerun was not required.

The difference is not evidence that total London mortality increased. The
observed borough totals are unchanged; only their modeled within-borough
locations and grid coverage changed.

## Final equal-budget Target30 v4 result

The production rerun uses InVEST 3.20.2, the approved 930,000-pixel Target30 v4
canopy budget, fixed 2021 population and mortality, and the same 25°C baseline
as Green30.

- After harmonizing Green30's NoData sentinel with baseline and Target30,
  Green30 yields 367.2235 all-cause deaths averted. This is 24.5936 fewer
  deaths (-6.28%) than the otherwise identical `255`-NoData run.
- Target30 increases from the provisional v3 result of 480.9062 to 510.3249
  deaths averted: +29.4187 deaths, or +6.12%.
- The final Target30 advantage over harmonized Green30 is 143.1014 deaths
  (+38.97%).
- Across 2,000 paired exposure-response draws, the mean Target30-minus-Green30
  advantage is 143.53 deaths, with a 95% interval of 100.16 to 185.22.

The 28°C sensitivity run gives effectively the same scenario differences
because changing the reference temperature shifts baseline and scenario fields
together; the health model uses their pixelwise difference.

The sensitivity check confirmed that the source Green30 LULC's NoData value of
255 affected InVEST's convolution, although all valid land-cover values were
unchanged. Recoding only NoData to the baseline/Target30 value of 0 restored 62
populated edge cells and made coverage 100%. It also made Green30 temperatures
0.06163°C warmer on average across their previously shared valid domain, so the
harmonized output is now the recommended production comparison. The original
and sensitivity workspaces are both retained for audit.

## Reproducible outputs

The machine-readable comparison tables are saved outside Git under:

```text
2_postprocess_intermediate/UCM_official_runs/health_v2/version_comparison_25c/
├── mortality_allocation_comparison.csv
├── health_impact_comparison_25c.csv
└── all_cause_scenario_comparison_25c.csv
```

They are generated by `compare_health_versions.py`. Final InVEST grid and
temperature comparisons are tracked in
`code/Urban_Cooling_Modeling_Runs/invest_ucm_health_grid_audit.csv` and
`invest_ucm_health_rerun_comparison.csv`; final Figure 7 totals and uncertainty
are under `figures/equity_map_biscale/`.
