# Reproducing the Figure 7 equity analysis

## Interpretation held constant across the project

All health analyses apply 2050 temperature fields to 2021 population and 2021
baseline mortality. This isolates land-cover and climate effects. It is not a
forecast of London's demographic composition in 2050.

## Inputs

The repository contains the documented inputs needed for the current Figure 7:

- `data/derived/fig7_vulnerability_lsoa11_nodata_harmonized.gpkg`;
- `data/derived/health_lsoa_invest3202_population_weighted_2021_nodata_harmonized.csv`;
- `data/derived/lsoa_population_2021_by_lsoa11cd.csv`; and
- paired 25°C Green30 and Target30 `_nodata_harmonized_city_total_draws_by_cause.csv`
  files.

Their manifests and official-code crosswalk are in the same folder. The
historical `figures/equity_map_biscale/health_sf.rds` is retained for audit but
is no longer an analytical input to the production Figure 7 workflow.

## Run Figure 7

From any directory, run the production wrapper. It does not require Pandoc:

```text
Rscript code/run-fig7.R
```

To create the accompanying HTML notebook when Pandoc is available, run:

```r
rmarkdown::render(
  "code/equity-health-fig7-production.Rmd",
  knit_root_dir = normalizePath(".")
)
```

The workflow checks official-code and scenario pairing, duplicate IDs, missing
benefit values and population join coverage. It writes the maps, paired
comparison, headline tables, age-75+ diagnostics, paired citywide uncertainty,
run metadata and R session information to
`figures/equity_map_biscale/`.

## Verify equal intervention budgets

Run the canopy-budget validator against the baseline, Green30 and Target30 LULC
rasters:

```text
python code/lc_scenarios/validate_scenario_canopy_budget.py \
  --baseline <baseline_lulc.tif> \
  --green30 <green30_lulc.tif> \
  --target30 <target30_lulc.tif> \
  --output figures/equity_map_biscale/fig7_canopy_budget_check.csv
```

The comparison passes only when both scenarios use the same grid and their
added tree-canopy areas differ by no more than the requested tolerance. Commit
the resulting CSV as Figure 7 provenance.

## Final review checklist

1. Confirm 4,835 paired LSOAs and complete 2021 population coverage.
2. Confirm Green30 thresholds are used for both scenarios and all maps.
3. Confirm the canopy-budget validation passes.
4. Confirm both model runs used seed `20260908` and 2,000 paired draws.
5. Inspect both exported PNGs at full size for clipped labels and legends.
6. Quote the fixed-2021-population assumption in Methods and figure captions.
