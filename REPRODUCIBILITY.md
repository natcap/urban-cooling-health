# Reproducing the Figure 7 equity analysis

## Interpretation held constant across the project

All health analyses apply 2050 temperature fields to 2021 population and 2021
baseline mortality. This isolates land-cover and climate effects. It is not a
forecast of London's demographic composition in 2050.

## Inputs

The repository contains the paired LSOA analysis object:

- `figures/equity_map_biscale/health_sf.rds`

For publication-ready population-normalized results, add:

- `data/derived/lsoa_population_2021.csv`

See `data/derived/README.md` for its exact schema and validation requirements.

For citywide exposure-response uncertainty, rerun the Green30 and Target30
health-model batch files and add the two draw files described there.

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

The workflow checks scenario pairing, duplicate IDs, missing benefit values and
population join coverage. It writes the maps, paired comparison, headline
tables, age-75+ diagnostics, run metadata and R session information to
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
4. Confirm both model runs used the same random seed and draw count.
5. Inspect both exported PNGs at full size for clipped labels and legends.
6. Quote the fixed-2021-population assumption in Methods and figure captions.
