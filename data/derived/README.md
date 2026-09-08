# Small derived inputs for Figure 7

The project deliberately uses 2021 population and 2021 baseline mortality for
all land-cover and 2050 temperature scenarios. These files provide the small
denominators and uncertainty outputs needed to reproduce Figure 7; large source
rasters remain on the shared drive.

## Required population lookup

Add `lsoa_population_2021.csv` with exactly one row for every LSOA in
`figures/equity_map_biscale/health_sf.rds` and these columns:

```text
id,population_2021
1,<2021 population for id 1>
2,<2021 population for id 2>
```

Requirements:

- `id` must match the existing numeric `id` in `health_sf.rds`.
- `population_2021` must be the total usual-resident population, not density,
  and must be positive and non-missing.
- The table must contain 4,835 unique IDs for the current London dataset.
- Preserve the source name, release date, geography vintage and aggregation
  method in a separate metadata note when creating the file.

The current numeric ID was created from row order. For the next full data build,
retain the official LSOA code in both the vulnerability layer and zonal-stat
outputs, then use that stable code for the join. Until then, verify that the
population table was derived from the exact same ordered LSOA layer.

## Optional Monte Carlo files

After rerunning Green30 and Target30 with the same seed and at least 2,000
draws, copy and rename the model outputs:

- `green30_city_total_draws_by_cause.csv`
- `target30_city_total_draws_by_cause.csv`

Each file must contain `draw` and `all_cause`. The production analysis pairs
matching draw numbers and reports uncertainty in the citywide difference. This
captures exposure-response uncertainty only; it does not propagate uncertainty
in temperature, population or baseline mortality.
