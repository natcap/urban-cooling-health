# Small derived inputs for Figure 7

The project deliberately uses 2021 population and 2021 baseline mortality for
all land-cover and 2050 temperature scenarios. These files provide the small
denominators and uncertainty outputs needed to reproduce Figure 7; large source
rasters remain on the shared drive.

## Population lookup

The tracked `lsoa_population_2021.csv` has exactly one row for every LSOA in
`figures/equity_map_biscale/health_sf.rds` and these columns:

```text
id,population_2021
1,<2021 population for id 1>
2,<2021 population for id 2>
```

It is reproduced with `code/health_assessment/prepare_lsoa_population_2021.R`.
Its manifest records input and output checksums, package versions, validation
totals and the eight legacy self-intersecting polygons repaired with
`sf::st_make_valid`. Current validation results are:

- 4,835 unique IDs, all positive and non-missing;
- LSOA sum: 8,832,324.67;
- count-preserved raster sum: 8,834,023.32 (LSOA allocation difference -0.019%);
- ONS Census 2021 TS001 London total: 8,799,776 (WorldPop difference +0.370%).

Requirements for any regenerated version:

- `id` must match the existing numeric `id` in `health_sf.rds`.
- `population_2021` must be the total usual-resident population, not density,
  and must be positive and non-missing.
- The table must contain 4,835 unique IDs for the current London dataset.
- Preserve the source name, release date, geography vintage and aggregation
  method in a separate metadata note when creating the file.

The current numeric ID was created from row order and represents 4,835 London
LSOA 2011 areas. The official Census 2021 TS001 table uses 4,994 London LSOA
2021 areas and must not be joined directly. For the current analysis, aggregate
the count-preserved WorldPop 2021 raster over the exact geometries stored in
`health_sf.rds`, while retaining the numeric ID. Label the resulting values as
modeled WorldPop estimates and validate their London total against TS001.

The full data build now restores `LSOA11CD` with a verified one-to-one centroid
crosswalk. The maximum centroid difference is 0.0195 m. The relevant files are:

- `svi_lsoa11_crosswalk_nodata_harmonized.csv` — numeric legacy `id` to official `LSOA11CD`;
- `fig7_vulnerability_lsoa11_nodata_harmonized.gpkg` — the scenario-independent vulnerability
  geometry and attributes used by the production Figure 7 workflow;
- `health_lsoa_invest3202_population_weighted_2021_nodata_harmonized.csv` — all five causes,
  four scenario/temperature combinations and 4,835 LSOAs; and
- `health_lsoa_invest3202_population_weighted_2021_nodata_harmonized.manifest.json` — checksums,
  software, crosswalk QA and boundary-edge allocation evidence.

## Optional Monte Carlo files

The completed Green30 and Target30 runs use the same seed and 2,000 draws. Their
25°C draw tables are copied from the versioned health workspace as:

- `green30_nodata_harmonized_city_total_draws_by_cause.csv`
- `target30_nodata_harmonized_city_total_draws_by_cause.csv`

Each file must contain `draw` and `all_cause`. The production analysis pairs
matching draw numbers and reports uncertainty in the citywide difference. This
captures exposure-response uncertainty only; it does not propagate uncertainty
in temperature, population or baseline mortality.
