# Small derived inputs for Figure 7

The project deliberately uses 2021 population and 2021 baseline mortality for
all land-cover and 2050 temperature scenarios. These files provide the small
denominators and uncertainty outputs needed to reproduce Figure 7; large source
rasters remain on the shared drive.

## Source licences and attribution

- **WorldPop population:** Bondarenko et al. (2025), United Kingdom 2021
  population counts, R2025A v1, DOI
  [10.5258/SOTON/WP00839](https://doi.org/10.5258/SOTON/WP00839), licensed
  CC BY 4.0. The values here are modified by count-preserving reprojection and
  LSOA aggregation.
- **ONS/Nomis mortality:** Source: Office for National Statistics, licensed
  under the Open Government Licence. Values here are transformed through
  population-weighted spatial allocation and health-impact modeling.
- **GLA vulnerability variables:** Greater London Authority and Bloomberg
  Associates, [Climate Risk Mapping
  2024](https://data.london.gov.uk/dataset/climate-risk-mapping-2oxg6),
  licensed under the Open Government Licence v3. Values here are selected,
  rescored and joined to official LSOA11 codes.
- **LSOA boundaries:** Source: Office for National Statistics licensed under
  the Open Government Licence v3.0. Contains OS data © Crown copyright and
  database right **[insert the year stated with the downloaded boundary]**.

The bracketed ONS/OS year must be completed from the original boundary
download record before public release. UKCEH LCM2023 terms also require a
project-specific confirmation for the modeled outputs; see
[`DATA_LICENCE_AND_REDISTRIBUTION.md`](../../DATA_LICENCE_AND_REDISTRIBUTION.md).

## Population lookup

The production `lsoa_population_2021_by_lsoa11cd.csv` has exactly one row for
every official LSOA11 used by Figure 7 and these columns:

```text
LSOA11CD,LSOA11NM,population_2021
E01000001,City of London 001A,<2021 population>
```

It is reproduced with `code/health_assessment/prepare_lsoa_population_2021.R`.
Its manifest records the committed script, clean-worktree status, input and
output checksums, package versions, validation totals and the eight source
polygons repaired with `sf::st_make_valid`. Current validation results are:

- 4,835 unique IDs, all positive and non-missing;
- LSOA sum: 8,832,324.67;
- count-preserved raster sum: 8,834,023.32 (LSOA allocation difference -0.019%);
- ONS Census 2021 TS001 London total: 8,799,776 (WorldPop difference +0.370%).

Requirements for any regenerated version:

- `LSOA11CD` must be unique, non-missing and match the production vulnerability
  and health tables one-to-one.
- `population_2021` must be the total usual-resident population, not density,
  and must be positive and non-missing.
- The table must contain 4,835 unique official codes for the current London
  dataset.
- Preserve the source name, release date, geography vintage and aggregation
  method in a separate metadata note when creating the file.

The earlier `lsoa_population_2021.csv` is retained as historical audit evidence
but is not a production Figure 7 input because its numeric `id` originated from
row order. The official-code table contains exactly the same 4,835 population
values (`max absolute difference = 0`) after matching through the verified
crosswalk. The official Census 2021 TS001 table uses 4,994 London LSOA21 areas
and must not be joined directly. Label the WorldPop values as modeled estimates
and retain TS001 only as a London-total validation benchmark.

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
