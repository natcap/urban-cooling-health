# Project TODO and reproduction roadmap

Last reviewed: 9 September 2026

This document tracks the remaining work needed to make the analysis portable,
fully reproducible and ready for publication. Start with the **P0** items: they
affect the validity or interpretation of the Green30–Target30 comparison.

## Priority guide

- **P0 — blocking:** complete before reporting final Figure 7 results.
- **P1 — important:** complete before manuscript submission or public release.
- **P2 — improvement:** improves maintenance, portability or onboarding.

## P0 — complete before final Figure 7 reporting

### 1. Add the 2021 LSOA population lookup

- [ ] Obtain total usual-resident population for the same LSOA geography used
  by `figures/equity_map_biscale/health_sf.rds`.
- [ ] Retain the official LSOA code in the source file and document its source,
  geography vintage, download date and licence.
- [ ] Create `data/derived/lsoa_population_2021.csv` with exactly:

  ```csv
  id,population_2021
  1,VALUE_FOR_ID_1
  2,VALUE_FOR_ID_2
  ```

- [ ] Confirm 4,835 unique IDs, with no missing, zero or negative population.
- [ ] Run `Rscript code/run-fig7.R` and confirm that
  `fig7_run_metadata.csv` reports a population source and the benefit unit is
  deaths averted per 100,000 residents.

**Data to add:** the derived lookup above. Also retain, outside Git if needed,
the original official population table and the reproducible crosswalk used to
map its LSOA code to the current numeric `id`. See
[`data/derived/README.md`](data/derived/README.md).

**Done when:** all Figure 7 rows have a valid 2021 population denominator and
the population-normalized tables and figures have been visually checked.

### 2. Resolve the targeted-scenario naming mismatch

The active scenario generator uses `710v3`, `720v3` and `730v3`, while the UCM,
health and plotting workflows use `510`, `520` and `530`.

- [ ] Decide which raster set is the approved Target10/20/30 version.
- [ ] Record a single mapping from manuscript label to raster filename and run
  directory.
- [ ] Update the three `tree_equity_*` scripts, UCM scripts, health batch files
  and `code/func_colors.R` to use that mapping consistently.
- [ ] Archive or clearly label superseded scenario rasters so they cannot be
  selected accidentally.

**Done when:** a repository-wide search finds no active conflicting Target
scenario names, and one documented input raster maps to each Target scenario.

### 3. Verify equal realized canopy intervention

- [ ] Locate the exact baseline, Green30 and approved Target30 LULC rasters used
  for the final UCM runs.
- [ ] Run:

  ```bash
  python code/lc_scenarios/validate_scenario_canopy_budget.py \
    --baseline /path/to/baseline_lulc.tif \
    --green30 /path/to/green30_lulc.tif \
    --target30 /path/to/target30_lulc.tif \
    --output figures/equity_map_biscale/fig7_canopy_budget_check.csv
  ```

- [ ] If the default 0.5% tolerance fails, adjust the Target30 allocation,
  regenerate its raster and repeat the check before rerunning downstream models.

**Data to locate or add:** the three final LULC rasters. Do not substitute
intermediate rasters with similar names.

**Done when:** `fig7_canopy_budget_check.csv` records a pass and identifies the
three raster files used.

### 4. Regenerate paired citywide uncertainty

- [ ] Rerun Green30 and Target30 health models with the same inputs, 2,000 or
  more draws, and seed `20260908`.
- [ ] Export the paired draw tables as:
  - `data/derived/green30_city_total_draws_by_cause.csv`
  - `data/derived/target30_city_total_draws_by_cause.csv`
- [ ] Each file must contain `draw` and `all_cause`, with matching unique draw
  IDs. Follow the sign convention documented in
  [`data/derived/README.md`](data/derived/README.md).
- [ ] Rerun Figure 7 and review `fig7_citywide_uncertainty.csv`.

**Done when:** the workflow finds all paired draws and the manuscript states
that this interval covers exposure-response uncertainty while holding
temperature, 2021 population and 2021 mortality fixed.

## P1 — complete before submission or public release

### 5. Replace row-order IDs with official geography codes

- [ ] Carry `LSOA21CD` or the approved official LSOA identifier through zonal
  statistics, vulnerability processing and `health_sf.rds`.
- [ ] Assert one-to-one joins and explicitly report unmatched or duplicate IDs.
- [ ] Rebuild the population lookup using the official code rather than row
  order.

**Done when:** rerunning an upstream step or sorting rows cannot change any
population, vulnerability or health-result join.

### 6. Centralize paths and scenario settings

- [ ] Move shared-drive roots, scenario filenames, climate settings, model
  parameters, random seeds and output roots into one version-controlled config
  file.
- [ ] Replace active machine-specific paths in Python, R, notebooks and batch
  files with values read from that config or explicit command-line arguments.
- [ ] Provide an example config containing no credentials or private paths.

**Done when:** a new user can configure the project without editing analysis
code and can see the full run configuration beside each result set.

### 7. Record a complete input manifest

- [ ] Create a manifest for every raw and intermediate input containing source,
  version/date, geography vintage, units, CRS, resolution, licence, checksum and
  preprocessing history.
- [ ] Mark each input as public, restricted or generated.
- [ ] Record which exact input snapshot produced every manuscript figure.

**Done when:** another authorized researcher can identify the exact files needed
for method reproduction and historical-result reproduction.

### 8. Synchronize manuscript claims and generated evidence

- [ ] Update Figure 7 caption, Results, Methods and abstract using only the
  regenerated population-normalized outputs.
- [ ] State “2050 temperature scenarios evaluated under 2021 population and
  mortality” wherever the temporal design could be misunderstood.
- [ ] Distinguish fixed-threshold paired comparisons, LSOA bootstrap sensitivity
  intervals and exposure-response Monte Carlo uncertainty.
- [ ] Cross-check every reported number against its saved CSV.

**Done when:** figures, captions, tables and manuscript text use the same
scenario version, units and values.

## P2 — improve portability and maintenance

### 9. Lock software environments

- [ ] Add a Python environment file with the validated InVEST and geospatial
  package versions.
- [ ] Add an R lockfile, preferably with `renv`, for the production analyses.
- [ ] Record operating-system requirements for the Windows health batch workflow.

### 10. Add automated checks

- [ ] Add lightweight tests for scenario-label mapping, raster compatibility,
  one-to-one LSOA joins, complete population coverage and paired Monte Carlo
  draws.
- [ ] Add a continuous-integration check for Markdown links and parseable R and
  Python production scripts that do not require restricted data.

### 11. Convert notebooks and legacy scripts into explicit stages

- [ ] Identify the authoritative notebook or script for every workflow stage.
- [ ] Move obsolete alternatives to an archive directory with a short reason.
- [ ] Clear misleading saved notebook output and document expected inputs and
  outputs at the top of each active notebook.

### 12. Add citation and data-access guidance

- [ ] Add the manuscript citation or preprint DOI when available.
- [ ] Explain how qualified collaborators can request restricted input data.
- [ ] Confirm that all committed derived data can be redistributed under their
  source licences.

## Recommended next run order

Complete the remaining work in this order to avoid unnecessary model reruns:

1. approve one Target10/20/30 scenario version;
2. add the official-code crosswalk and 2021 LSOA population;
3. validate Green30–Target30 realized canopy equivalence;
4. rerun UCM only if the approved or adjusted LULC rasters differ from the
   existing final inputs;
5. rerun Green30 and Target30 health models with paired Monte Carlo settings;
6. regenerate zonal outputs and `health_sf.rds` using official LSOA codes;
7. run `Rscript code/run-fig7.R`;
8. inspect figures and reconcile every manuscript number with the output CSVs;
9. save the input manifest, run metadata and software environment with the
   publication archive.

## Maintaining this list

When completing an item, check it off and add a link to the resulting file,
commit or archived evidence. Add newly discovered blocking issues under P0
rather than leaving them only in code comments or email discussions.
