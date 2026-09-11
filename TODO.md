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

### 1. Add 2021 population inputs and population-weighted mortality

- [x] Approve population-weighted allocation of borough mortality counts using
  the fixed 2021 population raster.
- [x] Add and unit-test
  `code/health_assessment/prepare_mortality_population_weighted.py` while
  retaining the area-weighted workflow for comparison.
- [x] Add a validated converter from the existing Nomis TSV export to
  `mortality_2021_long.csv`; this CSV is a generated intermediate, not a new
  data source.
- [x] Audit the archived Nomis exports. Use `387056911188607_data.tsv`, which
  contains the required broad circulatory group `I00-I99`; the newer `842...`
  export contains narrower cardiovascular subgroups and is not equivalent.
- [x] Document the mortality-source choice, fixed 2021 population,
  population-weighted allocation rationale, assumptions and limitations in
  `code/health_assessment/DATA_SELECTION_AND_METHOD_DECISIONS.md`.
- [x] Identify the legacy population-preprocessing error: bilinear reprojection
  of population counts reduced the London total to 4.74 million.
- [x] Generate and QA a count-preserved EPSG:27700 population raster directly
  from the raw WorldPop input. Its London total is 8,834,023, within 0.387% of
  the published 2021 Census benchmark of 8.8 million.
- [x] Generate five revised population-weighted mortality rasters in the
  separate `health_rasters_population_weighted_2021_count_preserved` folder.
  All borough/cause totals are conserved and all four 25°C/28°C health input
  sets pass exact-grid validation.
- [x] Compare the legacy and revised allocations and deterministic 25°C health
  results. See `code/health_assessment/HEALTH_VERSION_COMPARISON.md` and the
  machine-readable comparison tables named there.
- [x] After code review and commit, rerun the LSOA population preparation from
  a clean worktree. The official-code manifest records commit `6414f7a` and
  `git_worktree_dirty: false`.
- [x] Record the supplied borough and population paths, preserve the flawed
  processed raster for audit, and add a reproducible count-preserving
  reprojection from the raw WorldPop input without changing either source.
- [x] Identify the official WorldPop source as listing 135, UK 2021 item 75899,
  100 m constrained R2025A v1, DOI `10.5258/SOTON/WP00839`.
- [x] Compare a fresh official WorldPop download with the existing raster. The
  official file itself contains stale `R2024B v1` TIFF tags, while listing 135
  and item 75899 identify it as R2025A v1. The two files have identical grids,
  profiles, totals and pixel values, so this is an upstream metadata issue and
  the existing analytical input remains valid.
- [x] Add the strict, windowed `health_modeling_v2.py` runner and a
  version-controlled example configuration without enabling an unresolved
  Target scenario.
- [x] Check official small-area mortality availability. The current Nomis query
  interface does not expose a 2021 cause-specific LSOA option; continue with
  cause-specific borough counts and population-weighted allocation unless a
  compatible archived or bespoke ONS table is obtained.
- [x] Download the official Census 2021 TS001 bulk archive. It contains 4,994
  London LSOA21 rows totaling 8,799,776 usual residents.
- [x] Confirm the recommended geography decision: retain the 4,835 LSOA11
  Figure 7 areas and aggregate the count-preserved WorldPop 2021 raster to
  those polygons. Do not directly join the 4,994-row LSOA21 TS001 table.
- [x] Obtain modeled 2021 population for the exact 4,835-area production
  LSOA11 geography.
- [x] Restore `LSOA11CD` through a verified one-to-one crosswalk and remove the
  legacy row-order ID from all production Figure 7 joins.
- [x] Create `data/derived/lsoa_population_2021_by_lsoa11cd.csv` with exactly:

  ```csv
  LSOA11CD,LSOA11NM,population_2021
  E01000001,City of London 001A,VALUE_FOR_E01000001
  ```

- [x] Confirm 4,835 unique official codes, with no missing, zero or negative
  population.
- [x] Run `Rscript code/run-fig7.R` and confirm that
  `fig7_run_metadata.csv` reports a population source and the benefit unit is
  deaths averted per 100,000 residents.

**Result:** the WorldPop-derived LSOA total is 8,832,324.67, 0.370% above the
TS001 London total. The manifest records the exact input checksums and the eight
LSOA11 codes repaired before zonal aggregation. Retain TS001 as an external
validation reference; do not construct a direct LSOA21-to-LSOA11 crosswalk.
See
[`data/derived/README.md`](data/derived/README.md).

**Done when:** all Figure 7 rows have a valid 2021 population denominator and
the population-normalized tables and figures have been visually checked.

### 2. Resolve the targeted-scenario naming mismatch

The manuscript-facing files are now traced to their selected trials by SHA-256:
`510` is a copy of `710v2`, `520` is a copy of `730v2`, and `530` is a copy of
`730v3`. The Target20 mapping is intentionally non-sequential. These identity
checks resolve the naming history, but do not establish equal canopy budgets.
The active generator's `710v3/720v3/730v3` defaults do not reproduce the
confirmed Target10/20 selections and must not be used implicitly.

The exact file references and the two available 25°C and 28°C comparison sets
are recorded in [`code/health_assessment/SCENARIO_NAMING_AUDIT.md`](code/health_assessment/SCENARIO_NAMING_AUDIT.md).

- [x] Confirm all three historical copy/alias pairs by checksum.
- [x] Record the approved historical Target10/20 selections and explain the
  non-sequential Target20 source name.
- [x] Record a single manuscript-label-to-raster mapping and canopy audit.
- [ ] Update the three `tree_equity_*` scripts, UCM scripts, health batch files
  and `code/func_colors.R` after the equal-budget Target10/20 rasters are
  approved; do not redirect them to a provisional input.
- [ ] Archive or clearly label superseded scenario rasters so they cannot be
  selected accidentally.

**Done when:** a repository-wide search finds no active conflicting Target
scenario names, and one documented input raster maps to each Target scenario.

### 3. Rebuild the equal realized-canopy intervention

- [x] Complete the original code-100 transition audit. It found 930,000 Green30
  and 869,444 legacy Target30 transitions to code 100.
- [x] Check the available Target30 variants. None matches Green30: `730` adds
  1,201,968 pixels, `730v2` adds 542,985, and `530`/`730v3` add 869,444.
- [x] Generate Target30 v4 to match 930,000 code-100 transitions. Retain this as
  historical audit evidence rather than the final equal-canopy result.
- [x] Run:

  ```bash
  python code/lc_scenarios/validate_scenario_canopy_budget.py \
    --baseline /path/to/baseline_lulc.tif \
    --green30 /path/to/green30_lulc.tif \
    --target30 /path/to/target30_lulc.tif \
    --output figures/equity_map_biscale/fig7_canopy_budget_check.csv
  ```

- [x] Identify the limitation of the original audit: it treated only code 100
  as canopy and allowed transitions from all valid source codes.
- [x] Approve the revised definition: existing canopy `{1,2,100}`, eligible
  planting `{4,20,21}`, replacement code `100`, and `all_touched=False`.
- [x] Recover and document the historical Green Scenario Generator logs. They
  confirm InVEST 3.14.1, areas of 3,200/6,200/9,300 ha, focal and convertible
  codes `1 2 4 20 21`, replacement code `100`, nearest-to-edge and two steps.
- [x] Implement and test `generate_green_scenarios_invest.py` under InVEST
  3.20.2. The revised test outputs contain exactly 307,768/595,084/894,249
  eligible cells, no ineligible transitions, matching NoData, and full nesting.
- [x] Run a historical-parameter version check. InVEST 3.20.2 preserves the
  exact area totals but differs from the archived 3.14.1 outputs in
  9,594/18,970/23,276 full-raster cells for Green10/20/30.
- [x] Test historical Green nesting. Green10 has 4,451 converted cells absent
  from Green20; Green20 has 11,709 absent from Green30.
- [x] Audit Target10 and Target20. Target10 adds 276,430 pixels versus
  Green10's 320,000 (13.616% short); Target20 adds 542,985 versus Green20's
  620,000 (12.422% short).
- [x] Review the supplied Target `Methods.docx` and supporting folder. Document
  the street classes, 5 m spacing, canopy/building exclusions, UTCI-SVI rank
  method and remaining provenance gaps in
  `code/lc_scenarios/TARGET_OPPORTUNITY_METHOD_AUDIT.md`.
- [x] Build the reproducible Target opportunity mask from 3,378,103 screened
  points. It contains 158.4883 km² on `{4,20,21}`, including 97.0168 km² on
  Urban; store its raster and JSON audit in `tree_opportunity_mask/`
  `revised_v1_2026-09-10/`.
- [x] Generate the equal-area no-Urban Green sensitivity with `{4,21}`. It
  retains the 30.7768/59.5084/89.4249 km² budgets and is stored separately
  under `lc_green_scenarios_output/`.
- [x] Add `validate_green_target_scenarios.py` to validate configurable canopy
  and source-code sets, full transitions, equal budgets, nesting and NoData.
- [x] Run the tested revised Green generator into the permanent versioned
  shared folder `lc_green_scenarios_output/`
  `revised_v2_invest_3.20.2_2026-09-10/`; retain its three rasters, raw InVEST
  workspaces and JSON manifest.
- [x] Replace rank-only Target selection with deterministic `(rank, FID)`
  ordering. All three historical cutoffs intersect tied-score groups, so
  `nsmallest(N, "rank")` alone is not a reproducible prefix rule.
- [x] Generate all three Target rasters from the baseline, restricted to
  `{4,20,21}`, and stop at the corresponding corrected Green count.
- [x] Require exact counts, zero ineligible transitions, unchanged NoData,
  identical grids and nested 10-within-20-within-30 masks before UCM runs.

**Data to locate or add:** no additional source raster is needed for the Green
regeneration. The full Target ranking has now been located at
`My Drive/NatCap/projects/KCL_Welcome/london-equity-tree-scenario/Results/`
`Potential_Tree_Points_Ranked.shp` (including its `.dbf`, `.shx`, `.prj` and
`.cpg` sidecars). Before Target regeneration, validate that its `rank` field,
CRS, feature count and first-ranked geometries reproduce the approved trial
vectors. Do not substitute intermediate rasters or similarly named vectors.
The supplied Methods record does not identify the exact building file, OSM
download/version, 1 m canopy raster filename, or source-generation script;
retain these as provenance requests rather than blockers to regeneration from
the final screened points.

**Current result:** all six revised Green/Target rasters pass the scenario
gate. Each pair adds exactly 307,768/595,084/894,249 eligible cells, all tiers
are nested, and there are no ineligible transitions or NoData differences.
The prior Target30 v4, UCM and health reruns remain historical reproducibility
evidence and must be superseded by the revised outputs.

**Done when:** the audit records a pass and identifies the three final rasters.

### 4. Regenerate paired citywide uncertainty

Use 25°C as the primary manuscript setting and 28°C as sensitivity.

- [x] Add a dedicated UCM runner for `LULC_Scenario730v4_equal_budget.tif`
  that writes to a separate workspace and covers both temperature settings.
- [x] Create and verify the production `urban-cooling-invest-3.20.2`
  environment. Retain 3.14.1 only for optional historical comparison.
- [x] Validate and run Target30 v4 with InVEST 3.20.2 at both temperature
  settings in a versioned, health-only workspace.
- [x] Repeat Target30 v4 from clean commit `be4433c` in a publication workspace.
  Both 25 C and 28 C rasters are byte-identical to the earlier 3.20.2 outputs;
  the new manifests record the enforced model version and clean worktree.
- [x] Diagnose the 3.14.1-to-3.20.2 temperature difference. The dominant cause
  is the park Cooling Capacity bug fixed in InVEST 3.15.0; see
  `INVEST_VERSION_COMPARISON.md` for the function audit and decomposition.
- [x] Regenerate the no-intervention baseline and Green30 at 25 C and 28 C
  with InVEST 3.20.2. Do not compare the 3.20.2 Target30 output against legacy
  3.14.1 temperatures; see `INVEST_VERSION_COMPARISON.md`.
- [x] Rerun Green30 and Target30 health models with the same inputs, 2,000 or
  more draws, and seed `20260908`.
- [x] Export the paired draw tables as:
  - `data/derived/green30_nodata_harmonized_city_total_draws_by_cause.csv`
  - `data/derived/target30_nodata_harmonized_city_total_draws_by_cause.csv`
- [x] Each file contains `draw` and `all_cause`, with matching unique draw
  IDs. Follow the sign convention documented in
  [`data/derived/README.md`](data/derived/README.md).
- [x] Rerun Figure 7 and review `fig7_citywide_uncertainty.csv`.

**Done when:** the workflow finds all paired draws and the manuscript states
that this interval covers exposure-response uncertainty while holding
temperature, 2021 population and 2021 mortality fixed.

### 4a. Resolve the Green30 edge-mask sensitivity

- [x] Create a separate Green30 LULC copy whose NoData encoding is harmonized
  with baseline/Target30 (`0` instead of `255`) without changing any valid land
  class or canopy count.
- [x] Rerun Green30 at 25°C and 28°C in a new workspace and confirm that its
  62 missing populated cells (18.197 modeled residents; 99.999565% current
  coverage) are restored.
- [x] Compare city and LSOA results with the previous output and adopt the
  harmonized run as production without overwriting the earlier workspaces.

**Result:** coverage increased from 99.999565% to 100%. Green30 became 0.06163°C
warmer on average over the common valid domain and its all-cause deaths averted
changed from 391.8171 to 367.2235 (-6.28%). Figure 7 and paired uncertainty were
regenerated from the harmonized result.

**Done:** Green30 and Target30 now have complete, identical populated-cell
coverage and the NoData transformation is checksum-documented.

## P1 — complete before submission or public release

### 5. Replace row-order IDs with official geography codes

- [x] Carry `LSOA11CD` through the production vulnerability and zonal-statistics
  inputs. Figure 7 now reads these sources directly instead of rebuilding the
  historical `health_sf.rds`.
- [x] Assert one-to-one joins and explicitly report unmatched or duplicate IDs.
- [x] Rebuild the population lookup using `LSOA11CD` rather than row order.
  The 4,835 official-code values are exactly identical to the previous lookup,
  and Figure 7 now joins population, health and vulnerability by `LSOA11CD`.

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

- [x] Add `code/health_assessment/environment-health.yml` for the revised health
  workflow. Pin exact package builds after the first successful project-data
  run.
- [ ] Add an R lockfile, preferably with `renv`, for the production analyses.
- [ ] Record operating-system requirements for the Windows health batch workflow.

### 10. Add automated checks

- [x] Add initial unit tests for population allocation, borough-total
  conservation, temperature-effect direction and reproducible cause-specific
  random draws.
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
- [x] Audit licences for the revised Figure 7 inputs and list the exact
  remaining questions in `DATA_LICENCE_AND_REDISTRIBUTION.md`.
- [ ] Provide the UKCEH LCM2023 licence/order record and confirm whether the
  intended aggregate outputs and figures may be publicly redistributed.
- [ ] Record the original publisher, URL and licence for
  `data/London_Ward_aoi.*` and for the two tracked MIDAS weather RDS extracts.
- [x] Add WorldPop, Nomis/ONS and GLA/Bloomberg attribution text to the
  derived-data README.
- [ ] Add those attributions to the manuscript and replace the bracketed
  ONS/OS boundary copyright year before public release.

## Recommended next run order

Complete the remaining work in this order to avoid unnecessary model reruns:

1. [x] generate and validate revised Target10/20/30 scenario versions;
2. [x] validate all fourteen revised UCM configurations at 25 C and 28 C;
3. [x] add the official-code crosswalk and 2021 LSOA population;
4. [x] run the fourteen revised baseline/Green/Target UCM configurations with InVEST 3.20.2;
5. [ ] rerun the twelve revised health models with paired Monte Carlo settings
   (all input/grid preflight checks have passed);
6. [ ] regenerate LSOA zonal outputs using official `LSOA11CD` (Figure 7 now
   reads this table directly, superseding regeneration of `health_sf.rds`);
7. [ ] run `Rscript code/run-fig7.R` and visually inspect both output figures;
8. reconcile the revised output CSVs with the manuscript text, caption and abstract;
9. save the input manifest, run metadata and software environment with the
   publication archive.

## Maintaining this list

When completing an item, check it off and add a link to the resulting file,
commit or archived evidence. Add newly discovered blocking issues under P0
rather than leaving them only in code comments or email discussions.
