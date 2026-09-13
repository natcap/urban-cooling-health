# UCM post-processing

This folder converts Urban Cooling Model outputs into the borough and citywide
temperature, building-energy and work-productivity quantities used in the
manuscript. The production workflow is command-line driven and writes new,
versioned outputs; the older notebooks and R Markdown files remain as
historical records and should not be used without updating their paths and
scenario mappings.

## Which workflow should I use?

| Purpose | Production entry point | Status |
|---|---|---|
| Run the full production workflow | `../workflows/run_ucm_pipeline.py` | Recommended |
| Run final UCM scenarios only | `../Urban_Cooling_Modeling_Runs/run_ucm_scenarios.py` | Production component |
| Convert WBGT to heavy-work productivity | `calculate_hothaps_workability.py` | Production component |
| Summarize energy and productivity | `summarize_ucm_valuations.py` | Production component |
| Compare revised and manuscript-era summaries | `compare_ucm_valuation_versions.py` | Optional production component |
| Build and export the combined manuscript Figure 4 | `plot_figure4_citywide.R` | Recommended final assembly |
| Inspect Figure 4 source panels | `../invest_result_zonal_viz_2_energy.Rmd`, `../invest_result_zonal_viz_3_pd_NEW.Rmd`, `../health_assessment/health-modeling-output-plot-city.Rmd` | Current nine-scenario notebooks |
| Build Figure 5 borough maps | `../viz-es-change-due-to-lc.Rmd` or `plot_figure5_borough_cobenefits.R` | Current nine-scenario workflow |
| Legacy zonal statistics and archived plot sections | non-production sections retained in the notebooks | Historical reference only |

The Figure 4 scenario set is baseline plus AllBuilt, TreeRisk, TreeOpp,
Green10/20/30 and Target10/20/30. All nine comparisons must be run with InVEST
3.20.2. The first three retain their original LULC rasters; the Green/Target
set uses the revised equal-realized-canopy rasters. Do not mix these outputs
with the InVEST 3.14.1 manuscript-era folders or the archived `scenario4` and
`scenario510/520/530` outputs.

## Required external inputs

The repository intentionally does not store large or licence-restricted model
inputs. Under the configured Wellcome Trust project-data root, the valuation
workflow expects:

```text
1_preprocess/UrbanCoolingModel/OfficialWorkingInputs/
├── AOIs/London_Borough_aoi.shp
└── energy_buildings/
    ├── bld_with_attr_compact_ucm2.gpkg
    └── _UCM_Energy Consumption Table.csv
```

The building vector must contain an integer `type` field matching every type
in the energy table. The table's `cost` field is in £ per kWh, so InVEST's
`energy_sav` result is cost-adjusted and can be reported in pounds. The
supporting materials were assembled in late 2025 and cite Q4 2025 tariff
context, but a formally harmonized price year is not recorded for every
building-type rate. Use the qualified label **late-2025 input-price
assumptions** rather than claiming a single exact price year.

## Reproduce the revised energy and productivity results

Set a project-data path appropriate for your machine; do not edit source code
to insert a local drive letter.

### 1. Validate the UCM valuation configuration

```bash
conda run -n urban-cooling-invest-3.20.2 python \
  code/Urban_Cooling_Modeling_Runs/run_ucm_scenarios.py \
  /path/to/Wellcome\ Trust\ Project\ Data \
  --temperatures 25 \
  --include-valuations \
  --output-root /path/to/versioned/valuation-output \
  --repo-root . \
  --validate-only
```

The primary manuscript comparison uses 25°C. Add `28` only when the energy and
productivity sensitivity analysis is required. Validation checks all model
inputs and enforces InVEST 3.20.2.

### 2. Run the baseline and nine scenarios

Repeat the command without `--validate-only`. The runner refuses to overwrite
an existing temperature raster or run manifest. It records input checksums,
model arguments, software versions and output metadata beside each scenario.

With `--include-valuations`, InVEST produces:

- `T_air_*.tif` for downstream temperature and health calculations;
- `buildings_with_stats_*.shp`, containing `energy_sav` for each building;
- `intermediate/wbgt_*.tif`; and
- InVEST's built-in threshold-based work-loss rasters.

The built-in work-loss rasters are not the manuscript productivity measure.
They are retained only because WBGT is generated in the same model branch.

### 3. Calculate the project-specific Hothaps productivity measure

```bash
conda run -n urban-cooling-invest-3.20.2 python \
  code/post_processing_layers/calculate_hothaps_workability.py \
  /path/to/versioned/valuation-output --validate-only

# After validation, rerun without --validate-only.
```

For each valid WBGT pixel, the script calculates:

```text
workability = 0.1 + 0.9 / (1 + (WBGT / 30.94)^16.64)
```

The output is a fraction between 0.1 and 1.0. A difference of `0.01` between a
scenario and baseline is a **1 percentage-point** productivity change. The
script preserves NoData, refuses accidental overwrite and writes a JSON
manifest for every raster.

### 4. Produce citywide and borough summaries

```bash
conda run -n urban-cooling-invest-3.20.2 python \
  code/post_processing_layers/summarize_ucm_valuations.py \
  /path/to/versioned/valuation-output \
  "/path/to/OfficialWorkingInputs/AOIs/London_Borough_aoi.shp"
```

Outputs are written under `summary/`:

- `citywide_energy_productivity_summary.csv`;
- `borough_energy_productivity_summary.csv`; and
- `energy_productivity_summary_manifest.json`.

The citywide energy value sums each building exactly once. InVEST's borough
energy field counts a building in every borough polygon it intersects. The
summary therefore reports the difference between the summed borough field and
the unique-building city total. If that discrepancy is material, allocate
cross-boundary buildings by intersection-area share before using borough
energy results.

Productivity is summarized as the area-weighted mean across valid 10 m pixels.
If an exposure- or worker-weighted interpretation is desired, add a suitable
worker-location raster and report that as a separate sensitivity analysis.

### 5. Compare with the manuscript-era results

```bash
conda run -n urban-cooling-invest-3.20.2 python \
  code/post_processing_layers/compare_ucm_valuation_versions.py \
  /path/to/UCM_official_runs \
  /path/to/versioned/valuation-output/summary/citywide_energy_productivity_summary.csv
```

The comparison uses the preserved `scenario0`, `scenario41/42/43` and
`scenario510/520/530` 25°C building outputs. It recalculates historical
Hothaps workability directly from WBGT because the archived Green10 derived
productivity TIFF contains an unreadable compressed tile. Productivity gains
are paired cell by cell against the matching baseline on their common valid
footprint; separate raster means are not used to infer a change when the old
Green rasters have a smaller valid footprint. Energy gains are calculated
against each model version's own baseline.

The resulting CSV and JSON manifest are written to the revised run's
`summary/` directory. `--reuse-energy-from-output` is only a resume aid after
the energy columns have already been produced by a complete run; omit it for
an independent rebuild.

## Validation checklist

Before updating figures or manuscript text, confirm:

1. all seven scenarios use the same InVEST version, 25°C reference
   temperature, UHI maximum and humidity;
2. every scenario has a `T_air`, building-energy, WBGT and Hothaps output;
3. Hothaps values remain within 0.1–1.0 and use the same valid-cell mask;
4. the baseline exists at every reported temperature;
5. citywide energy uses the unique-building total;
6. borough energy duplication has been quantified and resolved if material;
7. scenario changes are calculated against the matching-temperature baseline;
8. all displayed values reconcile exactly to saved CSVs; and
9. no restricted raster or building input is added to Git.

The source building layer currently produces polygon winding-order warnings;
GDAL/InVEST autocorrects them during processing. Record the warning and confirm
that all 2,223,481 buildings receive energy values, as in the current run.

## Build manuscript Figure 4

Figure 4 pools energy, productivity and health as citywide co-benefits. Its
primary bars therefore use the natural London-wide estimand for each outcome:

- energy: total avoided cost summed once across unique buildings;
- productivity: area-weighted mean continuous Hothaps workability change over
  valid 10 m pixels; and
- health: total deaths averted using fixed 2021 population and registered
  mortality.

Run:

```bash
Rscript code/post_processing_layers/plot_figure4_citywide.R \
  --citywide-summary /path/to/run/summary/citywide_energy_productivity_summary.csv \
  --borough-summary /path/to/run/summary/borough_energy_productivity_summary.csv \
  --health-root /path/to/health_v3_revised_equal_area_population_weighted_2021_2026-09-10 \
  --temperature 25 \
  --output-dir /path/to/run/summary/figure4
```

The script writes PNG, PDF and SVG versions of the primary figure, its exact
plotting data, and a provenance manifest. When `--borough-summary` is supplied,
it also writes a boxplot and source table showing the equal-weight distribution
across 33 boroughs, plus a four-panel Extended Data comparison of the citywide
and unweighted-borough estimands. These borough results describe spatial
variation; they are not model confidence intervals.

Only the health panel has uncertainty bars. They are the 2.5th and 97.5th
percentiles of paired exposure-response draws. Energy and productivity remain
deterministic unless their parameter uncertainty is propagated separately.
Do not reuse the borough standard error as uncertainty around a citywide total.

Bars are directly labelled with signed values, following the established
`func_plot_change_point()` convention. Energy uses one decimal place,
productivity two decimal places and health whole deaths. Health labels are
placed beyond the 95% interval to avoid obscuring its uncertainty bars. The
shared theme uses an 11-point base font and exports vector PDF/SVG files for
final typesetting. At the intended 183 mm double-column width, the combined
figure scales to approximately 5–7 point text, matching Nature's current
figure-artwork guidance; recheck this after any layout or publisher resizing.

See `FIGURE4_METHODS.md` for the manuscript recommendation, estimand wording
and revised legend.

The three established Figure 4 R Markdown notebooks remain the transparent
panel-level entry points. Their production sections now read the same reviewed
summary files, include Green10/20/30 and Target10/20/30, and share ordering and
styling from `figure4_panel_helpers.R`. Their older code is retained below an
explicit archived heading and is not executed during knitting.

## Build manuscript Figure 5

Knit `../viz-es-change-due-to-lc.Rmd`, or run its batch builder directly:

```bash
Rscript code/post_processing_layers/plot_figure5_borough_cobenefits.R \
  --borough-summary /path/to/run/summary/borough_energy_productivity_summary.csv \
  --health-root /path/to/reviewed/health-output \
  --borough-vector /path/to/London_Borough_aoi.shp \
  --temperature 25 \
  --output-dir /path/to/run/summary/figure5
```

The production Figure 5 includes all nine manuscript scenarios. For
readability, the three original counterfactuals (AllBuilt, TreeRisk and
TreeOpp) form one block and the six Green/Target canopy-addition scenarios form
a second block. Every outcome uses a symmetric diverging colour scale shared
across both blocks, with zero as the neutral midpoint, so negative and positive
scenario effects remain directly comparable.

The builder writes the mapped data, PNG/PDF/SVG figures and a checksum
manifest. A common scale is used across all nine scenarios within each outcome
row. Health is aggregated from the reviewed all-cause mortality-change raster
by borough; no borough uncertainty interval is inferred.

## Historical files

These files preserve earlier exploratory processing and machine-specific
paths. They are useful for tracing older figures, but are not portable
production entry points:

- `aggregating_building_data.py` and `aggregate_interesected_buildings.R`;
- `processing_UCM_shapefiles.R`;
- `borough_postprocess_stats.Rmd`;
- `scenario_compare_postprocess_graphs_and_maps.Rmd`; and
- `scenario_compare_postprocess_stats.Rmd`.

The legacy notebooks in the repository root—`invest_result_zonal_stats_temp`,
`invest_result_zonal_stats_energy` and `invest_result_zonal_stats_productivity`—
also point to manuscript-era output folders. Preserve them for audit history
until their required plotting or tabulation logic has been migrated.
