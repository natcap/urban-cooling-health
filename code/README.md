# Code guide

Start new production analysis from [`workflows/`](workflows/README.md). The
other folders contain reusable pipeline components and historical analyses.

## Production entry points

| Task | Use this file |
|---|---|
| Complete UCM, energy and productivity workflow | [`workflows/run_ucm_pipeline.py`](workflows/run_ucm_pipeline.py) |
| UCM scenarios only | [`Urban_Cooling_Modeling_Runs/run_ucm_scenarios.py`](Urban_Cooling_Modeling_Runs/run_ucm_scenarios.py) |
| Health scenario batch | [`health_assessment/run_health_scenario_set.py`](health_assessment/run_health_scenario_set.py) |
| Figure 1 climate context | [`gcm-data-clip-stats-viz.Rmd`](gcm-data-clip-stats-viz.Rmd) |
| Figure 2 temperature/canopy equity | [`equity-temp-tcc.Rmd`](equity-temp-tcc.Rmd); [temperature input procedure](#figure-2-temperature-input) |
| Figure 3 temperature results | [`invest_result_zonal_viz_1_temp.Rmd`](invest_result_zonal_viz_1_temp.Rmd) |
| Figure 4 citywide co-benefits | [`invest_result_zonal_viz_2_energy.Rmd`](invest_result_zonal_viz_2_energy.Rmd), [`invest_result_zonal_viz_3_pd_NEW.Rmd`](invest_result_zonal_viz_3_pd_NEW.Rmd), [`health-modeling-output-plot-city.Rmd`](health_assessment/health-modeling-output-plot-city.Rmd); final assembly: [`plot_figure4_citywide.R`](post_processing_layers/plot_figure4_citywide.R) |
| Figure 5 borough co-benefit maps | [`viz-es-change-due-to-lc.Rmd`](viz-es-change-due-to-lc.Rmd); batch builder: [`plot_figure5_borough_cobenefits.R`](post_processing_layers/plot_figure5_borough_cobenefits.R) |
| Figure 6 exploratory analysis | [`equity-health.Rmd`](equity-health.Rmd) — not final until the [review gates](health_assessment/FIGURES6_7_REVIEW.md) are complete |
| Figure 7 paired equity comparison | [`run-fig7.R`](run-fig7.R); follow the [Figure 7 guide](../REPRODUCIBILITY.md) |
| Vulnerability indicator preparation | [`socio-economic-data.Rmd`](socio-economic-data.Rmd) |

## Figure 2 temperature input

The original external-drive folders (`E:\chelsa02\clipped_london_stats\` and
`E:\GCMs\clipped\stats\`) are now under `data/02_preprocess/` with the same
subdirectory names. The climate notebook uses those project-relative paths by
default and still permits `CHELSA_STATS_DIR` and `WORLDCLIM_STATS_DIR`
overrides. It defaults to borough summaries for Figure 1; select LSOA mode to
rebuild the Figure 2 cache using the original baseline–future calculation:

```r
Sys.setenv(FIGURE2_CLIMATE_SPATIAL_UNIT = "lsoa")
rmarkdown::render("code/gcm-data-clip-stats-viz.Rmd")
Sys.setenv(FIGURE2_OUTCOME = "temp_change")
rmarkdown::render("code/equity-temp-tcc.Rmd")
```

Inputs: `data/02_preprocess/chelsa02/clipped_london_stats/lsoa_stats_CHELSA_tasmax_07_2021_V.2.1_London.csv`
and the three `lsoa_stats_wc2.1_30s_tmax_{ACCESS-CM2,IPSL-CM6A-LR,MRI-ESM2-0}_ssp370_2041-2060_London.csv`
files in `data/02_preprocess/GCMs/clipped/stats/`. The LSOA CSVs retain the
column name `borough`, but its numeric values match `id` in
`data/Social_Vulnerability_Index_london_q.gpkg`; renaming it to `id` in the
model join is incorrect. The script retains the original `borough` join,
averages the three model results by LSOA, and converts that key to character
only for the final vulnerability-layer join. It checks missing inputs, unique
baseline keys and finite 2050 values before saving `data/temp_svi.RDS`.
The shared `figure2_input_checks.R` additionally requires the exact three-model
set per scenario/period, complete unique area keys and no missing values beyond
the four documented source-gap IDs. Figure 2 selects SSP370/2050 before ranking
indicators, preventing pooling when the cache includes other climate cases.
Run `Rscript code/tests/test_figure2_input_checks.R` for rejection-path tests.
These large local data files are ignored by Git; another machine needs its
own copy under the same project-relative paths or the two path overrides.

On 3 November 2026 the rebuilt cache contained 4,835 unique LSOAs, with
4,831 finite July 2021-to-2050 temperature changes (four source-data gaps).
The prior all-NA cache was preserved locally as
`data/temp_svi_invalid_pre_repair_2026-11-03.RDS`. The Figure 2 temperature
notebook completed and wrote the `figure2_temp_change_*` statistics in
`figures/figure2_statistical_reporting/`. Those rank tests do not adjust for
spatial dependence; review that inference before manuscript use. The original
CSV-generation record is [`gcm-data-clip-stats.ipynb`](gcm-data-clip-stats.ipynb).
Figures 2, 6 and 7 use the author-prepared four-theme `overall_score`.
Reproduce the former Figure 2 comparison with `Rscript code/compare_figure2_svi.R`.
Run `Rscript code/run_figure6_spatial.R` for the Green30 spatial sensitivity
(unadjusted and spatial basis sizes 50/100/200). See the
[results and interpretation](../docs/manuscript/SVI_SPATIAL_RESULTS.md).

## Shared R notebook setup

The `invest_result_zonal_viz_*` notebooks use one setup module instead of
repeating package imports, helper sources, baseline year and machine-specific
paths:

```r
source(here::here("code", "ucm_analysis_setup.R"))
ucm_config <- ucm_analysis_setup(export_legacy_names = TRUE)
```

Set `URBAN_COOLING_DATA_ROOT` once in your user-level `~/.Renviron`; copy the
template line from [`.Renviron.example`](../.Renviron.example), replace the
placeholder locally and restart R. Do not add a personal path to a tracked
notebook. `export_legacy_names = TRUE` supplies `dir.g`, `dir.aoi`, `dir.fig`,
`dir_ucm_out`, `dir_prod_new`, `year_baseline` and `ymd` for the existing
notebooks. New code should instead use explicit fields such as
`ucm_config$paths$ucm_output_dir` and `ucm_config$baseline_year`.

For an exactly reproducible rerun of legacy date-stamped filenames, also set
`URBAN_COOLING_RUN_DATE=YYYYMMDD`; otherwise it defaults to the run date. The
setup stops early with a named missing-package or missing-directory error.

## Naming convention

New production scripts use `verb_subject.py`:

- `run_*` orchestrates a model or multi-step workflow;
- `prepare_*` creates an input;
- `generate_*` creates a scenario;
- `validate_*` checks an input or result without changing it;
- `calculate_*` derives a modeled layer;
- `summarize_*` creates aggregate tables; and
- `compare_*` compares versions or scenarios.

Use lowercase `snake_case` for Python and JSON filenames. Put dates, model
versions and scenario revisions in committed configuration files and output
directory names, not in script filenames. Name production configurations
`ucm_pipeline_YYYY-MM-DD.json` and keep them under `workflows/configs/`. This
allows the code to remain stable while each run remains identifiable.

Existing collaborator scripts and R Markdown notebooks retain their original
names to preserve provenance. A name such as `scenario510`, `scenario4`,
`*_NEW`, or a numbered notebook should be treated as historical unless a
folder README explicitly marks it as production. The three Figure 4 source
notebooks above are explicit exceptions: their first sections are the current
nine-scenario Figure 4 workflow and their manuscript-era code is retained as a disabled
archive.

## Folder roles

- `workflows/` — one-command orchestration, run configurations and lightweight
  workflow tests under `workflows/tests/`;
- `lc_scenarios/` — scenario generation and validation;
- `Urban_Cooling_Modeling_Runs/` — UCM execution components and historical
  launchers;
- `post_processing_layers/` — Hothaps, energy/productivity summaries and
  version comparison, Figure 4 production and its aggregation/uncertainty
  decision record; unit tests are under `post_processing_layers/tests/`;
- `health_assessment/` — mortality, health impacts and LSOA aggregation;
- `preprocessing_layers/` — spatial input preparation; and
- top-level `*.Rmd` files — manuscript analysis, figures and historical
  exploration.

Do not copy a historical script to create another scenario-specific runner.
Add the scenario to a configuration file or extend the shared component.
