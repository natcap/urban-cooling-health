# Code guide

Start new production analysis from [`workflows/`](workflows/README.md). The
other folders contain reusable pipeline components and historical analyses.

## Production entry points

| Task | Use this file |
|---|---|
| Complete UCM, energy and productivity workflow | `workflows/run_ucm_pipeline.py` |
| UCM scenarios only | `Urban_Cooling_Modeling_Runs/run_ucm_scenarios.py` |
| Health scenario batch | `health_assessment/run_health_scenario_set.py` |
| Figure 4 citywide co-benefits | `invest_result_zonal_viz_2_energy.Rmd`, `invest_result_zonal_viz_3_pd_NEW.Rmd`, `health_assessment/health-modeling-output-plot-city.Rmd`; final assembly: `post_processing_layers/plot_figure4_citywide.R` |
| Figure 5 borough co-benefit maps | `viz-es-change-due-to-lc.Rmd`; batch builder: `post_processing_layers/plot_figure5_borough_cobenefits.R` |
| Figures 6–7 review gates | `health_assessment/FIGURES6_7_REVIEW.md` |
| Figure 6 exploratory analysis | `equity-health.Rmd` — not final until review gates are complete |
| Figure 7 | `run-fig7.R` |

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
