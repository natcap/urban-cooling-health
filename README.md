# Urban cooling, health and equity in London

This repository contains the research workflow used to compare how alternative
urban land-cover and tree-planting strategies affect heat exposure, health
outcomes and the distribution of benefits across London neighbourhoods.

The workflow connects four main components:

1. construct baseline and alternative land-cover scenarios;
2. simulate urban cooling with the InVEST Urban Cooling Model;
3. estimate heat-attributable mortality and other co-benefits; and
4. evaluate how benefits are distributed across social-vulnerability groups.

The repository is research code under active manuscript revision. It contains
enough derived data to reproduce the current Figure 7 analysis, but a complete
rebuild from raw data requires access to the project's external geospatial data
store.

Current blockers and the recommended order of work are tracked in
[`TODO.md`](TODO.md).

## Current analysis convention

All health analyses use **2021 population and 2021 baseline mortality**, even
when the temperature field represents a 2050 climate scenario. Holding
demography constant isolates the effect of climate and land-cover change.

Results must therefore be described as:

> 2050 temperature scenarios evaluated under 2021 population and mortality.

They are not projections of London's population or mortality patterns in 2050.

## Choose the workflow you need

| Goal | Start here | External data required? |
|---|---|---:|
| View the current results | [Figure 7 outputs](#4-inspect-the-outputs) | No |
| See outstanding work and required data | [Project TODO](TODO.md) | No |
| Reproduce Figure 7 | [Quick reproduction of Figure 7](#quick-reproduction-of-figure-7) | No for the current absolute-benefit version; yes for population-normalized results |
| Rerun the health model | [Health assessment](#6-run-the-health-assessment) | Yes |
| Rebuild every scenario and result | [Full end-to-end workflow](#full-end-to-end-workflow) | Yes |
| Check Green30 and Target30 planting equivalence | [Validate intervention budgets](#4-validate-intervention-budgets) | Yes |

## Scenarios

Scenario names used in the analysis are defined centrally in
[`code/func_colors.R`](code/func_colors.R).

| Analysis label | Internal label | Description |
|---|---|---|
| Baseline | `scenario0` | Current/reference land cover |
| AllBuilt | `scenario1` | Counterfactual built-land scenario |
| TreeRisk | `scenario2_TR` | Loss of trees considered at climatic risk |
| TreeOpp | `scenario3_TO` | Tree-planting opportunity scenario |
| Green10/20/30 | `scenario4_10/20/30` | General greening with 10%, 20% or 30% relative tree-canopy increase |
| Target10/20/30 | `scenario510/520/530` | Vulnerability-targeted planting at corresponding intervention levels |

The Green30–Target30 equity comparison is valid only after confirming that the
two final scenario rasters contain comparable **realized additional canopy
area**. Intended tree counts alone are not sufficient because existing-canopy
overlap and rasterization can change the realized intervention.

## Quick reproduction of Figure 7

This is the shortest fully scripted workflow in the repository.

### 1. Clone the repository

```bash
git clone https://github.com/natcap/urban-cooling-health.git
cd urban-cooling-health
```

### 2. Install the required R packages

The last validated package versions are recorded in
[`fig7_session_info.txt`](figures/equity_map_biscale/fig7_session_info.txt).
Install R and the packages below if they are not already available:

```r
install.packages(c(
  "biscale", "cowplot", "dplyr", "ggplot2", "knitr", "purrr",
  "ragg", "readr", "rmarkdown", "scales", "sf", "tidyr"
))
```

The `sf` and `ragg` packages may require system geospatial and graphics
libraries. A conda-based R environment or the official R installers can be used
when these dependencies are not already present.

### 3. Run the production workflow

From the repository root:

```bash
Rscript code/run-fig7.R
```

The wrapper does not require Pandoc. It validates scenario pairing, duplicate
IDs, missing benefit values and population coverage before producing outputs.

### 4. Inspect the outputs

The workflow writes the following files to
`figures/equity_map_biscale/`:

- `fig7_ab_upgraded.png`: comparable bivariate maps using common thresholds;
- `fig7_cd_upgraded.png`: paired scenario comparison and transition summary;
- `fig7_quantitative_headline_results.csv`: headline equity statistics;
- `fig7_paired_transition_matrix.csv`: exact Green30-to-Target30 transitions;
- `fig7_age75_*.csv`: diagnostic evidence for the age-75+ result;
- `fig7_citywide_tradeoff.csv`: citywide deaths-averted comparison;
- `fig7_run_metadata.csv`: run settings and fixed thresholds; and
- `fig7_session_info.txt`: the R environment used for the run.

![Figure 7 maps comparing Green30 and Target30](figures/equity_map_biscale/fig7_ab_upgraded.png)

![Figure 7 paired equity comparison](figures/equity_map_biscale/fig7_cd_upgraded.png)

### 5. Add 2021 population for the publication analysis

The tracked `health_sf.rds` currently lacks an LSOA population denominator.
Without it, the workflow transparently reports absolute deaths averted per LSOA
and leaves population-weighted fields blank.

Create:

```text
data/derived/lsoa_population_2021.csv
```

with exactly these columns:

```csv
id,population_2021
1,VALUE_FOR_ID_1
2,VALUE_FOR_ID_2
```

The file must contain 4,835 unique IDs with positive, non-missing total
usual-resident population values. See
[`data/derived/README.md`](data/derived/README.md) for the validation rules and
the current ID limitation. Rerun `Rscript code/run-fig7.R`; the production
workflow will then classify mortality benefit using deaths averted per 100,000
2021 residents.

For the full Figure 7 protocol, including Monte Carlo files and the final review
checklist, see [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md).

## Full end-to-end workflow

The following sequence reproduces the method from raw spatial inputs. Exact
historical results additionally require the same versioned input rasters and
vectors used in the original runs.

### 1. Prepare the software environments

The workflow currently uses Windows batch files for health-model runs, Python
for raster/vector processing and InVEST, R/R Markdown for statistics and
figures, and Jupyter notebooks for several preprocessing and zonal-statistics
steps.

A practical Python environment includes:

```bash
conda create -n geo_env -c conda-forge \
  python=3.11 geopandas jupyterlab matplotlib numpy pandas pyogrio \
  rasterio shapely
conda activate geo_env
pip install natcap.invest==3.14.1
```

InVEST 3.14.1 is the version recorded in the current model-run scripts. If a
different version is used, record it with the run outputs and check for changed
defaults or input requirements.

The complete R analysis uses additional packages beyond the Figure 7 subset,
including `exactextractr`, `here`, `mgcv`, `performance`, `terra`, `tmap`,
`tidyverse` and `writexl`. Install packages as requested by the relevant R
Markdown file and record the final `sessionInfo()`.

### 2. Assemble and version the input data

Most large project inputs are intentionally not committed. The scripts expect
the following broad input groups:

| Input group | Examples | Main use |
|---|---|---|
| Study boundaries | London borough and LSOA geometries | clipping, zonal statistics and mapping |
| Land cover and tree canopy | baseline LULC, tree-canopy cover and planting opportunities | scenario construction and cooling model |
| Climate | reference evapotranspiration and temperature/UHI settings | InVEST cooling simulations |
| Population | 2021 population raster | health-impact calculation and per-capita equity metrics |
| Mortality | 2021 cause-specific mortality rasters | baseline health burden |
| Buildings | building footprints and energy attributes | energy valuation |
| Vulnerability | income, age, ethnicity, housing and related LSOA indicators | SVI and equity analysis |

Store large data outside Git and preserve an immutable input manifest containing
source, licence, download date, geography vintage, units, CRS, resolution,
checksum and any preprocessing performed. The original scripts commonly expect
the shared-drive structure below:

```text
Wellcome Trust Project Data/
├── 1_preprocess/UrbanCoolingModel/OfficialWorkingInputs/
│   ├── AOIs/
│   ├── LULC/
│   ├── evapotranspiration/
│   ├── pop_raster/
│   ├── health_rasters_10m_bng/
│   └── energy_buildings/
└── 2_postprocess_intermediate/UCM_official_runs/
```

### 3. Prepare baseline layers and construct scenarios

Run only the components needed for the scenarios under study:

1. Align and clip core layers with
   [`preprocess_and_align_london_layers.py`](code/preprocessing_layers/preprocess_and_align_london_layers.py).
2. Prepare the population surface with
   [`pop_01_prep_clip_reproj.ipynb`](code/preprocessing_layers/pop_01_prep_clip_reproj.ipynb)
   and [`pop_02_resample_for_health_model.py`](code/preprocessing_layers/pop_02_resample_for_health_model.py).
3. Build pavement and tree-opportunity scenarios with
   [`scenario_1_pavement_and_2_opportunity_trees.ipynb`](code/lc_scenarios/scenario_1_pavement_and_2_opportunity_trees.ipynb).
4. Build the tree-risk scenario using the `tree-at-climate-risk-*` scripts in
   [`code/lc_scenarios/`](code/lc_scenarios/).
5. Create general Green10/20/30 rasters using the InVEST Scenario Generator and
   retain the final LULC rasters referenced by the scenario-4 run scripts.
6. Create Target10/20/30 planting polygons and rasters, in order, with:
   - [`tree_equity_1_number_of_trees_to_polygon.py`](code/lc_scenarios/tree_equity_1_number_of_trees_to_polygon.py)
   - [`tree_equity_2_scenario_engine.py`](code/lc_scenarios/tree_equity_2_scenario_engine.py)
   - [`tree_equity_3_lulc_stats.py`](code/lc_scenarios/tree_equity_3_lulc_stats.py)

**Configuration gate:** the currently active generator configuration uses
`710v3`, `720v3` and `730v3` filenames, while downstream UCM and health scripts
refer to `510`, `520` and `530`. Before running, select one approved scenario
version and make its names and paths consistent across all three scenario
scripts, the UCM run scripts, the health batch files and the scenario-label
mapping. Do not combine outputs from different versions.

Several older scenario scripts contain machine-specific paths. Review every
input and output path before running them; do not assume the defaults point to
the intended data snapshot.

### 4. Validate intervention budgets

Before comparing Green30 with Target30, run:

```bash
python code/lc_scenarios/validate_scenario_canopy_budget.py \
  --baseline /path/to/baseline_lulc.tif \
  --green30 /path/to/green30_lulc.tif \
  --target30 /path/to/target30_lulc.tif \
  --output figures/equity_map_biscale/fig7_canopy_budget_check.csv
```

The default tolerance is 0.5% difference in added tree-canopy pixels. A failed
check means Target30 must be regenerated with an adjusted allocation, then
revalidated before the UCM and health-model steps are rerun.

### 5. Run the InVEST Urban Cooling Model

Scripts are grouped under
[`code/Urban_Cooling_Modeling_Runs/`](code/Urban_Cooling_Modeling_Runs/).
They accept the external data root as the first argument. For example:

```bash
python code/Urban_Cooling_Modeling_Runs/execute_invest_urban_cooling_model_current_lulc.py \
  "G:/Shared drives/Wellcome Trust Project Data" --eap
```

Run the baseline and every required scenario with identical model settings.
Target10/20/30 scripts are in
[`Scenario_510_to_530_runs/`](code/Urban_Cooling_Modeling_Runs/Scenario_510_to_530_runs/).
Check each script's `variables` array, land-cover path, model parameters and
output suffix before execution.

### 6. Run the health assessment

Detailed health-model notes are in
[`code/health_assessment/README.md`](code/health_assessment/README.md).

1. Prepare cause-specific 2021 mortality inputs with
   [`health-model-01-prep-input-ONS-mortality-data.Rmd`](code/health_assessment/health-model-01-prep-input-ONS-mortality-data.Rmd).
2. Confirm that temperature, population and mortality rasters share the intended
   grid, CRS, extent and units.
3. Review the paths in the relevant `health-modeling_*.bat` file.
4. Activate `geo_env` and run the batch file from Windows. For Target30:

   ```bat
   cd D:\natcap\urban-cooling-health\code\health_assessment
   health-modeling_s0_s530_2050_2050.bat
   ```

5. For Green30 and Target30 uncertainty, use at least 2,000 draws and the same
   seed. The current batch files use seed `20260908`.

The health model produces aligned temperature differences, attributable
fractions, cause-specific excess-death rasters, deterministic city totals and
Monte Carlo draw files.

### 7. Calculate zonal statistics

Run
[`health-modeling-zonal-stats.ipynb`](code/health_assessment/health-modeling-zonal-stats.ipynb)
to aggregate health outputs to borough or LSOA level. Confirm that each output
contains one row per geography, scenario and outcome before joining other
attributes.

Other service outputs use:

- [`invest_result_zonal_stats_temp.ipynb`](code/invest_result_zonal_stats_temp.ipynb)
- [`invest_result_zonal_stats_energy.ipynb`](code/invest_result_zonal_stats_energy.ipynb)
- [`invest_result_zonal_stats_productivity.ipynb`](code/invest_result_zonal_stats_productivity.ipynb)

### 8. Build vulnerability indicators

Run [`socio-economic-data.Rmd`](code/socio-economic-data.Rmd) to clean LSOA
vulnerability metrics and construct the composite SVI. Preserve the official
LSOA code in future rebuilds; the current derived object also contains a numeric
row-order ID retained for compatibility.

### 9. Produce analysis and manuscript figures

| Output | Main script |
|---|---|
| Climate context / Figure 1 | [`gcm-data-clip-stats-viz.Rmd`](code/gcm-data-clip-stats-viz.Rmd) |
| Temperature and canopy equity / Figure 2 | [`equity-temp-tcc.Rmd`](code/equity-temp-tcc.Rmd) |
| Temperature results / Figure 3 | [`invest_result_zonal_viz_1_temp.Rmd`](code/invest_result_zonal_viz_1_temp.Rmd) |
| Energy, productivity and health summaries / Figures 4–5 | the `invest_result_zonal_viz_*`, health-output and [`viz-es-change-due-to-lc.Rmd`](code/viz-es-change-due-to-lc.Rmd) workflows |
| Health-equity models / Figure 6 and supplementary table | [`equity-health.Rmd`](code/equity-health.Rmd) |
| Paired Green30–Target30 equity comparison / Figure 7 | [`equity-health-fig7-production.Rmd`](code/equity-health-fig7-production.Rmd), preferably through [`run-fig7.R`](code/run-fig7.R) |

Treat [`equity-health-fig7-upgrade.Rmd`](code/equity-health-fig7-upgrade.Rmd)
as an archived August 2026 workflow. It is retained for audit history, not as
the current production script.

### 10. Apply review gates before reporting results

Do not treat a successful script run as sufficient validation. Confirm:

1. input provenance, years, units, CRS and raster alignment are recorded;
2. scenario names map to the intended LULC and temperature rasters;
3. Green30 and Target30 pass the realized-canopy budget check;
4. health estimates explicitly state the fixed-2021-demography convention;
5. LSOA joins are one-to-one and contain all 4,835 expected areas;
6. population-normalized results use a complete 2021 population lookup;
7. uncertainty statements distinguish paired-LSOA sensitivity intervals from
   exposure-response Monte Carlo uncertainty; and
8. final figures and tables are visually inspected and agree with their saved
   source CSVs.

## Repository map

```text
urban-cooling-health/
├── README.md                         # project entry point
├── TODO.md                           # prioritized blockers and next actions
├── REPRODUCIBILITY.md                 # detailed Figure 7 protocol
├── data/
│   └── derived/                      # small documented Figure 7 inputs
├── figures/
│   └── equity_map_biscale/           # Figure 7 data, figures and metadata
├── code/
│   ├── preprocessing_layers/        # align and prepare spatial inputs
│   ├── lc_scenarios/                # construct and validate LULC scenarios
│   ├── Urban_Cooling_Modeling_Runs/ # execute InVEST UCM scenarios
│   ├── health_assessment/           # mortality model and zonal statistics
│   ├── post_processing_layers/      # aggregate model outputs
│   ├── equity-health.Rmd            # broader health-equity analysis
│   └── run-fig7.R                  # portable Figure 7 entry point
└── LICENSE
```

## Reproducibility status and limitations

- **Figure 7 method:** directly runnable from the tracked `health_sf.rds`.
- **Population-normalized Figure 7:** requires the documented 2021 LSOA
  population lookup.
- **Citywide Monte Carlo comparison:** requires paired Green30 and Target30 draw
  files generated with the same seed.
- **Full model chain:** requires external raw and intermediate geospatial data
  that are not stored in this repository.
- **Historical result recreation:** requires the original versioned input
  snapshots, not merely datasets with the same names.
- **Portability:** several legacy scripts still contain Windows/shared-drive
  paths and should be reviewed before use on another computer.

## Contributing and reporting issues

When proposing a change, state which scenario, data snapshot and output it
affects. Keep generated headline tables synchronized with their figures, retain
run metadata, and avoid committing private or licence-restricted source data.
Use the repository's GitHub issue tracker for questions or reproducibility
problems.

## Licence

See [`LICENSE`](LICENSE).
