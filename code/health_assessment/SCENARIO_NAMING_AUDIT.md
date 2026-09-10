# Target-scenario naming audit

Reviewed: 9 September 2026

This audit records the active and downstream references for the renamed
`710v3/720v3/730v3` Target10/20/30 inputs. A checksum comparison resolves the
Target30 input used in Figure 7; Target10 and Target20 still require review.

## Checksum result

The two Target30 rasters are byte-for-byte identical:

| File | SHA-256 |
|---|---|
| `LULC/LULC_Scenario530.tif` | `1999f4760c3c94acf95f4578ef846ba69597b6a729c4bcfb01cfdd2230d432c7` |
| `LULC/lc_tree_equity_scenarios_output/LULC_Scenario730v3.tif` | `1999f4760c3c94acf95f4578ef846ba69597b6a729c4bcfb01cfdd2230d432c7` |

Therefore the existing `scenario530` UCM outputs describe the same legacy v3
Target30 input. They are retained for comparison but cannot be used as the
final equal-budget Target30 result. This does not extend to the other
intervention levels: the `510`/`710v3` and `520`/`720v3` checksum pairs differ.

## Approved equal-budget Target30

`LULC_Scenario730v4_equal_budget.tif` was generated separately on 9 September
2026. It extends v3 with the next-ranked candidate locations until the realized
addition is exactly 930,000 tree pixels (93 km²), equal to Green30. The audit
passes with a 0.000% difference and no removal of baseline canopy. The adjacent
manifest records the input and output checksums, five prefix-sequence checks,
software versions, feature counts and rank cutoff. Existing files were not
overwritten.

## Current construction references

| Stage | File | Active reference |
|---|---|---|
| Allocate target trees | [`tree_equity_1_number_of_trees_to_polygon.py`](../lc_scenarios/tree_equity_1_number_of_trees_to_polygon.py) | `tree_equity_scenario710v3`, `720v3`, `730v3` |
| Rasterize target planting | [`tree_equity_2_scenario_engine.py`](../lc_scenarios/tree_equity_2_scenario_engine.py) | reads the three `v3` GeoPackages and writes `LULC_Scenario710v3.tif`, `720v3.tif`, `730v3.tif` |
| Calculate LULC statistics | [`tree_equity_3_lulc_stats.py`](../lc_scenarios/tree_equity_3_lulc_stats.py) | the `v3` rasters are commented out; `LULC_Scenario510.tif`, `520.tif`, `530.tif` are active |

## Current downstream references

The following files still use `510/520/530`:

- `code/Urban_Cooling_Modeling_Runs/Scenario_510_to_530_runs/execute_invest_urban_cooling_model_scenario510.py`
- `code/Urban_Cooling_Modeling_Runs/Scenario_510_to_530_runs/execute_invest_urban_cooling_model_scenario520.py`
- `code/Urban_Cooling_Modeling_Runs/Scenario_510_to_530_runs/execute_invest_urban_cooling_model_scenario530.py`
- `code/Urban_Cooling_Modeling_Runs/Scenario_510_to_530_runs/UCM_sherlock_runs.sbatch`
- `code/health_assessment/health-modeling_s0_s510_2050_2050.bat`
- `code/health_assessment/health-modeling_s0_s520_2050_2050.bat`
- `code/health_assessment/health-modeling_s0_s530_2050_2050.bat`
- `code/health_assessment/health-modeling-output-plot-city.Rmd`
- `code/health_assessment/health-modeling-zonal-stats-viz-borough.Rmd`
- `code/equity-health.Rmd`
- `code/func_colors.R`

## Climate settings

Each `execute_invest_urban_cooling_model_scenario{510,520,530}.py` script
currently has only `[28, 5, 45]` active in its `variables` array. Inspection of
the shared project folder confirmed that `scenario530` contains both the
`25deg_5uhi_45hum` and `28deg_5uhi_45hum` Target30 outputs, and `scenario43`
contains the corresponding Green30 outputs. The revised health configuration
therefore includes both temperature settings, each paired with its matching
baseline.

## Confirmation checklist

For each Target scenario, record:

| Manuscript label | Approved LULC raster | Realized added canopy | UCM run directory | Temperature/UHI/humidity | Health output directory |
|---|---|---:|---|---|---|
| Target10 | TBD | TBD | TBD | TBD | TBD |
| Target20 | TBD | TBD | TBD | TBD | TBD |
| Target30 | `LULC_Scenario730v4_equal_budget.tif` | 930,000 pixels / 93 km² | `scenario730v4_equal_budget_health_invest3202` | both 25/5/45 and 28/5/45 | `health_v2_invest3202_population_weighted_2021_nodata_harmonized/target30_25c` and `target30_28c` |

For the unresolved Target10 and Target20 scenarios, after approval:

1. update construction, LULC-statistics, UCM, health and plotting references in
   one commit;
2. validate realized canopy area against the corresponding Green scenario;
3. regenerate UCM outputs if either the approved LULC or climate configuration
   differs from the existing run;
4. regenerate health and zonal outputs; and
5. retain the old mapping and output checksums in the publication archive.
