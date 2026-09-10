# Target-scenario identity and canopy-budget audit

Reviewed: 10 September 2026

This audit separates two questions that must not be conflated:

1. **File identity:** was a trial raster copied to a manuscript-facing
   `510/520/530` filename without changing its contents?
2. **Budget comparability:** does that raster realize the same added tree-canopy
   area as the corresponding Green10/20/30 raster?

## Confirmed copy/alias mapping

All three user-declared pairs are byte-for-byte identical. Copies in both the
parent `LULC` folder and `lc_tree_equity_scenarios_output` have the same hash.

| Manuscript label | Selected trial raster | Manuscript-facing copy | SHA-256 |
|---|---|---|---|
| Target10 | `LULC_Scenario710v2.tif` | `LULC_Scenario510.tif` | `b756510124c996a08ab180aeca0518b057a4c84884083f602ae66e1bd295e520` |
| Target20 | `LULC_Scenario730v2.tif` | `LULC_Scenario520.tif` | `15edfce32b571bdd04faa875768a3ca911175eb8f7d85f4b218a4c07a1c1ec40` |
| Target30 (legacy) | `LULC_Scenario730v3.tif` | `LULC_Scenario530.tif` | `1999f4760c3c94acf95f4578ef846ba69597b6a729c4bcfb01cfdd2230d432c7` |

The numbering is intentionally non-sequential: Target20 came from the
`730v2` trial, not `720v2` or `720v3`. Future scripts and documentation must
use this explicit mapping rather than infer a source filename from the
manuscript label.

## Original code-100 transition comparison

The copy operation is verified. The table below records the original audit,
which counted every valid transition from a non-100 code to code 100. It did
not count woodland codes 1 and 2 as existing canopy and did not exclude
ineligible source classes. Each pixel is 100 m2.

| Comparison | Green added pixels | Selected Target added pixels | Target deficit | Equal budget? |
|---|---:|---:|---:|---|
| Green10 vs Target10 (`510`/`710v2`) | 320,000 (32.0000 km2) | 276,430 (27.6430 km2) | 43,570 (13.616%) | No |
| Green20 vs Target20 (`520`/`730v2`) | 620,000 (62.0000 km2) | 542,985 (54.2985 km2) | 77,015 (12.422%) | No |
| Green30 vs legacy Target30 (`530`/`730v3`) | 930,000 (93.0000 km2) | 869,444 (86.9444 km2) | 60,556 (6.511%) | No |
| Green30 vs Target30 v4 (`730v4_equal_budget`) | 930,000 (93.0000 km2) | 930,000 (93.0000 km2) | 0 (0.000%) | Yes under code-100 audit only |

Therefore, describe `510/520/530` as selected historical Target trials or
aliases, not as equal-canopy scenarios. The code-100 figures remain useful
diagnostics but are superseded for production by the common definition below.

The machine-readable record is
[`target_scenario_alias_and_canopy_audit.csv`](target_scenario_alias_and_canopy_audit.csv).

## Revised common canopy and eligibility definition

The approved definition is:

```text
existing canopy = {1, 2, 100}
eligible planting source = {4, 20, 21}
replacement code = 100
rasterization = pixel centre (all_touched = false)
```

Under this definition, Green10/20/30 provide reference additions of 307,768,
595,084 and 894,249 cells. Historical Target10/20 and Target30 v3 contain
273,242, 535,737 and 858,121 eligible new-canopy cells respectively. Target30
v4 contains 917,794, which is 23,545 cells (2.633%) above Green30. Consequently
none of the current Target rasters is the final equal-area input.

Historical Green construction and the exact regeneration procedure are in
[`../lc_scenarios/README.md`](../lc_scenarios/README.md).

## Historical Target30 v4 audit

`LULC_Scenario730v4_equal_budget.tif` was generated separately on 9 September
2026. It extends v3 with the next-ranked candidate locations until the original
code-100 audit reaches 930,000 transitions (93 km2), equal to Green30 under
that limited definition. It removes no code-100 baseline cells. The adjacent
manifest records input and output checksums,
five prefix-sequence checks, software versions, feature counts and the rank
cutoff. Existing files were not overwritten.

The clean UCM rerun used InVEST 3.20.2 and is byte-identical to the earlier
3.20.2 Target30 workspace at both temperatures. These outputs are retained as
reproducibility evidence, but they are not the final common-eligibility
comparison.

## Current construction references

| Stage | File | Active reference |
|---|---|---|
| Allocate target trees | [`tree_equity_1_number_of_trees_to_polygon.py`](../lc_scenarios/tree_equity_1_number_of_trees_to_polygon.py) | `tree_equity_scenario710v3`, `720v3`, `730v3` |
| Rasterize target planting | [`tree_equity_2_scenario_engine.py`](../lc_scenarios/tree_equity_2_scenario_engine.py) | reads the three `v3` GeoPackages and writes `LULC_Scenario710v3.tif`, `720v3.tif`, `730v3.tif` |
| Calculate LULC statistics | [`tree_equity_3_lulc_stats.py`](../lc_scenarios/tree_equity_3_lulc_stats.py) | the `v3` rasters are commented out; `LULC_Scenario510.tif`, `520.tif`, `530.tif` are active |

These construction defaults do not reproduce the now-confirmed Target10 and
Target20 selections. They must be revised when the equal-budget Target10/20
calibration is implemented; until then, use the explicit mapping above.

## Current downstream references

The legacy UCM, health and plotting workflows still use `510/520/530`,
including:

- `code/Urban_Cooling_Modeling_Runs/Scenario_510_to_530_runs/`
- `code/health_assessment/health-modeling_s0_s{510,520,530}_2050_2050.bat`
- `code/health_assessment/health-modeling-output-plot-city.Rmd`
- `code/health_assessment/health-modeling-zonal-stats-viz-borough.Rmd`
- `code/equity-health.Rmd`
- `code/func_colors.R`

The legacy Target runners have only `[28, 5, 45]` active. The revised
publication convention is 25 C as primary and 28 C as sensitivity, always
paired with matching baseline and Green runs.

## Publication status and next action

| Label | Current selected raster | Budget status | Publication action |
|---|---|---|---|
| Target10 | `LULC_Scenario510.tif` (alias of `710v2`) | 34,526 eligible cells short | Rebuild from baseline to exactly 307,768 eligible cells, then rerun UCM and health |
| Target20 | `LULC_Scenario520.tif` (alias of `730v2`) | 59,347 eligible cells short | Rebuild from baseline to exactly 595,084 eligible cells, then rerun UCM and health |
| Target30 v4 | `LULC_Scenario730v4_equal_budget.tif` | 23,545 eligible cells over | Rebuild from baseline to exactly 894,249 eligible cells, then rerun UCM and health |

Preserve all existing rasters. Build all revised Targets from the baseline,
rather than extending historical rasters that already contain ineligible
transitions. Validate the selected ranking prefixes, restrict candidate cells
to baseline codes `{4,20,21}`, stop at the corresponding Green count, and
write manifests and nesting checks before replacing manuscript results.
