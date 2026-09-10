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

## Realized canopy comparison

The copy operation is verified, but the legacy selections do not all have the
same realized added-canopy budget as their Green counterparts. Counts are
relative to `LCM2023_London_10m_clip2aoi_tcc24.tif`; each pixel is 100 m2.

| Comparison | Green added pixels | Selected Target added pixels | Target deficit | Equal budget? |
|---|---:|---:|---:|---|
| Green10 vs Target10 (`510`/`710v2`) | 320,000 (32.0000 km2) | 276,430 (27.6430 km2) | 43,570 (13.616%) | No |
| Green20 vs Target20 (`520`/`730v2`) | 620,000 (62.0000 km2) | 542,985 (54.2985 km2) | 77,015 (12.422%) | No |
| Green30 vs legacy Target30 (`530`/`730v3`) | 930,000 (93.0000 km2) | 869,444 (86.9444 km2) | 60,556 (6.511%) | No |
| Green30 vs final Target30 (`730v4_equal_budget`) | 930,000 (93.0000 km2) | 930,000 (93.0000 km2) | 0 (0.000%) | Yes |

Therefore, describe `510/520/530` as the selected historical Target trials or
aliases, not as proven equal-budget scenarios. Green10, Green20, Target10 and
Target20 may remain in the manuscript, but a like-for-like equity comparison
requires recalibrating Target10 and Target20 to 320,000 and 620,000 added
pixels respectively, then rerunning their UCM and health workflows.

The machine-readable record is
[`target_scenario_alias_and_canopy_audit.csv`](target_scenario_alias_and_canopy_audit.csv).

## Approved equal-budget Target30

`LULC_Scenario730v4_equal_budget.tif` was generated separately on 9 September
2026. It extends v3 with the next-ranked candidate locations until the realized
addition is exactly 930,000 tree pixels (93 km2), equal to Green30. It removes
no baseline canopy. The adjacent manifest records input and output checksums,
five prefix-sequence checks, software versions, feature counts and the rank
cutoff. Existing files were not overwritten.

The clean publication UCM rerun used InVEST 3.20.2 and is byte-identical to the
earlier 3.20.2 Target30 workspace at both temperatures. The existing revised
health outputs remain valid because their temperature-input checksums match
the publication rasters.

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
| Target10 | `LULC_Scenario510.tif` (alias of `710v2`) | 43,570 pixels short | Generate a separate equal-budget version, then rerun UCM and health at 25 C and 28 C |
| Target20 | `LULC_Scenario520.tif` (alias of `730v2`) | 77,015 pixels short | Generate a separate equal-budget version, then rerun UCM and health at 25 C and 28 C |
| Target30 | `LULC_Scenario730v4_equal_budget.tif` | Equal to Green30 | Complete; use the InVEST 3.20.2 publication outputs |

Recommended implementation for Target10/20: preserve all existing rasters;
extend each approved selected trial with the next-ranked eligible locations,
using the same prefix-preserving procedure validated for Target30, until the
exact corresponding Green budget is reached. Suggested new filenames are
`LULC_Scenario710v4_equal_budget.tif` and
`LULC_Scenario720v4_equal_budget.tif`. Generate manifests, validate no baseline
tree removal, and obtain scientific approval before replacing manuscript
results.
