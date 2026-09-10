# Land-cover scenario construction

This folder contains the scripts and notebooks used to prepare the baseline
land-cover raster and construct the Green and vulnerability-targeted tree
scenarios. Read this file before running any scenario script: the historical
rasters are useful evidence, but the current Green/Target pairs do not yet meet
the revised common-eligibility and equal-realized-canopy definition.

## Current scientific definition

The revised comparison uses one definition for every Green and Target scenario:

```text
Existing canopy codes:        1, 2, 100
  1 = Broadleaved woodland
  2 = Coniferous woodland
  100 = Tree Cover

Eligible planting codes:      4, 20, 21
  4 = Improved grassland
  20 = Urban
  21 = Suburban

New tree code:                100
Rasterization rule:           pixel centre (all_touched = false)
Reference grid:               baseline 10 m EPSG:27700 raster
```

“Added canopy” means a valid baseline cell in `{4,20,21}` that becomes `100`.
A transition from woodland code `1` or `2` to `100` is a relabeling, not new
canopy. Changes from water, wetland, coastal habitat, rock or NoData are not
eligible.

## Required external data layout

Large rasters and vectors are intentionally outside Git. In the shared data
root, this workflow uses:

```text
1_preprocess/UrbanCoolingModel/OfficialWorkingInputs/LULC/
├── LCM2023_London_10m_clip2aoi_tcc24.tif
├── LCM2023_London_10m_clip2aoi_tcc24_scenario4_nearest_to_edge_10prc_canopy_increase.tif
├── LCM2023_London_10m_clip2aoi_tcc24_scenario4_nearest_to_edge_20prc_canopy_increase.tif
├── LCM2023_London_10m_clip2aoi_tcc24_scenario4_nearest_to_edge_30prc_canopy_increase.tif
└── lc_tree_equity_scenarios_output/
    ├── tree_equity_scenario710.gpkg
    ├── tree_equity_scenario710v2.gpkg
    ├── tree_equity_scenario730.gpkg
    ├── tree_equity_scenario730v2.gpkg
    ├── tree_equity_scenario730v3.gpkg
    ├── LULC_Scenario510.tif
    ├── LULC_Scenario520.tif
    └── LULC_Scenario530.tif
```

Do not commit the source, scenario or UCM rasters until the UKCEH licence
determination in the repository-level data-licence review is complete.

## How the baseline was created

The baseline starts with the 10 m UKCEH LCM2023 London raster. The notebook
[`reclassify-lulc-using-tcc.ipynb`](reclassify-lulc-using-tcc.ipynb) rasterizes
`TreeCanopyCover24_stitched.shp` to the LCM grid using the pixel-centre rule and
assigns intersecting valid cells the new code `100`. The resulting file is:

```text
LCM2023_London_10m_clip2aoi_tcc24.tif
```

Codes `1` and `2` remain in the raster outside the 2024 tree-canopy polygons,
so all three codes `{1,2,100}` must be treated as existing canopy when
measuring total tree-covered land.

## How the current Green10/20/30 rasters were created

The current Green rasters have filenames produced by the InVEST
[Scenario Generator: Proximity Based](https://storage.googleapis.com/releases.naturalcapitalproject.org/invest-userguide/latest/en/scenario_gen_proximity.html):

```text
LCM2023_London_10m_clip2aoi_tcc24_scenario4_nearest_to_edge_10prc_canopy_increase.tif
LCM2023_London_10m_clip2aoi_tcc24_scenario4_nearest_to_edge_20prc_canopy_increase.tif
LCM2023_London_10m_clip2aoi_tcc24_scenario4_nearest_to_edge_30prc_canopy_increase.tif
```

The preserved rasters establish the following facts:

- the baseline was `LCM2023_London_10m_clip2aoi_tcc24.tif`;
- conversion followed the “nearest to edge” output pattern;
- converted cells were assigned replacement code `100`;
- the 10/20/30 outputs changed 320,000, 620,000 and 930,000 cells to `100`;
- source codes changed were `1`, `2`, `4`, `20` and `21`; and
- the scenarios were generated as separate outputs and are not perfectly
  nested: 4,451 Green10 conversion cells are absent from Green20, and 11,709
  Green20 conversion cells are absent from Green30.

The original InVEST Scenario Generator log or datastack was not found. The
exact historical InVEST version, focal-code list, convertible-code list,
requested maximum areas, AOI and conversion-step count therefore remain
unverified. The output filenames and cell transitions support the method above
but are not a substitute for those missing run parameters. The non-nested
outputs also show that the three levels cannot be reconstructed as simple
prefixes of one preserved distance ranking. This is method reconstruction, not
exact historical reproduction.

### Observed Green transitions

| Baseline code changed to 100 | Green10 | Green20 | Green30 |
|---|---:|---:|---:|
| 1 — Broadleaved woodland | 10,224 | 21,062 | 30,348 |
| 2 — Coniferous woodland | 2,008 | 3,854 | 5,403 |
| 4 — Improved grassland | 82,950 | 163,622 | 240,117 |
| 20 — Urban | 72,055 | 145,763 | 229,866 |
| 21 — Suburban | 152,763 | 285,699 | 424,266 |
| All transitions to 100 | 320,000 | 620,000 | 930,000 |
| Eligible new canopy from `{4,20,21}` | **307,768** | **595,084** | **894,249** |

The machine-readable table is
[`historical_green_transition_audit.csv`](historical_green_transition_audit.csv).

The original 320,000/620,000/930,000 totals counted woodland relabeling as
added canopy. They must not be used as the stopping targets for the revised
Target scenarios.

## How the current Target trials were created

[`tree_equity_1_number_of_trees_to_polygon.py`](tree_equity_1_number_of_trees_to_polygon.py)
reads ranked potential-tree points, selects prefixes of the ascending `rank`
field, buffers each point by 5 m, and writes scenario GeoPackages.
[`tree_equity_2_scenario_engine.py`](tree_equity_2_scenario_engine.py) then
rasterizes the buffers with `all_touched=False` and assigns code `100`.

The confirmed manuscript aliases are:

```text
Target10: tree/raster trial 710v2 -> LULC_Scenario510.tif
Target20: tree/raster trial 730v2 -> LULC_Scenario520.tif
Target30: tree/raster trial 730v3 -> LULC_Scenario530.tif
```

The existing rasterizer excludes only NoData. It does not enforce allowed
source codes, which is why the historical Target rasters include changes from
water, wetland, coastal and woodland classes.

## Revised regeneration procedure

Do not overwrite historical files. The next implementation should perform
these stages in order:

1. **Generalize the transition audit.** Update
   [`validate_scenario_canopy_budget.py`](validate_scenario_canopy_budget.py)
   to accept canopy codes `{1,2,100}` and eligible source codes `{4,20,21}` and
   to export the full source-to-target transition matrix.
2. **Create corrected Green copies from the baseline.** Retain historical
   Green conversions only where the baseline is `{4,20,21}`. Expected new
   canopy counts are 307,768, 595,084 and 894,249 cells.
3. **Validate the target ranking.** Confirm that each selected trial vector is
   a geometry-and-rank prefix of its approved full ranked layer. Stop if any
   prefix check fails.
4. **Build each Target from the baseline.** Rasterize candidates in ascending
   rank, accept only baseline `{4,20,21}` cells, and stop at the corresponding
   corrected Green count. Do not extend a historical raster containing
   ineligible changes.
5. **Run scenario QA.** Require identical grid/NoData, exact Green–Target
   counts, no changes outside `{4,20,21}`, no lost canopy, and nested
   10-within-20-within-30 masks. Write a checksum manifest for every output.
6. **Review before modeling.** Visually compare the six change masks and review
   the transition tables and rank cutoffs. Only then run InVEST 3.20.2 at the
   approved 25 C primary and 28 C sensitivity settings.

Suggested new filenames:

```text
Green10_equal_area_eligible_v2.tif
Green20_equal_area_eligible_v2.tif
Green30_equal_area_eligible_v2.tif
LULC_Target10_equal_area_eligible_v2.tif
LULC_Target20_equal_area_eligible_v2.tif
LULC_Target30_equal_area_eligible_v2.tif
```

## Script inventory

| File | Purpose | Current status |
|---|---|---|
| `reclassify-lulc-using-tcc.ipynb` | Overlay 2024 tree-canopy polygons as code 100 | Baseline lineage; retains machine-specific paths |
| `scenario_1_pavement_and_2_opportunity_trees.ipynb` | Early pavement and opportunity-tree scenarios | Exploratory/legacy; not the Green10/20/30 generator |
| `tree_equity_1_number_of_trees_to_polygon.py` | Select and buffer ranked Target candidates | Historical construction; configuration is hard-coded |
| `tree_equity_2_scenario_engine.py` | Burn Target buffers into baseline | Historical construction; lacks eligibility masking |
| `tree_equity_3_lulc_stats.py` | Summarize LULC classes | Diagnostic; active filenames are legacy aliases |
| `validate_scenario_canopy_budget.py` | Compare code-100 transitions | Diagnostic only until canopy/eligibility sets are generalized |
| `calibrate_target30_canopy_budget.py` | Extend legacy Target30 to a code-100 cell target | Historical audit evidence; do not use for final regeneration |
| `lc_scenario_change_detection*.py` | Inspect and visualize raster changes | Diagnostic |
| `tree-at-climate-risk-*` | Construct the separate tree-risk scenario | Separate workflow; not part of Green–Target equal-area comparison |

## Reproducibility record for every new scenario

Retain together:

- input and output SHA-256 checksums;
- Git commit and clean/dirty state;
- Python, GDAL, rasterio, pyogrio, numpy and InVEST versions;
- canopy, eligibility and replacement-code definitions;
- CRS, transform, dimensions, pixel area and NoData value;
- full transition matrix and realized area;
- ranked-vector layer, feature count and final rank cutoff;
- nesting and prefix-validation results; and
- a small PNG change-mask preview for human review.
