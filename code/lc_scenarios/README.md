# Land-cover scenario construction

This folder contains the scripts and notebooks used to prepare the baseline
land-cover raster and construct the Green and vulnerability-targeted tree
scenarios. Read this file before running any scenario script: the historical
rasters are useful evidence, while the production Green/Target pairs now pass
the revised equal-realized-canopy and source-class checks.

## Current scientific definition

The revised comparison uses one canopy-accounting definition for every Green
and Target scenario, while retaining their different spatial designs:

```text
Existing canopy codes:        1, 2, 100
  1 = Broadleaved woodland
  2 = Coniferous woodland
  100 = Tree Cover

Allowed source codes:         4, 20, 21
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

### Complete land-cover code list

The baseline follows the UKCEH LCM2023 class codes, with project-specific code
`100` added for the 2024 tree-canopy overlay.

| Code | Land-cover class | Role in the revised scenarios |
|---:|---|---|
| 0 | NoData/background in the clipped baseline | Never convert |
| 1 | Broadleaved woodland | Existing canopy; focal only |
| 2 | Coniferous woodland | Existing canopy; focal only |
| 3 | Arable and horticulture | Not eligible |
| 4 | Improved grassland | Eligible planting class |
| 5 | Neutral grassland | Not eligible |
| 6 | Calcareous grassland | Not eligible |
| 7 | Acid grassland | Not eligible |
| 8 | Fen, marsh and swamp | Not eligible |
| 9 | Heather | Not eligible |
| 10 | Heather grassland | Not eligible |
| 11 | Bog | Not eligible |
| 12 | Inland rock | Not eligible |
| 13 | Saltwater | Not eligible |
| 14 | Freshwater | Not eligible |
| 15 | Supralittoral rock | Not eligible |
| 16 | Supralittoral sediment | Not eligible |
| 17 | Littoral rock | Not eligible |
| 18 | Littoral sediment | Not eligible |
| 19 | Saltmarsh | Not eligible |
| 20 | Urban | Eligible planting class |
| 21 | Suburban | Eligible planting class |
| 100 | Project tree-canopy overlay/new canopy | Existing canopy, focal and replacement |

Codes `4`, `20` and `21` are shared allowed source classes, not a claim that
every cell is physically or legally plantable. Green uses these broad classes
to define an idealized area-based canopy scenario. Target additionally uses
screened street-tree candidate points. The Target Methods document does not
define Green eligibility; see
[`TARGET_OPPORTUNITY_METHOD_AUDIT.md`](TARGET_OPPORTUNITY_METHOD_AUDIT.md).

## Required external data layout

Large rasters and vectors are intentionally outside Git. In the shared data
root, this workflow uses:

```text
1_preprocess/UrbanCoolingModel/OfficialWorkingInputs/LULC/
├── LCM2023_London_10m_clip2aoi_tcc24.tif
├── LCM2023_London_10m_clip2aoi_tcc24_scenario4_nearest_to_edge_10prc_canopy_increase.tif
├── LCM2023_London_10m_clip2aoi_tcc24_scenario4_nearest_to_edge_20prc_canopy_increase.tif
├── LCM2023_London_10m_clip2aoi_tcc24_scenario4_nearest_to_edge_30prc_canopy_increase.tif
├── lc_tree_equity_scenarios_output/
    ├── tree_equity_scenario710.gpkg
    ├── tree_equity_scenario710v2.gpkg
    ├── tree_equity_scenario730.gpkg
    ├── tree_equity_scenario730v2.gpkg
    ├── tree_equity_scenario730v3.gpkg
    ├── LULC_Scenario510.tif
    ├── LULC_Scenario520.tif
    └── LULC_Scenario530.tif
    └── revised_v2_rank_fid_equal_area_2026-09-10/
        ├── LULC_Target10_equal_area_eligible_v2.tif
        ├── LULC_Target20_equal_area_eligible_v2.tif
        ├── LULC_Target30_equal_area_eligible_v2.tif
        ├── target_scenario_manifest_revised.json
        ├── target_transition_audit.csv
        ├── green_target_pair_qa.json
        ├── green_target_pair_qa.csv
        └── target_change_masks.png
└── lc_green_scenarios_output/
    ├── revised_v2_invest_3.20.2_2026-09-10/
        ├── Green10_equal_area_eligible_v2.tif
        ├── Green20_equal_area_eligible_v2.tif
        ├── Green30_equal_area_eligible_v2.tif
        ├── green_scenario_manifest_revised.json
        └── invest_workspaces/
    └── sensitivity_no_urban_equal_area_invest_3.20.2_2026-09-10/
        ├── Green10_equal_area_eligible_v2.tif
        ├── Green20_equal_area_eligible_v2.tif
        ├── Green30_equal_area_eligible_v2.tif
        └── green_scenario_manifest_revised.json
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

The preserved rasters and recovered InVEST logs establish the following facts:

- the baseline was `LCM2023_London_10m_clip2aoi_tcc24.tif`;
- conversion followed the “nearest to edge” output pattern;
- converted cells were assigned replacement code `100`;
- the 10/20/30 outputs changed 320,000, 620,000 and 930,000 cells to `100`;
- source codes changed were `1`, `2`, `4`, `20` and `21`; and
- the scenarios were generated as separate outputs and are not perfectly
  nested: 4,451 Green10 conversion cells are absent from Green20, and 11,709
  Green20 conversion cells are absent from Green30.

The logs were recovered from the shared-data folder
`EP_preliminary_tests/clipped_lulc/UKECH/`. They confirm that the main runs used
InVEST **3.14.1**, no AOI, nearest-to-edge conversion, two fragmentation steps,
replacement code `100`, and the following parameters:

The colleague-supplied Green20 log is retained unchanged in this folder as
`InVEST-natcap.invest.scenario_gen_proximity-log-2025-09-18--10_53_40.txt`.
Its SHA-256 is
`dd3e32890f4d3c3edf6199e862f5165f13f003d18ab9a7f8754ba6a3a4b14974`.

| Scenario | Maximum area (ha) | Focal codes | Convertible codes |
|---|---:|---|---|
| Green10 | 3,200 | `1 2 4 20 21` | `1 2 4 20 21` |
| Green20 | 6,200 | `1 2 4 20 21` | `1 2 4 20 21` |
| Green30 | 9,300 | `1 2 4 20 21` | `1 2 4 20 21` |

This confirms two methodological problems. First, woodland codes `1` and `2`
were eligible for conversion to `100`, so part of each reported increase was a
label change rather than added canopy. Second, the focal list omitted the
project's explicit canopy code `100` and instead included every convertible
class. “Nearest to edge” therefore did not mean nearest to the complete
existing-canopy definition `{1,2,100}`.

A controlled rerun of the confirmed inputs under InVEST 3.20.2 reproduced the
total conversion budgets exactly, but not every selected cell. Relative to the
preserved 3.14.1 outputs, 9,594, 18,970 and 23,276 full-raster cell values
differed for Green10, Green20 and Green30, respectively (0.036%, 0.072% and
0.088% of all cells). Exact historical reproduction therefore requires the
archived 3.14.1 software environment; InVEST 3.20.2 provides a close method
replication and should be used consistently for all new production scenarios.

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

### Rough tree-equivalent interpretation

A 10 m raster cell represents canopy area, not one tree. For a rough planning
translation, the existing Target workflow buffers tree points by 5 m, implying
a 10 m mature crown diameter and about 78.54 m² of crown area per tree. Dividing
eligible added-canopy area by that crown area gives:

| Scenario | Eligible pixels | Added canopy (km²) | Approx. 10 m-crown tree equivalents | Sensitivity for 8–12 m crowns |
|---|---:|---:|---:|---:|
| Green10 | 307,768 | 30.7768 | ~392,000 | ~272,000–612,000 |
| Green20 | 595,084 | 59.5084 | ~758,000 | ~526,000–1,184,000 |
| Green30 | 894,249 | 89.4249 | ~1,139,000 | ~791,000–1,779,000 |

These are **mature-crown equivalents**, not planting-stem requirements. Actual
planting counts require species mix, expected mature crown diameter, spacing,
crown overlap, establishment mortality and replacement assumptions. Report
pixels or area as the primary scenario quantity and use tree equivalents only
as a clearly labelled approximation.

## Recommended Green10/20/30 regeneration

Use [`generate_green_scenarios_invest.py`](generate_green_scenarios_invest.py)
with InVEST 3.20.2. The recommended profile changes the historical settings as
follows:

| Parameter | Recommended setting | Reason |
|---|---|---|
| Focal codes | `1 2 100` | Represents all existing canopy used in this project |
| Convertible codes | `4 20 21` | Uses the shared allowed source classes while retaining Green's area-based design |
| Replacement code | `100` | Preserves the project-specific new-canopy class |
| Direction | nearest to edge only | Expands canopy outwards from existing canopy |
| Conversion steps | `1` per tier | Makes each tier's distance calculation explicit and avoids an arbitrary within-tier recalculation count |
| Scenario sequence | baseline → Green10 → Green20 → Green30 | Guarantees nesting |
| Cumulative budgets | 307,768; 595,084; 894,249 cells | Matches eligible canopy actually added by the historical Green rasters |
| AOI | none | The baseline is already clipped to the London study grid |
| NoData | restore baseline value `0` | Avoids the previous `0`/`255` mismatch in downstream UCM runs |

Run the revised profile with:

```bash
conda run -n urban-cooling-invest-3.20.2 python \
  code/lc_scenarios/generate_green_scenarios_invest.py \
  --baseline /path/to/LCM2023_London_10m_clip2aoi_tcc24.tif \
  --output-dir /path/to/new/versioned/green_scenarios \
  --profile revised
```

The runner keeps the InVEST workspaces, writes harmonized final rasters, checks
that only codes `{4,20,21}` changed to `100`, enforces the exact cumulative
budgets and writes a JSON manifest with parameters, transitions and SHA-256
checksums. A temporary test run under 3.20.2 produced exactly 307,768, 595,084
and 894,249 eligible cells, with Green10 fully inside Green20 and Green20 fully
inside Green30.

The verified production run is stored under
`lc_green_scenarios_output/revised_v2_invest_3.20.2_2026-09-10/` in the shared
LULC folder. Its final transitions from the original baseline are:

| Scenario | `4 -> 100` | `20 -> 100` | `21 -> 100` | Eligible total |
|---|---:|---:|---:|---:|
| Green10 | 69,962 | 72,704 | 165,102 | 307,768 |
| Green20 | 138,011 | 148,736 | 308,337 | 595,084 |
| Green30 | 194,268 | 242,156 | 457,825 | 894,249 |

### No-Urban Green sensitivity

Because LCM Urban mixes potentially plantable hard-surface contexts with
buildings and other unsuitable surfaces, an equal-area Green sensitivity was
also generated using only `{4,21}`. It keeps the same focal codes, sequence and
cumulative areas as the main Green scenarios:

| Scenario | `4 -> 100` | `21 -> 100` | Total |
|---|---:|---:|---:|
| Green10 | 92,813 | 214,955 | 307,768 |
| Green20 | 181,685 | 413,399 | 595,084 |
| Green30 | 260,492 | 633,757 | 894,249 |

Spatial overlap with the main `{4,20,21}` Green masks is 76.38%, 73.71% and
71.78% at Green10, Green20 and Green30. Use these outputs to test sensitivity
to Urban inclusion, not as proof that every code-4 or code-21 cell is a
feasible planting site.

For a version-comparison audit only, use `--profile historical-audit`. Do not
use that profile for manuscript production because it intentionally retains
the historical focal/eligibility problems.

## How the historical Target trials were created

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

The supplied `Methods.docx` confirms that Target candidates were generated
from selected OSM street classes and filtered by existing-canopy distance,
building distance and point spacing before being ranked by UTCI and social
vulnerability. The full evidence and remaining provenance gaps are recorded in
[`TARGET_OPPORTUNITY_METHOD_AUDIT.md`](TARGET_OPPORTUNITY_METHOD_AUDIT.md).
The ranked data contain tied scores at each historical selection cutoff, so
revised selection must use the retained `FID` as a deterministic secondary
sort key rather than `rank` alone.

### Revised Target10/20/30 results

[`generate_target_scenarios.py`](generate_target_scenarios.py) sorts all
3,378,103 screened candidates by `(rank, FID)`, accepts unique pixel centres
strictly within a 5 m candidate radius, restricts conversions to `{4,20,21}`
and stops at the revised Green budgets. The production outputs are nested and
contain:

| Scenario | `4 -> 100` | `20 -> 100` | `21 -> 100` | Total |
|---|---:|---:|---:|---:|
| Target10 | 21,412 | 219,337 | 67,019 | 307,768 |
| Target20 | 43,287 | 411,309 | 140,488 | 595,084 |
| Target30 | 67,752 | 602,026 | 224,471 | 894,249 |

Urban accounts for 71.3%, 69.1% and 67.3% of revised Target additions, versus
23.6%, 25.0% and 27.1% in the corresponding revised Green scenarios. This is
not blanket conversion of LCM Urban: every Target location comes from the
screened street-tree candidate dataset. Nevertheless, this composition
difference must be reported when interpreting the Green–Target comparison.

Relative to the manuscript's historical Target aliases, all historically
eligible cells are retained. The revised scenarios add 34,526, 59,347 and
36,128 eligible cells and exclude 3,188, 7,248 and 11,323 historical changes
from ineligible land-cover classes at levels 10, 20 and 30, respectively.

Run the generator and independent paired QA with:

```bash
python code/lc_scenarios/generate_target_scenarios.py \
  --baseline /path/to/LCM2023_London_10m_clip2aoi_tcc24.tif \
  --candidate-points /path/to/Potential_Tree_Points_Ranked.shp \
  --output-dir /path/to/new/versioned/target_scenarios \
  --repo-root /path/to/urban-cooling-health

python code/lc_scenarios/validate_green_target_scenarios.py \
  --baseline /path/to/baseline.tif \
  --green /path/to/Green10.tif /path/to/Green20.tif /path/to/Green30.tif \
  --target /path/to/Target10.tif /path/to/Target20.tif /path/to/Target30.tif \
  --output-json /path/to/green_target_pair_qa.json \
  --output-csv /path/to/green_target_pair_qa.csv
```

The paired QA passed exact budgets, allowed transitions, NoData/grid identity,
canopy retention and nesting. Same-level Green–Target overlap is 0.85%, 2.03%
and 3.64%, confirming that the area-based and vulnerability-targeted methods
allocate the same canopy quantities to substantially different locations.

## Completed revised regeneration procedure

Historical files were not overwritten. The revised workflow performs these
stages in order:

1. **Validate complete transitions.** Use
   [`validate_green_target_scenarios.py`](validate_green_target_scenarios.py)
   with canopy codes `{1,2,100}` and source codes `{4,20,21}`.
2. **Generate revised Green scenarios.** Run the recommended InVEST profile
   above. Expected cumulative new-canopy counts are 307,768, 595,084 and
   894,249 cells.
3. **Validate the target ranking.** Use the supplied screened candidate points,
   preserve `FID`, and sort by `(rank, FID)`. Record the tie group and final
   selected `FID` at every scenario cutoff.
4. **Build each Target from the baseline.** Rasterize candidates in ascending
   rank, accept only baseline `{4,20,21}` cells, and stop at the corresponding
   corrected Green count. Do not extend a historical raster containing
   ineligible changes.
5. **Run scenario QA.** Require identical grid/NoData, exact Green–Target
   counts, no changes outside `{4,20,21}`, no lost canopy, and nested
   10-within-20-within-30 masks. Write a checksum manifest for every output.
6. **Review before modeling.** Review the change-mask preview, transition
   tables, rank cutoffs and paired QA. Then validate InVEST 3.20.2 at the
   approved 25 C primary and 28 C sensitivity settings before running it.

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
| `generate_green_scenarios_invest.py` | Reproduce historical parameters or generate corrected nested Green scenarios | Recommended Green generator; requires InVEST 3.20.2 |
| `build_tree_opportunity_mask.py` | Rasterize screened Target candidates using a 5 m crown and pixel-centre rule | Target audit and regeneration input; not the primary Green definition |
| `TARGET_OPPORTUNITY_METHOD_AUDIT.md` | Record Target Methods evidence, data checks, capacity and provenance gaps | Current decision record |
| `generate_target_scenarios.py` | Sort screened candidates by `(rank, FID)` and generate exact nested Target rasters | Recommended Target generator |
| `validate_green_target_scenarios.py` | Independently validate the six revised Green/Target rasters | Recommended scenario gate |
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
