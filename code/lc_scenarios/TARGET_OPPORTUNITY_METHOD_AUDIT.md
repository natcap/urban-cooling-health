# Target tree-opportunity method audit

## Scope and decision

This audit applies to the **Target** scenarios only. The source Methods
document describes screened and ranked street-tree candidates; it does not
describe how the area-based Green scenarios were constructed and must not be
used as their primary eligibility definition.

The resulting decisions are:

- Target scenarios will be regenerated from the screened candidate points.
- Green scenarios retain their LCM-based design. The main analytical case uses
  `{4,20,21}` and the no-Urban sensitivity uses `{4,21}` at the same areas.
- Green scenarios constrained to the Target street-candidate footprint are
  exploratory cross-design tests only, not replacement manuscript inputs.

## Source record

The reviewed source is the external file:

```text
My Drive/NatCap/projects/KCL_Welcome/london-equity-tree-scenario/
└── Results/Methods.docx
```

SHA-256: `faab3a07c926a249431856434ced6cd245df3416a04a079b6bf0b44a4eba5a70`

The document records this Target workflow:

1. Select OSM pedestrian paths, footways and cycleways and place points every
   5 m along their centre lines.
2. For residential and unclassified vehicle roads, offset the centre line by
   3.5 m on both sides and place points every 5 m.
3. Remove points within a 5 m dilation of the 1 m existing-canopy raster.
4. Remove points intersecting a 2 m building-footprint buffer.
5. Remove clustered points until retained points are at least 5 m apart.
6. Extract 14:00 UTCI, normalize it to 0–1, join the SVI percentile, calculate
   `score = UTCI_norm * SVI_percentile`, and rank descending.

## Supplied-data checks

| Item | Observed evidence | Assessment |
|---|---|---|
| Ranked candidates | 3,378,103 Point features; EPSG:27700; fields include `UTCI_raw`, `UTCI_norm`, `overall_pc`, `score`, `raw_score`, `rank` | Suitable Target source |
| Score formula | Highest-ranked samples satisfy `score = UTCI_norm * overall_pc` | Consistent with Methods |
| UTCI raster | `UTCI_20220801_int16.tif`; 1 m; EPSG:27700; same London extent as the 10 m baseline | Spatial description confirmed |
| UTCI encoding | Point value 33.64 corresponds to raster value 3364 | A scale factor of 0.01 was used but is not stored in raster metadata |
| LCM baseline | My Drive and OfficialWorkingInputs copies have identical grids and cell values | Different file checksums reflect file-level encoding/metadata, not cell differences |
| OSM roads | 548,068 LineString features; EPSG:4326; includes `fclass` | Plausible source, but download date/version is missing |
| OSM railways | 15,478 LineString features; EPSG:4326 | Present, but Methods does not state whether railways were excluded |
| Tree canopy | 467 Polygon features; EPSG:27700 | Available for spatial checking |
| GiGL opportunities | 1,725 Polygon features; EPSG:27700 | Present, but Methods does not identify it as a Target input |
| Building footprints | Mentioned in Methods | The exact source file used for the 2 m exclusion is not present in this folder |

The ranked file contains tied scores/ranks. Each approved historical selection
ends within a tie group:

| Historical vector | Features | Maximum selected rank | Features selected at maximum rank |
|---|---:|---:|---:|
| `tree_equity_scenario710v2.gpkg` | 565,000 | 565,000 | 1 |
| `tree_equity_scenario730v2.gpkg` | 1,129,000 | 1,128,997 | 4 |
| `tree_equity_scenario730v3.gpkg` | 1,819,000 | 1,818,994 | 7 |

The full ranked file also ends with repeated rank `3,377,981` despite containing
3,378,103 features. Selecting with `nsmallest(N, "rank")` alone is therefore
not a stable rule at a tied cutoff. Revised Target construction must retain the
original `FID` and sort by `(rank, FID)` so the selected subset is deterministic.

## Rasterized Target opportunity capacity

[`build_tree_opportunity_mask.py`](build_tree_opportunity_mask.py) converts the
3,378,103 screened points to the 10 m baseline grid. A cell is a candidate when
its centre lies within the assumed 5 m crown radius of at least one point. This
matches the project's pixel-centre rasterization rule without creating millions
of temporary buffer polygons.

The versioned output is stored under:

```text
OfficialWorkingInputs/LULC/tree_opportunity_mask/
└── revised_v1_2026-09-10/
    ├── tree_opportunity_mask_10m.tif
    └── tree_opportunity_mask_audit.json
```

| Baseline class within candidate footprint | Candidate pixels | Area (km²) |
|---|---:|---:|
| 4 — Improved grassland | 145,479 | 14.5479 |
| 20 — Urban | 970,168 | 97.0168 |
| 21 — Suburban | 469,236 | 46.9236 |
| Eligible `{4,20,21}` total | **1,584,883** | **158.4883** |
| Eligible `{4,21}` without Urban | **614,715** | **61.4715** |

The screened Target candidate footprint is large enough for the 89.4249 km²
Target30 budget only when screened Urban candidates are retained. Removing
Urban leaves a 27.9534 km² shortfall (31.259%). In this Target workflow,
“Urban” is therefore not blanket permission to plant in every LCM Urban cell;
it is an underlying LCM label at a location already screened by the street,
canopy, building and spacing procedure.

## Green sensitivity results

The primary area-based Green outputs remain the LCM-based `{4,20,21}` version.
An equal-area no-Urban sensitivity was generated separately with `{4,21}`:

```text
OfficialWorkingInputs/LULC/lc_green_scenarios_output/
└── sensitivity_no_urban_equal_area_invest_3.20.2_2026-09-10/
```

| Scenario | Improved grassland pixels | Suburban pixels | Total pixels |
|---|---:|---:|---:|
| Green10 | 92,813 | 214,955 | 307,768 |
| Green20 | 181,685 | 413,399 | 595,084 |
| Green30 | 260,492 | 633,757 | 894,249 |

Compared with the main `{4,20,21}` Green version, the no-Urban scenario retains
76.38%, 73.71% and 71.78% spatial overlap at Green10, Green20 and Green30. It is
an equal-area policy sensitivity, not a planting-feasibility map.

The separately generated `revised_v3_common_opportunity...` and
`sensitivity_no_urban_capacity_limited...` folders apply the Target footprint
to Green. Their folder names predate the scope clarification: “common
opportunity” does not mean a scientifically shared Green–Target definition.
Preserve them as exploratory audit evidence, but do not use them for the main
Green–Target analysis unless this cross-design constraint is approved as a new
scientific assumption.

## Information still needed for full Target provenance

Request or record, when available:

- the script/notebook that created the ranked points;
- the OSM extract provider, URL, licence and download date;
- the exact road `fclass` values retained and any bridge/tunnel exclusions;
- the building-footprint filename, source, date, licence and CRS;
- the 1 m canopy-raster filename, date and dilation implementation;
- UTCI model provenance and the explicit `0.01` integer scale;
- the SVI source, geography, year and formula used for `overall_pc`; and
- software versions and checksums for all final Target inputs.

These gaps do not prevent a controlled regeneration from the supplied final
candidate points, but they prevent independent reconstruction of those points
from raw source data.
