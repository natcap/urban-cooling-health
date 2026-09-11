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
| Legacy zonal statistics and manuscript plots | notebooks and `*.Rmd` files listed below | Historical; migrate only the required plot logic |

The final scenario set is baseline plus Green10/20/30 and Target10/20/30,
generated from the revised equal-realized-canopy rasters and run with InVEST
3.20.2. Do not mix these outputs with the manuscript-era `scenario4` or
`scenario510/520/530` folders.

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
in the energy table. The energy table currently includes a `cost` column, so
InVEST's `energy_sav` result is cost-adjusted. Record the currency, price year
and period represented by that table before using a currency symbol in the
manuscript.

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

### 2. Run the seven scenarios

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
