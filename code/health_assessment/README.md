# Health assessment

This folder estimates changes in heat-attributable mortality from spatial
temperature scenarios. All current analyses hold **2021 population and 2021
registered mortality** constant, including comparisons using 2050 temperature
fields.

Describe results as:

> 2050 temperature scenarios evaluated under 2021 population and mortality.

Do not describe them as projections of London's 2050 population or mortality.

Use 25°C as the primary manuscript setting and 28°C as a sensitivity analysis.
Keep population, mortality, exposure-response inputs and Monte Carlo settings
identical between them.

The rationale for mortality-source selection, fixed 2021 population,
population-weighted allocation and Target30 identity is recorded in
[`DATA_SELECTION_AND_METHOD_DECISIONS.md`](DATA_SELECTION_AND_METHOD_DECISIONS.md).
The quantitative old-versus-new results are in
[`HEALTH_VERSION_COMPARISON.md`](HEALTH_VERSION_COMPARISON.md).

## Workflow status

Two workflows currently coexist:

- **Revised workflow under validation:**
  `prepare_mortality_population_weighted.py` allocates borough mortality by
  2021 pixel population and records quality-control evidence.
- **Legacy workflow retained for comparison:**
  `health-model-01-prep-input-ONS-mortality-data.Rmd`, `health-modeling.py` and
  the `health-modeling_*.bat` launchers. These files remain unchanged so that
  old and revised results can be compared before replacement.

The revised mortality preparation, InVEST 3.20.2 UCM reruns, four deterministic
health runs, four 2,000-draw uncertainty runs and scripted LSOA11 aggregation
have passed project-data QA. Figure 7 now reads the revised LSOA table directly
and no longer depends on the opaque historical `health_sf.rds`.

## Why mortality allocation is changing

The legacy preparation divides each borough's deaths by borough land area and
assigns the resulting value to pixels. The revised method uses:

```text
pixel deaths = borough registered deaths
             × pixel population_2021
             / borough population_2021
```

This preserves each observed borough death total while locating the modeled
mortality burden where the 2021 population lives. Pixels with zero population
receive no allocated deaths.

The mortality source is the ONS Nomis dataset
[Mortality statistics—underlying cause, sex and age](https://www.nomisweb.co.uk/datasets/mortsa).
It reports numbers of deaths registered in each calendar year. Counts below
region level are disclosure-controlled, including rounding of small counts;
retain those source flags and limitations in the archived input metadata.

## Availability of LSOA mortality

The current Nomis query interface offers 2021 LSOAs only for 2023 onward. Its
historical 2013–2022 option is 2011 MSOA, not LSOA. A
[2022 ONS response](https://www.ons.gov.uk/aboutus/transparencyandgovernance/freedomofinformationfoi/lsoaaggregateddailyallcausemortalitydata)
states that LSOA mortality had been available through Nomis for 2013–2021, but that
historical LSOA option is not exposed by the current interface and should not be
assumed reproducibly available. ONS also has separate LSOA death releases
covering 2021-era periods, but the standard files identified during this review
provide all-cause counts by age/sex rather than the detailed causes used here.
See the ONS release
[Deaths by LSOA, mid-year periods 2010 to 2023](https://www.ons.gov.uk/peoplepopulationandcommunity/birthsdeathsandmarriages/deaths/adhocs/14319deathsbylowerlayersuperoutputarealsoaenglandandwalesmidyearperiods1julyto30june2011to2020),
which also records a November 2025 correction to its geography linkage.

Therefore, the current recommendation is:

1. retain the selected 2021 cause-specific borough death counts;
2. allocate each borough/cause total using 2021 pixel population;
3. preserve the borough totals exactly; and
4. contact `health.data@ons.gov.uk` about an archived or bespoke 2021
   cause-specific LSOA table if finer observed mortality becomes essential.

## Prepare the revised mortality rasters

### 1. Create the software environment

From the repository root:

```bash
conda env create -f code/health_assessment/environment-health.yml
conda activate urban-cooling-health
```

### 2. Add the original Nomis export

`mortality_2021_long.csv` is not a new external dataset. It is a standardized
intermediate created from the same Nomis table already read by
`health-model-01-prep-input-ONS-mortality-data.Rmd`.

The folder contains three archived Nomis exports. For the revised workflow use:

- `387056911188607_data.tsv`; and
- `387056911188607_geog.tsv` (retain for provenance, although the converter
  only needs the data TSV).

This export contains the five exact cause groups currently used by the health
model, including broad circulatory mortality (`I00-I99`). The newer
`842881776194293_data.tsv` referenced later in the exploratory R Markdown does
not contain `I00-I99`; it contains the narrower `I20-I25` and `I60-I69` groups.
Do not add those two groups as a substitute because they do not cover all
circulatory deaths.

The archived files currently reside in:

```text
data/health-data/
```

The converter removes exact duplicate rows present in the archived download,
but stops if duplicate borough/cause rows contain conflicting values.

### 3. Create the standardized mortality table

Run the converter rather than editing values manually:

```bash
python code/health_assessment/prepare_nomis_mortality_long.py \
  --nomis-data-tsv "data/health-data/387056911188607_data.tsv" \
  --output-csv "data/health-data/_processed/mortality_2021_long.csv"
```

The script selects calendar year 2021, the 33 `E09...` London boroughs and
these ICD-10 groups: all cause (`A00-R99, U00-Y89`), mental and behavioural
disorders (`F00-F99`), circulatory diseases (`I00-I99`), respiratory diseases
(`J00-J99`) and intentional self-harm (`X60-X84`). It stops if a cause is
missing or a borough/cause combination is duplicated.

The resulting table has one row per borough and cause:

Create a long CSV with one row per borough and cause:

```csv
borough_id,year,cause,deaths
E09000001,2021,all_cause,VALUE
E09000001,2021,cardiovascular,VALUE
```

The structural example `mortality_2021_long.example.csv` contains placeholders
only and must never be used in an analysis.

Required inputs:

- `OfficialWorkingInputs/AOIs/London_Borough_aoi.shp`, containing 33 polygons
  and the `GSS_CODE` field;
- the generated 2021 Nomis registered-death CSV above; and
- the raw WorldPop count raster
  `0_source_data/population_raster/gbr_pop_2021_CN_100m_R2025A_v1.tif` and a
  trusted UCM temperature raster defining the target grid.

Use a GeoPackage rather than a Shapefile when possible. If a Shapefile is used,
the manifest hashes all sidecar files with the same basename.

### 4. Reproject population counts without losing totals

Do not use `resampled_10m/gbr_pop_2021_10m_areal.tif`: its upstream bilinear
reprojection reduced London population from about 8.8 million to 4.74 million.
Preserve it for audit and create the production raster directly from the raw
WorldPop count raster using count-preserving sum resampling:

```bash
python code/health_assessment/prepare_population_2021_count_preserved.py \
  --source-population "$HEALTH_DATA_ROOT/0_source_data/population_raster/gbr_pop_2021_CN_100m_R2025A_v1.tif" \
  --reference-raster "$HEALTH_DATA_ROOT/2_postprocess_intermediate/UCM_official_runs/scenario0/work_and_energy_runs/intermediate/T_air_london_scenario_25.0deg_5.0uhi_45.0hum_energy_productivity.tif" \
  --boroughs "$HEALTH_DATA_ROOT/1_preprocess/UrbanCoolingModel/OfficialWorkingInputs/AOIs/London_Borough_aoi.shp" \
  --borough-field GSS_CODE \
  --output "$HEALTH_DATA_ROOT/1_preprocess/UrbanCoolingModel/OfficialWorkingInputs/pop_raster/resampled_10m/gbr_pop_2021_10m_count_preserved_bng.tif"
```

This step uses GDAL sum resampling for the extensive population counts, applies
the borough mask by pixel centre and checks the result against an 8.8 million
London benchmark.

To create the official-code population denominator for the production Figure 7
LSOA11 layer, run:

```bash
Rscript code/health_assessment/prepare_lsoa_population_2021.R \
  data/derived/fig7_vulnerability_lsoa11_nodata_harmonized.gpkg \
  "$HEALTH_DATA_ROOT/1_preprocess/UrbanCoolingModel/OfficialWorkingInputs/pop_raster/resampled_10m/gbr_pop_2021_10m_count_preserved_bng.tif" \
  data/derived/lsoa_population_2021_by_lsoa11cd.csv \
  data/derived/lsoa_population_2021_by_lsoa11cd.manifest.json
```

This uses the 4,835 production polygons and joins downstream data exclusively
by official `LSOA11CD`. The generated total is checked against both the source
raster and the official TS001 London total. The earlier numeric-ID table is
retained only as historical audit evidence.

WorldPop listing 135 and UK 2021 item 75899 identify this source as R2025A v1.
The official TIFF retains stale internal `R2024B v1` tags; a fresh download was
verified to have exactly the same grid and pixel values as the project source.
Use the listing, item, DOI `10.5258/SOTON/WP00839`, URL and checksum as the
authoritative provenance rather than the internal TIFF description.

### 5. Run the population-weighted allocation

```bash
python code/health_assessment/prepare_mortality_population_weighted.py \
  --boroughs "$HEALTH_DATA_ROOT/1_preprocess/UrbanCoolingModel/OfficialWorkingInputs/AOIs/London_Borough_aoi.shp" \
  --borough-field GSS_CODE \
  --mortality-csv "data/health-data/_processed/mortality_2021_long.csv" \
  --mortality-id-field borough_id \
  --population-2021 "$HEALTH_DATA_ROOT/1_preprocess/UrbanCoolingModel/OfficialWorkingInputs/pop_raster/resampled_10m/gbr_pop_2021_10m_count_preserved_bng.tif" \
  --year 2021 \
  --output-dir "$HEALTH_DATA_ROOT/1_preprocess/UrbanCoolingModel/OfficialWorkingInputs/health_rasters_population_weighted_2021_count_preserved"
```

The command refuses to overwrite existing outputs unless `--force` is supplied.
Review the existing run before using that option.

### 6. Review the outputs

The output directory contains:

- one `baseline_deaths_*_population_weighted_2021.tif` per cause;
- `allocation_qa.csv`, comparing observed and allocated totals by borough; and
- `manifest.json`, recording the method, input and output checksums, source,
  software versions, Git commit and population-coverage checks.

Do not continue unless:

- every required borough and cause appears exactly once;
- no borough has zero population;
- allocation differences are within the configured tolerance;
- the population-preparation manifest reports a plausible London total and no
  unintended population remains outside the borough mask;
- raster CRS, transform, resolution and extent match the intended 2021
  population grid; and
- the result has been compared with the legacy area-weighted output.

## Run the tests

```bash
python -m unittest \
  code/health_assessment/tests/test_prepare_nomis_mortality_long.py \
  code/health_assessment/tests/test_prepare_mortality_population_weighted.py -v
```

The current tests verify proportional allocation, exact borough-total
conservation, zero/outside-population handling and rejection of invalid inputs.
An end-to-end test with project geospatial data remains required.

## Run the revised health model

Before running the health model, use the reviewed Green30 LULC copy with the
same NoData sentinel (`0`) as baseline and Target30. Create it once with
`prepare_green30_nodata_harmonized.py`; the script refuses to overwrite files,
checks the two source grids, verifies that every valid land-cover value is
unchanged, and writes a checksum manifest. Then run Green30 in a separate
workspace by passing the resulting raster to
`execute_invest_urban_cooling_model_health_scenarios.py` with
`--green30-lulc` and `--green30-workspace`. Do not replace or edit the original
Green30 raster or historical InVEST workspace.

The new `health_modeling_v2.py` replaces the repeated batch-file arguments with
one JSON configuration and one scenario key. It does not silently align data:
all temperature, population and mortality rasters must already match the 2021
population grid exactly.

1. Copy `health-analysis-v2.example.json` to a local working configuration.
2. Set `HEALTH_DATA_ROOT` to the external project-data directory.
3. Review every input path. The configuration includes separate Green30 and
   Target30 runs for both the 25°C and 28°C settings.
4. Validate the inputs without writing model outputs:

   ```powershell
   $env:HEALTH_DATA_ROOT = "G:\Shared drives\Wellcome Trust Project Data"
   python code/health_assessment/health_modeling_v2.py `
     --config code/health_assessment/health-analysis-v2.example.json `
     --scenario green30_25c `
     --validate-only
   ```

5. After validation succeeds, remove `--validate-only` to run the model.

The revised runner:

- processes one raster window at a time rather than loading all inputs at once;
- uses one fixed 2021 population input;
- requires an exact common CRS, extent, resolution and transform;
- refuses missing files, missing CRS, negative values and unintended output
  overwrites;
- uses cause-stable random streams, so the same cause and seed receive identical
  exposure-response draws across scenarios;
- evaluates Monte Carlo city totals from 12 weighted delta-temperature moments;
  this is numerically equivalent to direct cell-by-draw evaluation at the
  tested range (unit-test tolerance `1e-12`) but completes in seconds rather
  than tens of minutes per scenario;
- saves both excess deaths and deaths averted in city summaries; and
- writes input checksums, configuration, software versions, coverage and Git
  commit to `run_manifest.json`.

Run each configured key separately: `green30_25c`, `target30_25c`,
`green30_28c` and `target30_28c`. The Target30 v4 temperature rasters must first
be generated under the separate `scenario730v4_equal_budget` workspace with
`execute_invest_urban_cooling_model_target30_equal_budget.py`. The older
`scenario530` outputs correspond to v3 and must not be substituted. See
`SCENARIO_NAMING_AUDIT.md` for checksums and the unresolved Target10/20
distinctions.

For each temperature setting, compare Green30 with Target30 only when both use
the matching baseline: 25°C with 25°C, and 28°C with 28°C. Do not mix the two
settings in one delta-temperature calculation.

## Existing health-model outputs

The legacy model produces:

- `deltaT_degC.tif`;
- `AF_{cause}.tif`;
- `Excess_{cause}.tif`;
- `city_totals_deterministic.csv`;
- `city_totals_monte_carlo.csv`; and
- `city_total_draws_by_cause.csv`.

Negative excess-death values represent avoided deaths when a scenario is cooler
than its baseline. The Figure 7 workflow reverses this sign when reporting
deaths averted.

For Green30 and Target30, retain at least 2,000 draws and use the same explicit
seed so exposure-response draws can be compared pairwise.

## Aggregate the revised outputs to LSOA11

After all four health scenarios finish, create the Figure 7 table and restore
official LSOA11 codes with:

```bash
python code/health_assessment/prepare_health_lsoa_zonal_stats.py \
  "$HEALTH_DATA_ROOT" \
  --output-csv data/derived/health_lsoa_invest3202_population_weighted_2021_nodata_harmonized.csv \
  --crosswalk-csv data/derived/svi_lsoa11_crosswalk_nodata_harmonized.csv \
  --vulnerability-gpkg data/derived/fig7_vulnerability_lsoa11_nodata_harmonized.gpkg \
  --manifest data/derived/health_lsoa_invest3202_population_weighted_2021_nodata_harmonized.manifest.json
```

The script checks a one-to-one match between the 4,835 vulnerability polygons
and the official 2011 London LSOAs. It assigns only valid boundary-edge cells
outside the generalized LSOA outline to the nearest LSOA, records the maximum
distance, and requires zero unassigned valid cells before writing outputs.

Then copy the paired 25°C draw tables to the filenames documented under
`data/derived/README.md` and run:

```bash
HEALTH_DATA_ROOT="$HEALTH_DATA_ROOT" Rscript code/run-fig7.R
```

## Target-scenario status

The final Target30 LULC is the approved equal-budget v4 raster. Its canopy audit,
InVEST 3.20.2 UCM run, population-weighted health run and Figure 7 regeneration
pass. Target10 and Target20 remain unresolved because their old/new raster pairs differ. The exact
affected files are listed in
[`SCENARIO_NAMING_AUDIT.md`](SCENARIO_NAMING_AUDIT.md).
