# Daily-mean temperature and canopy-cooling checks

12 November 2026. Diagnostic only; no production outputs, figures, mortality results or Word files changed. No push. This is an assessment of the evidence available locally, not successful validation of an intervention model.

## Decision

**Do not select full or half cooling transfer from these results.** The temperature definition changes threshold exceedance, and the canopy association is sensitive to adjustment. Canopy does not improve geographic holdout prediction in the tested design. The available data therefore fail to establish a defensible daily-mean intervention-cooling factor.

Retain the [London health-response pilot](health_calculation_nodata_audit.md#london-all-cause-response-pilot-12-november-completed-not-adopted) as conditional sensitivity evidence. Its existing numerical results remain unchanged. This report supplies the temperature-validation evidence; do not repeat its tables in the main health audit.

## Literature justification

| Source and access | Supported choice | Boundary |
|---|---|---|
| [Arbuthnott et al. 2020](https://discovery.ucl.ac.uk/id/eprint/10106443/), original Methods 2.1–2.2 and Table 2 reviewed in the linked health audit | London all-cause response uses daily mean defined as `(Tmax + Tmin)/2`, threshold 18.9°C and heat lags 0–2 | A 24-hour hourly average is not exactly this metric. Central-grid exposure cannot automatically be replaced by a park station or every local grid cell without checking compatibility. |
| [Taylor et al. 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11009716/), full-text Methods 2.1–2.3 | London canopy–air-temperature modelling accounts for buildings and checks unseen stations before applying a mortality response | Their observational GAM is not causal planting validation. Our coarse land-cover covariate is not their building-height/footprint data. Their metric/lag descriptions differ from the original response source; do not copy uncritically. |
| [InVEST documentation](https://storage.googleapis.com/releases.naturalcapitalproject.org/invest-userguide/latest/en/urban_cooling_model.html), cooling formulation and limitations | Check spatial temperature patterns and cooling-parameter sensitivity | A reference-temperature adjustment changes the offset, not validation of canopy response. Static 25/5°C conditions are not observed summer weather. |
| [Met Office Crowd-Grid catalogue](https://catalogue.ceda.ac.uk/uuid/1ee84eb83cf8406e8ec86f914aaf172d/), product description | Provides daily maximum/minimum air temperature at 1-km resolution; existing Heat Packs average these over MSOAs | Prototype including official and crowdsourced observations. Comparison with MIDAS is not necessarily independent; MSOAs are not independent sensors. |

This is a targeted methodological check, not a systematic literature review. The key distinction is between validating baseline exposure and estimating the change caused by additional canopy. Urban–rural UHI contrasts alone do not identify that intervention effect.

## Data and design

- Existing Crowd-Grid Heat Packs: 1,002 London MSOAs, June–August 2018–2023 (552 dates initially). Use the midpoint of matched daily maximum/minimum values, not maximum temperature alone.
- Quality check found 83 MSOA maximum/minimum inversions, all on **18 June 2021**. Retain the original data unchanged; exclude this date across all areas and metrics, giving **551 common dates**. Do not swap extrema. Separate interpolation or observation-window definitions could contribute; their cause is unresolved. This is a sensitivity-data flag, not proof of instrument failure.
- Baseline canopy proxy: project LCM2023 plus 2024 canopy overlay, codes 1, 2 and 100. Urban/suburban fraction: codes 20 and 21. Neither is a direct measurement of building height or footprint.
- Rasterize MSOA polygons to the existing 10-m UCM grid; aggregate land-cover fractions and corrected baseline/Green30/Target30 temperatures on the same valid cells. All 1,002 MSOAs pass the prespecified 95% coverage rule; minimum coverage is 99.25%. Equal-weight MSOAs are used for these spatial diagnostics, not population-weighted citywide health accounting.
- Compare each MSOA's six-summer mean temperature with three linear specifications: canopy only; canopy plus urban/suburban cover; and both plus quadratic geographic terms (easting, northing, squares and interaction). Repeat associations by year. Do not treat the 551 repeated dates as independent spatial replicates or report naive pixel-level significance.
- Spatial holdout: five contiguous easting strips, each withheld in turn. Compare a built-cover/geography model with and without canopy. This tests extrapolation across strips, not random-cell interpolation, and is only one spatial partition.
- Station comparison: the already quality-controlled 2021 St James's Park hourly series against its containing **Westminster 018** MSOA. There are 91 common JJA dates after excluding 18 June. Hourly extrema are proxies for official daily extrema, not replacements validated here. Original station dates are retained when forming lag histories.

## Findings

### 1. Daily-temperature definitions affect exposure classification

Differences below are relative to the station's `(hourly maximum + hourly minimum)/2` on the same 91 dates.

| Series | Mean difference (°C) | RMSE (°C) | Days above 18.9°C, unlagged | Days above 18.9°C, lag 0–2 mean |
|---|---:|---:|---:|---:|
| Station extrema midpoint | 0 | 0 | 31 | 26 |
| Station hourly average | −0.217 | 0.520 | 28 | 22 |
| Containing MSOA Crowd-Grid midpoint | +0.569 | 0.700 | 39 | Not calculated |
| Equal-weight London MSOA midpoint average | +0.223 | 0.447 | 37 | Not calculated |

Grid lag counts were not calculated because the selected summer subset lacks pre-June history; no dates were imputed. The original station pilot's 92-day count of 26 remains unchanged. These differences combine metric, spatial support, observation-window and product effects; they do not isolate a measurement bias or prove either source correct. The London MSOA average is not a population-weighted exposure. Actual daily-extrema MIDAS files were not among the hourly inputs used here and were not newly acquired.

### 2. Canopy associations are not robust enough to calibrate planting benefits

Coefficients are °C associated with **10 percentage points** more baseline canopy—not a 10% relative canopy increase and not the effect of Green10.

| Specification | Daily maximum | Daily minimum | Daily midpoint |
|---|---:|---:|---:|
| Canopy only | −0.159 | −0.137 | −0.148 |
| Adjust urban/suburban fraction | −0.055 | +0.098 | +0.022 |
| Also adjust quadratic geography | −0.163 | −0.075 | −0.119 |

Canopy and urban/suburban fractions correlate at −0.651. The sign change demonstrates confounding/specification sensitivity. The geographically adjusted midpoint association is negative in each year (−0.150 to −0.099°C per 10 percentage points), but that consistency does not eliminate spatial confounding or identify a causal intervention effect.

### 3. Canopy does not improve spatial holdout performance in this test

| Metric | RMSE without canopy (°C) | RMSE with canopy (°C) |
|---|---:|---:|
| Maximum | 0.481 | 0.499 |
| Minimum | 0.253 | 0.254 |
| Midpoint | 0.290 | 0.300 |

Training fit alone would overstate support. This result does not establish that trees have no cooling effect: the coarse product, correlated land-cover variables, missing built-form/elevation information and geographic extrapolation can obscure it. Additional folds or more flexible fits should not be selected merely to produce a preferred sign.

### 4. Current UCM patterns are only partly concordant

The corrected static 25/5°C baseline's MSOA means correlate **0.472** with six-summer Crowd-Grid midpoints. After removing each field's citywide mean, their RMSE is **0.389°C**. This measures spatial-pattern agreement only; comparing raw absolute levels would confound assumed warm conditions with historical weather.

Using the same built/geography regression, the UCM baseline canopy coefficient is **−0.342°C per 10 percentage points**, versus **−0.119°C** in Crowd-Grid. These are baseline spatial associations in different temperature settings, not paired intervention responses. **Do not divide them to obtain a correction factor.** The diagnostic exports Green30/Target30 cooling by MSOA for inspection, but no observed counterfactual exists to validate those changes directly.

## Implications and feasible next steps

1. **Keep the 50% and 100% transfer cases labelled assumptions.** Neither is validated; neither is a formal confidence limit. Do not update health totals or choose a preferred factor from these regression slopes.
2. **Resolve exposure compatibility first:** acquire official daily extrema or a source-compatible gridded daily series, check observation-day definitions, compare central-grid and population-representative exposure, and settle the original response's lag implementation. This can improve threshold handling without claiming validated planting effects.
3. **For stronger canopy validation:** use independent station-level temperatures across canopy and built-form gradients, with building height/footprint and elevation where available. A single screened urban-background station cannot identify the spatial canopy effect. Alternatively, seek the published London study's reusable model/predictions and verify transferability to our canopy scenarios; no author contact or new acquisition has been undertaken here.
4. **Deadline fallback:** report health estimates only as explicit conditional scenarios/sensitivities, or defer absolute seasonal mortality claims until the cooling input is adequately supported. Relabelling does not repair a metric mismatch. A future-mortality claim additionally requires a compatible future daily series and assumptions about susceptibility; this historical diagnostic does not provide them.

No age-specific expansion, additional health calculations, UCM reruns or production replacement were performed.

## Reproduce and inspect

Script: [`validate_daily_mean_canopy.py`](../../code/preprocessing_layers/validate_daily_mean_canopy.py), using the InVEST 3.20.2 Python environment. Arguments: `DATA_ROOT NEW_OUTPUT_DIRECTORY`; an existing output directory is refused.

New outputs under the external data root:

`2_postprocess_intermediate/UCM_official_runs/diagnostics/daily_mean_canopy_validation_2026-11-12/`

- `manifest.json`: input/script hashes, sample, exclusions, diagnostics and limitations.
- `station_grid_comparison.csv`: daily matched temperature series.
- `canopy_associations.csv`: all specifications and annual sensitivities.
- `spatial_holdout.csv`: five-strip held-out prediction summaries.
- `msoa_comparison.csv` and `msoa_input_coverage.csv`: spatial predictors, temperatures, scenario cooling and coverage.
- `inconsistent_extrema_pairs.csv`: source date/MSOA pairs flagged by QC.

Checks passed: unique dates/MSOA names; complete source series; exact raster alignment; nonempty zones; full-rank design matrices; and the midpoint regression coefficient equals the average of maximum/minimum coefficients to numerical precision. No figures were generated. Existing sources and outputs were read only. Remaining limitations include product dependence, spatial aggregation, static 2023/2024 canopy versus earlier weather, omitted elevation/building morphology, and lack of observed planting counterfactuals.
