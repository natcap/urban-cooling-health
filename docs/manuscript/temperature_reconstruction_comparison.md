# Temperature reconstruction comparison: assumed 2050 climate

12 November 2026 · Temperature-only diagnostic · **No health estimates updated; no push.**

## Decision summary

**WorldClim-aligned remains the preferred candidate for the next health pilot**, because its absolute monthly temperatures match the selected 2041–2060 GCM/SSP climate. Retain the 25/5°C-anchored version as an assumed warm-condition sensitivity, not a calibrated projection. Neither reconstruction validates the InVEST spatial Tmax pattern or its 5°C UHI setting.

Both versions retain exactly the same spatial temperature contrasts and greening cooling. The difference is their absolute temperature anchor. Under SSP370, the anchored reconstruction is **2.97°C warmer** on average across the three GCMs; under SSP585 the difference is **2.62°C**. These offsets materially change threshold exposure, so absolute temperature anchoring should be settled before replacing health results.

## Main comparison

Equal-weight averages across **three GCMs and five observed weather-pattern years (2018–2022)**. Each reconstructed summer has 92 days. The temperature mean is area-weighted over the shared valid footprint; threshold days are population-weighted using fixed 2021 population and **lag-0–1 Tmax >24.7°C**. Threshold days are average days per resident, not days when all London is above threshold, numbers of deaths, or unique people exposed.

| Climate | Reconstruction | Baseline JJA mean Tmax (°C) | Baseline days above threshold | Green30 days, full transfer | Target30 days, full transfer |
|---|---|---:|---:|---:|---:|
| SSP370 | **WorldClim-aligned** | **24.56** | **42.09** | **38.36** | **37.35** |
| SSP370 | 25/5°C-anchored | 27.53 | 73.47 | 69.84 | 68.74 |
| SSP585 | **WorldClim-aligned** | **24.93** | **46.25** | **42.29** | **41.07** |
| SSP585 | 25/5°C-anchored | 27.55 | 73.79 | 70.19 | 69.18 |

The nearly identical SSP370/585 anchored means are expected: both are forced to the same July UCM mean, retaining only GCM/SSP differences in June/August relative to July. This branch does **not** preserve the projected difference in July warming between SSPs.

Across GCMs, averaging the five weather patterns first, WorldClim-aligned baseline exposure ranges from **33.91–49.73 days** for SSP370 and **37.11–53.10 days** for SSP585. These are model-spread ranges, not confidence intervals. Using only the 2021 weather pattern gives 48.68 and 53.62 days respectively: a single summer is not representative of the five-pattern average.

### Spatial differences and greening effects

- Baseline UCM July mean on the test footprint: **28.53°C**. `25°C` is the model's rural reference parameter, not its London mean.
- WorldClim July area means: **25.56°C (SSP370)** and **25.91°C (SSP585)**. These are not the manuscript's equal-borough means; their spatial support and weighting differ.
- Baseline July spatial P5–P95 spread is **1.42°C in both reconstructions**; the spatial pattern is translated, not flattened. The anchored P5–P95 values are 27.73–29.15°C.
- Full-transfer area-mean cooling is **0.312°C for Green30** and **0.348°C for Target30**; population-weighted cooling is 0.357°C and 0.462°C. Both reconstructions retain these identical contrasts. Values use the corrected UCM rasters and the new common climate footprint, not the old production support.
- Under half transfer, WorldClim-aligned population-weighted threshold days are **40.25 / 39.76** (Green30 / Target30, SSP370), and **44.24 / 43.66** (SSP585). Half transfer changes greening cooling only; it does not halve the baseline UHI pattern or alter the climate anchor.

Target30 lowers mean temperature and threshold exposure more than Green30 in the aggregate comparisons. **This does not establish mortality benefits, outcome-specific rankings or LSOA equity conclusions**; no mortality data or response coefficients were used.

## Reconstruction definitions

Let `U0(i)` and `Us(i)` be the corrected 25/5°C baseline and scenario UCM temperature rasters; `C(i)=U0(i)-Us(i)`; `f=1` or `0.5`. Let `M(m)` be the future monthly WorldClim London area mean and `A(y,d)` the observed station Tmax minus its **own year-and-month mean**.

```text
WorldClim-aligned: T(i,d) = U0(i) − f C(i) − mean(U0) + M(month) + A(year,day)
25/5°C-anchored:   T(i,d) = U0(i) − f C(i) + M(month) − M(July) + A(year,day)
```

Their difference is exactly `mean(U0) − M(July)` for every pixel/day within a GCM/SSP. July is the anchor; WorldClim supplies May–August seasonality, with May needed for the 1 June lag. Daily reconstructed temperatures are averaged over current/previous day **before** comparing with 24.7°C. No extra warming increment is added to either future anchor.

WorldClim contributes a London monthly scalar, **not a second fine-scale spatial field**. The fine-scale pattern comes entirely from centred UCM, avoiding the addition of two spatial temperature patterns with overlapping urban signals. Centring each observed year separately removes observed interannual monthly-mean anomalies; years represent different within-month variability and sequencing, not independent future climates.

The factorised storage (source spatial rasters plus saved daily offsets) exactly represents pixel-specific daily temperatures without duplicating tens of thousands of large rasters. The threshold-exposure calculation uses sorted pixel values and cumulative population sums; direct arithmetic was checked for equivalence. No spatial bins or city-mean-before-threshold shortcut were used.

## Data and coverage

- Corrected UCM baseline/Green30: `diagnostics/nodata_correction_2026-11-12/{scenario}_nodata0/T_air_test.tif`; corrected Target30: `diagnostics/nodata_target30_2026-11-12/target30_nodata0/T_air_test.tif`. These are existing temperature inputs; InVEST was not rerun.
- WorldClim: `data/02_preprocess/GCMs/clipped/wc2.1_30s_tmax_{GCM}_{SSP}_2041-2060_London.tif`, monthly bands 5–8. GCMs: ACCESS-CM2, IPSL-CM6A-LR, MRI-ESM2-0; SSP370/585 remain separate. Bilinear reprojection onto the equal-area analysis grid enables consistent averaging; it does not create 10-m climate information.
- Weather: existing MIDAS St James's Park station 00697 annual hourly files, 2018–2023, in `OfficialWorkingInputs/climate/uhi_station_pilot_20261111_v1/hourly/00697/`. Retain version 1 and non-suspect temperature codes, deduplicate timestamps using latest metadata timestamp, and require at least 21 distinct hourly readings daily. May–August is complete for 2018–2022. **2023 excluded:** 18 low-coverage days, including 12 in May and six in July; dates are listed in `weather_qc.csv`. No imputation.
- Population: existing count-preserving 2021 BNG population raster; no future population projection or mortality raster.
- Boundary: London borough shapefile; cell-centre assignment. UCM/population-valid footprint was 14,258,588 cells and 8,834,005 people. Intersection with climate coverage leaves **14,020,635 cells and 8,731,620 people**. Climate-edge gaps exclude **1.67% of those cells and 1.16% of that population**. Both reconstructions use the same remaining footprint, without rescaling population. These are not full-boundary totals.

## Validation and interpretation limits

Passed: 300 seasonal rows, 900 monthly rows and 27,600 daily rows; complete lag history; zero monthly anomaly means; exact WorldClim monthly-mean reproduction; constant analytic difference between reconstructions; preservation of spatial P5–P95 spread; direct versus cumulative threshold calculations; independent daily-to-seasonal aggregation; full-transfer exposure no greater than half-transfer or baseline; post-run input SHA-256 verification.

**Share as a diagnostic with caveats, not validated production exposure.**

1. The 5°C UHI and UCM spatial pattern remain unvalidated as monthly mean daily Tmax. A better climate anchor cannot correct that structural limitation. See the [station/UHI assessment](ucm_climate_parameter_pilot.md).
2. Hourly station maxima approximate daily Tmax; all pixels share synchronous weather anomalies. Spatially varying weather, future variance/heatwave changes, changing UHI, acclimatisation and seasonal evapotranspiration are not modelled.
3. Monthly offsets change at month boundaries. Lagging after reconstruction handles arithmetic consistently but does not remove those steps. A smooth seasonal interpolation is a later sensitivity, not silently introduced here.
4. Applying a regional mortality threshold to pixel exposure is a spatial-transfer assumption; computational pixel detail is not independent epidemiological validation. Review this when designing the health comparison.
5. The first diagnostic uses climate-clipped inputs. Before full-coverage production use, recover buffered/unclipped WorldClim inputs or explicitly retain and disclose this common coverage restriction.

## Recommendation and next step

Retain **WorldClim-aligned as preferred**, with SSP370 primary and SSP585 sensitivity to match the manuscript climate framing. Preserve 25/5°C-anchored as a warm-condition sensitivity. The temperature test is complete. No production adoption is implied.

**Follow-up completed after authorization:** the [pixel-specific five-outcome health comparison](spatial_reconstructed_health_comparison.md) uses both branches and full/half transfer, fixed 2021 demography, and the existing seasonal mortality assumptions. Its integration now handles pixel-specific heat excess. These temperature outputs are unchanged. Resolve/disclose climate coverage and the regional-threshold-to-pixel assumption before treating the health outputs as manuscript replacements.

## Reproduction and outputs

Generator: [`test_temperature_reconstructions.py`](../../code/preprocessing_layers/test_temperature_reconstructions.py), using the existing `valuation_accounting.py` boundary/mask/hash utilities. No figures produced; results are tables and factorised daily temperature fields. Run in the InVEST 3.20.2 Python environment:

```text
python code/preprocessing_layers/test_temperature_reconstructions.py DATA_ROOT NEW_OUTPUT_DIRECTORY
```

The script refuses an existing output directory and disables GDAL input sidecar writes. Output folder relative to the external data root:

`2_postprocess_intermediate/UCM_official_runs/diagnostics/temperature_reconstructions_2050_2026-11-12_v1/`

Files: `seasonal_comparison.csv`, `paired_reconstruction_difference.csv`, `monthly_spatial_summary.csv`, `daily_temperature_exposure.csv`, `worldclim_monthly_means.csv`, `scenario_cooling.csv`, `station_daily_anomalies.csv`, `weather_qc.csv`, and `manifest.json` (formulas, checks, coverage, exact input paths and hashes). Earlier health pilots and all production results are unchanged.

Sources: [WorldClim CMIP6 documentation](https://www.worldclim.org/data/cmip6/cmip6climate.html) describes monthly, bias-corrected 20-year climate fields; it does not provide daily future sequences. [Li, Horton & Kinney (2013)](https://doi.org/10.1038/nclimate1902) provides precedent for monthly climate changes combined with observed daily variability, not validation of this project's UCM spatial reconstruction.
