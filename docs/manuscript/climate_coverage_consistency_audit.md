# Climate coverage and health-consistency audit

12 November 2026. **Coverage fix verified in a separate diagnostic folder. Existing climate, health and figure outputs unchanged; no full health rerun or push.**

## Decision

**Current implementation status:** [all nine corrected scenario temperatures are now available and verified](corrected_temperature_family.md), including compatibility with the recomputed full-coverage offsets and signed warming calculation. The full seasonal health rerun remains pending approval. Earlier checks and requirements below are retained as the audit trail.

The climate-induced coverage gap is resolved for the tested corrected baseline/Green30/Target30 UCM–population footprint. The exposure timing, candidate response coefficients and seasonal-allocation arithmetic pass the focused checks. The WorldClim-aligned approach can proceed to an **approved, versioned implementation**, subject to the explicit modelling assumptions below—not automatic replacement of manuscript estimates.

Two implementation requirements remain: use the buffered inputs and recompute their full-footprint temperature offsets; extend the cooling-only pilot to handle signed temperature changes before running all nine scenarios. Neither requires a new epidemiological study.

## 1. Coverage: cause, correction and measured effect

The old native-resolution WorldClim files were masked tightly to London. Reprojecting those already-masked rasters to the 10 m UCM grid left boundary cells without climate values and altered interpolation close to the mask. The fix is to retain surrounding native climate cells, interpolate first, and apply the London/UCM/population mask afterwards. No nearest-neighbour gap filling, extrapolation or population renormalisation was used.

Downloaded six small, native-grid rectangular subsets from the official WorldClim 30-arcsecond files: ACCESS-CM2, IPSL-CM6A-LR and MRI-ESM2-0; SSP370 and SSP585; 2041–2060. Each retains all 12 bands; the current reconstruction uses May–August. The buffered extent is longitude −0.65 to 0.40, latitude 51.20 to 51.75. Downloads and hashes are recorded in the diagnostic manifest.

| Coverage measure | Earlier climate intersection | Buffered candidate |
|---|---:|---:|
| Valid 10 m cells | 14,020,635 | 14,258,588 |
| 2021 population on those cells | 8,731,620 | 8,834,005 |
| Remaining climate-missing cells within tested UCM/population footprint | 237,953 | **0** |

Recovered **102,385 residents**, approximately 1.16% of the pre-climate population footprint. “Complete” here means complete relative to the valid corrected UCM/population mask, not proof that every geographical London cell or census resident is represented. Coverage must be checked again across all nine scenario rasters.

**Dataset identity check:** all overlapping valid native cells match exactly across all 12 bands for every file (maximum difference 0°C). This is a coverage/interpolation correction, not a GCM, period or climate-product substitution.

Across the 24 model/SSP/month comparisons, full-footprint area-mean Tmax is **0.00195–0.00292°C lower** than the earlier clipped-footprint mean. Near the clipped edges, individual interpolated cells differ by up to **0.200°C** because surrounding native cells are now available. These interpolated climate fields supply a London monthly mean; they are not added as a second pixel-scale spatial pattern to UCM.

The small change in monthly mean does **not** establish a negligible change in health totals: the recovered cells also restore their population and mortality, and threshold responses are nonlinear. Existing death estimates have not been recalculated. Recompute both WorldClim means and the UCM centring mean on the expanded common footprint before the approved health run; do not reuse the old offsets unchanged.

## 2. Focused exposure check

The [official WorldClim documentation](https://www.worldclim.org/data/cmip6/cmip6_clim30s.html) identifies monthly average maximum temperature in °C. This supports the monthly anchor, not a direct daily exposure series. Historical within-month daily anomalies remain the approximation used to construct daily Tmax.

The [response study, Methods and Tables 1–2](https://www.ag-myresearch.com/uploads/1/3/8/6/13864925/gasparrini_oem2012.pdf) confirms lag-0–1 Tmax, London's 24.7°C threshold and pooled all-age slopes. The candidate table matches its five central estimates and intervals. June–August is a subset of its June–September study season, not an annual mortality estimate.

| Check | Finding |
|---|---|
| Temperature metric | Daily Tmax proxy from hourly station maxima, not daily mean or a single monthly extreme. Monthly WorldClim means anchor reconstructed daily values. |
| Lag order | Reconstruct current/previous temperatures, average, then threshold; do not threshold each day first. Independent reconstruction matches all 5,520 saved baseline day/branch combinations to 5.33 × 10⁻¹⁵°C. |
| First summer day | May data retained, so June 1 includes May 31. All included patterns have 92 finite summer values. |
| Weather QC | Original hourly completeness rule reproduced; five included years 2018–2022, 2023 excluded. Separate monthly/year centring retains daily variability but removes observed interannual monthly-mean differences. |
| Threshold/coefficients | Candidate extraction remains unchanged: one London threshold, five pooled England/Wales all-age coefficients. These are not newly fitted local or disease-specific thresholds. |
| Formula edge cases | Baseline-cold, threshold-crossing, both-hot and zero-cooling cases agree with direct risk-ratio arithmetic to 2.11 × 10⁻¹⁵ deaths. |

Canonical coefficient values and ICD definitions remain in [the response extraction table](manuscript_health_response_candidates.csv); detailed rationale remains in [the literature audit](cause_specific_response_literature.md). Do not copy the old production coefficients from the example health configuration: the pilots deliberately read the candidate table instead.

**Interpretation, not a bug:** for baseline expected count D, the intervention calculation is `D × (1 − RR_scenario/RR_baseline)`. It is valid conditional on D representing mortality under the chosen baseline. Holding allocated 2021 D fixed under future climate is a modelling convention, not a projection of warming-related growth in baseline deaths. It must not be described as reproducing the source paper's attributable-burden calculation verbatim or as an unconditional 2050 forecast.

## 3. Focused mortality check

The local ONS workbook was reopened read-only: its cover and sheet 1 identify **England, month of occurrence, all persons**, 2014–2023; the extraction uses 2021 and sums disjoint age bands. An independent chapter/code-based aggregation reproduced every saved monthly count for all five outcomes exactly (60 cause/month comparisons). No male/female sheets were added to the all-persons sheet.

All 12 fractions per cause are non-negative and sum to one; saved fractions equal monthly count divided by annual count. Repeating each monthly daily allocation over its calendar days reproduces the summer fraction exactly. All five mortality rasters are aligned, finite, non-negative and non-NoData on the expanded footprint.

| Outcome | June–August fraction | Annual allocated deaths, full tested footprint | Estimated summer baseline deaths |
|---|---:|---:|---:|
| All-cause | 22.00% | 56,944.89 | 12,527.07 |
| Cardiovascular | 23.55% | 12,936.98 | 3,046.07 |
| Respiratory | 22.77% | 4,809.99 | 1,095.38 |
| Mental/behavioural | 22.68% | 3,543.99 | 803.64 |
| Intentional self-harm | 26.17% | 438.00 | 114.64 |

These are input baseline counts, **not deaths averted**. All-cause already includes the selected causes; rows must not be summed. No division by 12 or additional summer multiplier is applied after the daily seasonal allocation.

**Unresolved approximation:** England occurrence-month proportions applied to London borough annual registrations are not observed London seasonal deaths. Counts are uniform within each month. The focused check verifies extraction and arithmetic, not equivalence of registration and occurrence dates or England and London seasonality. Retain the proxy transparently rather than delaying the manuscript for a custom data request. Registration delays are particularly relevant to self-harm; treat cause-specific totals as supporting conditional estimates.

## 4. What remains before a full rerun

**Implementation follow-up:** [the nine-scenario readiness check](nine_scenario_health_readiness.md) now verifies full coverage, recomputes shared offsets and tests signed warming effects. It identifies seven scenario temperature runs still requiring the established NoData correction before corrected all-scenario health adoption. The checklist below records the original implementation requirements; consult that follow-up for current status.

1. **Approve the analysis settings:** WorldClim-aligned SSP370, fixed 2021 population/mortality, full transfer and nominal 5°C UHI, with the already-tested lower-UHI/half-transfer and regional sensitivities. No calibration claim is justified by this audit.
2. **Implement a new run configuration:** point to buffered climate candidates and explicitly selected candidate responses; compute new offsets using the full common mask; keep every old run immutable.
3. **Check all nine scenario inputs:** ensure consistent corrected NoData handling and common coverage, including scenarios not included in this three-scenario pilot.
4. **Support warming as well as cooling:** `pilot_spatial_reconstructed_health.py` assumes non-negative cooling and clips tiny negative numerical results. It must not be used unchanged for AllBuilt/TreeRisk. Preserve signed health differences (negative deaths averted for warming) and validate cold/hot/crossing cases against direct calculations before all-scenario production. Do not clamp genuine harm to zero.
5. **Then regenerate spatial health outputs and check conclusions:** citywide, borough and LSOA totals must reconcile; Figures 6–7 require reassessment. This audit does not establish unchanged equity patterns or provide new health totals.

No additional temperature reconstructions, age-specific modelling or new model calibration are prerequisites for a conditional analysis. Full cooling transfer, UHI strength, synchronous station anomalies, historical variability, regional-to-pixel response transfer and fixed baseline mortality remain explicit limitations, rather than “resolved” by computational checks.

## Reproduction and files

Main audit script: [`check_health_coverage_consistency.py`](../../code/preprocessing_layers/check_health_coverage_consistency.py), using the InVEST 3.20.2 Python environment. It refuses an existing output folder, disables input sidecar writes, checks original temperature-input hashes before and after, downloads native buffered subsets, verifies coverage, reconstructs lags and checks seasonal/formula arithmetic. It generates **no figures and no health estimates**.

```text
python code/preprocessing_layers/check_health_coverage_consistency.py DATA_ROOT NEW_OUTPUT_FOLDER
```

New folder relative to the project data root:

```text
2_postprocess_intermediate/UCM_official_runs/diagnostics/climate_coverage_consistency_2026-11-12_v1/
```

Contents: `buffered_climate/`, `climate_coverage_comparison.csv`, `mortality_coverage_checks.csv`, `weather_qc.csv`, and `manifest.json`. Native-overlap identity and independent workbook extraction were additional read-only checks documented above. Workbook: `diagnostics/seasonal_health_2026-11-12/ons_england_monthly_cause_2014_2023.xlsx`; existing extraction workflow: [`pilot_five_outcome_thresholds.py`](../../code/health_assessment/pilot_five_outcome_thresholds.py), with saved comparison fractions in `diagnostics/five_outcome_thresholds_2026-11-12/monthly_cause_fractions.csv`.

Earlier numerical comparisons remain authoritative in [the reconstruction comparison](spatial_reconstructed_health_comparison.md) and [UHI/spatial-scale sensitivity](uhi_threshold_scale_health_sensitivity.md), with their original restricted coverage. This audit does not silently revise those numbers.
