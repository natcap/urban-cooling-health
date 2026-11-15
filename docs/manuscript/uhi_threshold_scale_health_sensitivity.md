# Lower-UHI and spatial threshold sensitivity

12 November 2026 · Diagnostic only; existing outputs, manuscript and figures unchanged. Not pushed.

## Decision summary

**Target30 retains greater citywide health benefits than Green30 for all five outcomes in every one of 8,100 matched comparisons. Absolute benefits remain sensitive to UHI.** Regional rather than pixel-level response application changes ensemble means much less, but does not validate transferring a regional epidemiological relationship to neighbourhoods.

Keep WorldClim-aligned reconstruction as the preferred candidate, with 2/3/5°C UHI sensitivity reported explicitly. These tests do not identify a calibrated best UHI, validate full cooling transfer, or establish that Figures 6–7's equity conclusions remain unchanged. Do not replace production health outputs yet.

## Results

For the earlier 2021 station-based full-Tmax pilot alongside both newer reconstruction families, see the [three-approach comparison table](spatial_reconstructed_health_comparison.md#results-full-cooling-transfer). It is retained there as the canonical comparison; the lower-UHI tests below were not run for that earlier approach.

Entries are **Green30 / Target30 expected deaths averted per modelled June–August summer**: WorldClim-aligned SSP370, central response coefficients, full cooling transfer, equally averaged over three GCMs and five weather patterns. Baseline avoided deaths are zero by definition. All-cause and cause-specific rows must not be added together.

### Pixel-specific temperatures and response

| Outcome | UHI 2°C | UHI 3°C | UHI 5°C |
|---|---:|---:|---:|
| All-cause | 15.68 / 20.23 | 23.31 / 30.25 | 38.01 / 50.01 |
| Cardiovascular | 3.25 / 4.20 | 4.83 / 6.27 | 7.88 / 10.37 |
| Respiratory | 2.61 / 3.37 | 3.88 / 5.03 | 6.32 / 8.30 |
| Mental/behavioural | 1.43 / 1.82 | 2.12 / 2.72 | 3.46 / 4.49 |
| Intentional self-harm | 0.190 / 0.245 | 0.283 / 0.366 | 0.461 / 0.606 |

For all-cause mortality, lowering UHI from 5°C to 3°C reduces benefits by approximately 39–40%; using 2°C reduces them by approximately 59–60%. The other outcomes behave similarly. This reflects reduced greening cooling and narrower baseline spatial temperature contrasts, not a change in response coefficient.

### Regional temperatures and response

| Outcome | UHI 2°C | UHI 3°C | UHI 5°C |
|---|---:|---:|---:|
| All-cause | 16.43 / 21.07 | 24.54 / 31.43 | 40.64 / 51.75 |
| Cardiovascular | 3.39 / 4.35 | 5.07 / 6.49 | 8.40 / 10.70 |
| Respiratory | 2.77 / 3.56 | 4.14 / 5.30 | 6.85 / 8.71 |
| Mental/behavioural | 1.56 / 2.00 | 2.33 / 2.98 | 3.85 / 4.90 |
| Intentional self-harm | 0.192 / 0.246 | 0.287 / 0.367 | 0.475 / 0.605 |

Across both SSPs, all three UHI values, both greening scenarios and five outcomes, regional ensemble means differ from pixel estimates by **−0.24% to +11.39%** in the WorldClim-aligned branch. Most differences are positive; the largest proportional differences concern mental/behavioural mortality. These are differences between ensemble means, not bounds on individual runs.

The hybrid diagnostic—common regional baseline but pixel-specific cooling and mortality—differs from pixel estimates by **−3.53% to +2.67%**. At UHI 5°C under SSP370, its all-cause estimates are **38.95 / 48.43**, compared with pixel **38.01 / 50.01** and fully regional **40.64 / 51.75**. Thus the fully regional comparison is not a test of threshold placement alone: exposure averaging and mortality–cooling covariance also matter.

The rural-reference-anchored branch also preserves the Target30 advantage throughout; regional ensemble estimates are approximately **0.34–10.44%** above pixel estimates. Complete results for both reconstructions, SSP370/585, individual GCM/weather patterns and response-coefficient endpoints are saved in the diagnostic CSVs.

## What was tested—and held fixed

The [previous pixel-health comparison](spatial_reconstructed_health_comparison.md) is authoritative for mortality inputs, coefficients, formula and seasonal assumptions; the [temperature reconstruction report](temperature_reconstruction_comparison.md) documents climate inputs, daily anomalies and common coverage. This test changes only UHI and spatial response application.

**UHI:** the corrected baseline, Green30 and Target30 5°C temperature rasters were transformed analytically, with fixed heat mitigation and normalized spatial mixing:

```text
T_u = 25 + (u/5) × (T_5 − 25)
cooling_u = (u/5) × cooling_5
```

This follows the installed InVEST 3.20.2 temperature equation and linear normalized mixing. It is **not a fresh InVEST run** or an empirical UHI calibration. Scaling corrected rasters avoids confounding UHI with the earlier NoData discrepancy. Values 2 and 3°C are sensitivity settings, not established summer averages.

WorldClim-aligned monthly area means remain fixed as UHI changes; only baseline spatial contrasts and intervention cooling change. In the anchored sensitivity, the rural reference remains 25°C, so lower UHI also lowers the monthly baseline. The inherited CSV label `ucm_25_5_anchored` identifies that reconstruction family; at UHI 2 or 3°C it means **25/2 or 25/3**, not an unchanged 25/5 configuration.

**Spatial scale:** the same 24.7°C threshold and lag-0–1 Tmax response are used in three modes:

1. `pixel`: apply the threshold and response to each pixel's temperatures, then sum cause-specific pixel effects.
2. `regional_baseline_pixel_cooling`: use a common population-weighted London baseline, retaining pixel cooling and mortality. This isolates baseline-exposure homogenisation more closely, although it is not a pure threshold-parameter test.
3. `regional`: population-weight baseline and scenario temperatures first, then apply the response to total cause-specific deaths. This is a regional-scale sensitivity, **not an exact reconstruction of the source study's historical station network or exposure weights**.

Thresholding and averaging do not commute: some pixels exceed the threshold when the regional mean does not, and vice versa. Population-weighted average cooling also need not equal cause-specific mortality-weighted cooling. These explain why the regional calculation need not match the pixel sum.

**Held fixed:** full cooling transfer, five all-age response coefficients and their endpoints, fixed 2021 population and mortality, England cause-specific monthly fractions, five weather patterns, three GCMs, and the shared mask (14,020,635 cells; 8,731,620 residents). Full transfer is held fixed to separate this test from the previous half-transfer sensitivity. Lower UHI is not equivalent to half transfer, which changes scenario cooling but not baseline spatial contrasts.

## Validation and interpretation limits

- 24,300 unique result rows; baseline effects exactly zero; all values finite and non-negative; coefficient endpoints correctly ordered.
- Target30 exceeds Green30 in all 8,100 matched pairs, including both reconstructions, both SSPs, each GCM/weather pattern, all UHI values, three spatial modes, five outcomes and three coefficient values. These are deterministic sensitivity combinations, not independent statistical replications.
- The 5°C pixel branch reproduces the previous full-transfer results to a maximum absolute difference of **1.42 × 10⁻¹⁴ deaths**.
- Direct arithmetic on 1,500 sampled real pixels agrees with the accelerated pixel/hybrid calculations to **1.33 × 10⁻¹⁴ deaths**. Input hashes still match the saved manifest.
- Neither spatial mode estimates age-specific heat susceptibility. Cause-specific slopes remain pooled England/Wales estimates; the common London threshold and regional-to-pixel transfer are assumptions.
- Seasonal baseline deaths remain fixed counterfactual counts, not projected future total mortality. Common climate-edge exclusions, seasonal-fraction approximation and daily Tmax/cooling-transfer uncertainty remain unresolved limitations.
- Stable citywide ranking does not establish stable borough/LSOA rankings, vulnerability curves or Figure 7 transitions. No neighbourhood maps were generated here.

## Reproducibility and next decision

See the subsequent [coverage and consistency audit](climate_coverage_consistency_audit.md) for the verified buffered-input fix and the remaining implementation requirements before an approved full rerun. Numbers here remain unchanged.

Script: [`pilot_uhi_threshold_scale.py`](../../code/health_assessment/pilot_uhi_threshold_scale.py), reusing the integration and grid helpers from [`pilot_spatial_reconstructed_health.py`](../../code/health_assessment/pilot_spatial_reconstructed_health.py). No figures were generated.

New output folder, relative to the configured project data root:

```text
2_postprocess_intermediate/UCM_official_runs/diagnostics/uhi_threshold_scale_health_2026-11-12_v1/
```

`ensemble_summary.csv` holds means and GCM/weather ranges; `health_sensitivity.csv` contains every run; `ranking_checks.csv` stores matched differences; `uhi_temperature_scaling.csv` records cooling and spatial contrasts; `direct_calculation_checks.csv` and `manifest.json` document checks and hashed sources. Ranges are not total uncertainty intervals.

**Recommended next step:** agree the headline UHI framing and full-transfer assumption before production adoption. If retaining 5°C, describe it as an assumed stronger-UHI summer configuration and retain 2/3°C as sensitivity, rather than presenting 5°C as calibrated typical conditions. Retain pixel calculations as the spatial-analysis candidate and regional calculations as sensitivity, with the regional-to-pixel limitation explicit. Finish exposure/season/coverage checks, then obtain approval to regenerate all nine scenarios and re-evaluate Figures 4–7; these citywide tests alone do not authorize or justify replacing those figures.
