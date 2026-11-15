# Corrected seasonal-health rerun: results and equity comparison

12 November 2026. **New versioned outputs only. Existing production, manuscript figures and Word documents unchanged; not pushed.**

## Main results

For consistency with the other Figure 4 outcomes, see the [energy and work-capacity audit](energy_work_seasonal_consistency_audit.md). It identifies an energy-calibration issue and proposes a separate daily-capacity pilot; existing valuation outputs have not been updated.

The full nine-scenario/five-outcome calculation is complete. Primary results use **WorldClim-aligned SSP370, 2041–2060; nominal 5°C UHI; full Tmax cooling transfer; fixed 2021 population and baseline mortality; June–August**. Daily threshold responses are calculated before averaging equally across three GCMs and five complete historical weather patterns. These are conditional intervention estimates, not forecasts of actual future mortality.

Expected **deaths averted per modelled summer**; negative values indicate additional deaths under warming. The four cause-specific outcomes are subsets of all-cause mortality and must not be added to it.

| Scenario | All-cause | Cardiovascular | Respiratory | Mental/behavioural | Intentional self-harm |
|---|---:|---:|---:|---:|---:|
| AllBuilt | −136.85 | −28.30 | −23.41 | −13.09 | −1.582 |
| TreeRisk | −11.41 | −2.36 | −1.89 | −1.00 | −0.134 |
| TreeOpp | 6.08 | 1.24 | 1.05 | 0.60 | 0.073 |
| Green10 | 13.83 | 2.87 | 2.30 | 1.25 | 0.168 |
| Green20 | 26.03 | 5.39 | 4.34 | 2.37 | 0.317 |
| Green30 | **38.37** | 7.95 | 6.38 | 3.49 | 0.467 |
| Target10 | 20.01 | 4.17 | 3.33 | 1.73 | 0.251 |
| Target20 | 35.90 | 7.43 | 5.98 | 3.20 | 0.444 |
| Target30 | **50.40** | 10.45 | 8.37 | 4.52 | 0.612 |

Baseline deaths averted are zero, not zero baseline deaths. Target exceeds Green at each matched canopy level for all five outcomes in **all 90 ensemble-mean comparisons** across the retained reconstruction/SSP/coefficient combinations. These are not 90 independent replications or a check of every individual GCM/weather-year ranking.

### What differs from the previous numbers?

| Comparison | Green30 all-cause | Target30 all-cause | Interpretation |
|---|---:|---:|---|
| Earlier annual production | 387.37 | 495.54 | Annual, previous response/formula and temperature inputs |
| Previous restricted-footprint seasonal pilot | 38.01 | 50.01 | Same preferred seasonal framework; climate-edge exclusions |
| **New full-footprint seasonal run** | **38.37** | **50.40** | Corrected temperatures and buffered climate on the shared valid footprint |

The annual-to-seasonal contrast combines changes in mortality period, thresholded exposure, response coefficients, temperature reconstruction and corrected inputs. It is **not** a controlled percentage effect of one change. The smaller difference from the preceding seasonal pilot reflects restored coverage and recomputed climate/UCM centring; older outputs were not modified.

The saved `previous_annual_vs_revised_summer.csv` compares every scenario and cause, with explicit period labels. Its annual source is each existing production `city_totals_deterministic.csv`, not the older restricted seasonal pilot.

### Retained comparisons and uncertainty

For Green30 / Target30 all-cause mortality:

- WorldClim-aligned SSP585: **42.21 / 55.31** per summer.
- 25/5°C-anchored SSP370: **69.17 / 89.46**.
- 25/5°C-anchored SSP585: **69.53 / 89.91**.
- Primary SSP370 coefficient-endpoint estimates: **29.34–47.34 / 38.55–62.16**.

The last ranges propagate only the source RR endpoints. They are **not total 95% uncertainty intervals**. In warming scenarios the higher RR gives a more negative effect; endpoint ordering is checked accordingly. This run retains three coefficient values for the primary reconstruction/SSP and central coefficients for the other three combinations. It does not regenerate individual-GCM/weather spread, lower-UHI or half-transfer results for all nine scenarios; earlier three-scenario sensitivities remain separately documented, not relabelled as new full-run uncertainty.

## Equity: Figure 7 comparison

The same 4,835 LSOAs, corrected four-dimension SVI and five vulnerability indicators are retained. High vulnerability means the top tertile of each indicator. Low benefit means the bottom Green30 tertile; Green30 thresholds are fixed for Target30 **within each analysis version**. Annual numerical cutoffs are not imposed on seasonal rates.

| High-vulnerability group | Previous annual: low benefit, Green30 → Target30 | Revised summer: low benefit, Green30 → Target30 |
|---|---:|---:|
| Overall SVI | 45.5% → 4.5% | **44.8% → 5.5%** |
| Income deprivation | 41.6% → 6.1% | **41.0% → 7.5%** |
| Age <5 | 45.8% → 22.2% | **45.7% → 23.3%** |
| Age 75+ | 28.7% → 59.0% | **29.6% → 61.1%** |
| Ethnic-minority share | 51.4% → 3.5% | **49.2% → 4.1%** |

**The main distributional contrast persists.** Target30 reduces low-benefit prevalence among high-SVI, income-deprived, younger and ethnic-minority-share neighbourhoods, but increases it among neighbourhoods with high older-population shares. This is a modelled allocation pattern, not evidence of an age-specific physiological response.

Among 1,612 high-SVI LSOAs, **706 leave** the low-benefit class, **16 remain**, and **72 enter** under Target30. Their total estimated benefit increases by **12.25 deaths per summer (100.3%)**. Among 1,608 high-age-75+ LSOAs, **242 leave**, **234 remain** and **748 enter**; their total benefit decreases by **2.08 deaths per summer (17.1%)** relative to Green30. Both the transition counts and the absolute-benefit comparison are needed: a low-benefit class alone does not establish losses.

New Green30 rate cutoffs are **3.5062 and 5.0539 deaths averted per million residents per summer**. The Figure 7 comparison concerns relative distribution under each version's internally fixed thresholds; it does not assert that annual and seasonal low-benefit categories represent the same absolute health level.

## Figure 6 curve comparison

Matched models use all 4,835 LSOAs, the same standardized indicators and robust scaled-t GAM specification (REML, shrinkage smooths, k = 10). Predictions include the intercept; P1–P99 limits affect display only. Existing manuscript curves are not overwritten.

The revised descriptive P90–P10 contrasts below are differences in **deaths averted per million residents per summer**, not causal effects of vulnerability.

| Indicator | Green10 | Green20 | Green30 | Target10 | Target20 | Target30 |
|---|---:|---:|---:|---:|---:|---:|
| Income deprivation | −0.298 | −0.477 | −0.579 | 2.963 | 4.740 | 5.440 |
| Age <5 | −0.570 | −0.893 | −1.183 | 1.190 | 2.411 | 2.568 |
| Age 75+ | 0.100 | 0.225 | 0.272 | −2.353 | −3.721 | −4.018 |
| Ethnic-minority share | −0.553 | −0.919 | −1.186 | 3.427 | 5.533 | 6.510 |
| Overall SVI | −0.312 | −0.525 | −0.653 | 3.539 | 5.203 | 5.852 |

**Contrast directions persist across all six greening scenarios and agree with the previous Green30/Target30 results.** Target30 curve shapes are very similar after removing the annual-versus-seasonal magnitude difference. Green30 curves are flatter. In particular, its age-75+ curve rises initially but declines at the high end: a positive P90–P10 contrast does not establish a monotonic increase. Income-deprivation curves are also not strictly monotonic.

The focused **spatial k = 100** check changes one important qualification. Green30's income-deprivation contrast is now **−0.014 (nominal model-conditional 95% interval −0.047 to 0.019)**, compared with an entirely negative interval previously. The under-5 contrast also includes zero; the age-75+ contrast is weakly positive, **0.072 (0.002–0.142)**. Green30 ethnic-minority and SVI contrasts remain negative. Target30 retains positive income, under-5, ethnic-minority and SVI contrasts and a negative age-75+ contrast after spatial adjustment. These are associations conditional on the fitted geographic smooth, not estimates of targeting effects. This run did not repeat k = 50 and 200; it must not be described as a complete new spatial-sensitivity suite.

The same Target30-versus-Green30 direction holds for **all five outcomes** in high-vulnerability groups: total benefits increase for high-SVI, income-deprived, under-5 and ethnic-minority-share groups, but decrease for the high-age-75+ group. Cause-specific GAMs were not fitted; the five-outcome comparison is based on group totals and rates.

Diagnostic PNGs are in the new run's `comparison_figures/` folder:

- `figure6_shape_comparison_v1.png`: old and new full fitted curves divided by each version's mean LSOA rate, to compare shape without confusing annual and seasonal magnitudes. This diagnostic has no uncertainty bands and is not a replacement manuscript figure.
- `figure7_classification_comparison_v1.png`: old and new high-vulnerability low-benefit percentages under internally fixed Green30 thresholds.

## Recommended next decisions

### New manuscript-layout candidates (12 November 2026)

- [Figure 6 seasonal v2](../../figures/figure6/manuscript/equity_health_All_cause_gam_Green30_20261112_10_per_million_population_full_prediction_seasonal_v2.png): same five-panel Green30 layout, full predictions and conditional bands, densities, outward arrows and vulnerability labels. Units now deaths averted per million residents per modelled summer.
- [Figure 7 seasonal v5](../../figures/figure7/png/fig7_main_mapled_SVI_20261112_seasonal_v5.png): same four-panel map-led layout, palette, bottom-to-top benefit classes, directional connectors and percentage labels. Seasonal Green30 cutoffs are fixed for Target30. Previous annual figures remain unchanged.

Generator: [`plot_seasonal_manuscript_figures.R`](../../code/health_assessment/plot_seasonal_manuscript_figures.R), using the existing Figure 6 plotting chunk in `equity-health.Rmd` and Figure 7 layout in `equity-health-fig7-production.Rmd`. It reads the completed seasonal run and saved GAM predictions; it does not refit models or import old annual Monte Carlo intervals. Run with the versioned seasonal run folder as its argument. Existing figures are hash-checked; `--refresh` replaces only these two new candidates. Versions are identified in filenames, leaving the manuscript layout uncluttered.

**Proposed Figure 6 legend.** Non-linear associations between neighbourhood vulnerability and modelled seasonal health benefits under Green30. Curves show full fitted predictions, including the intercept, from robust scaled-t GAMs fitted by REML across 4,835 London LSOAs. All-cause deaths averted per million residents per modelled summer compare Green30 with baseline land use under WorldClim-aligned mid-century SSP370 conditions, with fixed 2021 population and mortality. Predictors are standardized z-scores; dotted lines mark citywide means. Shading indicates pointwise 95% model-conditional intervals; grey densities show predictor distributions on an independent scale. All LSOAs enter the fits; curves display P1–P99. These descriptive associations are not adjusted for location, and intervals exclude spatial dependence and upstream uncertainty.

**Proposed Figure 7 legend.** Distribution of modelled seasonal health benefits under general and targeted greening. **a,b,** Bivariate maps of four-dimension SVI and all-cause deaths averted per million residents per modelled summer under Green30 and Target30. Vulnerability tertiles are identical across scenarios; Green30 benefit tertiles (3.5062 and 5.0539) define fixed thresholds for both. **c,** Low, moderate and high benefit classes, from bottom to top, among the same 1,612 high-SVI LSOAs. **d,** Low-benefit shares within each indicator's highest-vulnerability tertile; arrows connect Green30 to Target30. Both scenarios are compared with baseline land use under the same WorldClim-aligned mid-century SSP370 conditions, with 2021 population and mortality fixed. Classes describe relative benefit, not absence of benefit.

The seasonal Results and spatial qualifications above accompany these candidates. Existing annual copy-ready manuscript text remains a historical version until the seasonal candidates are selected; do not combine its annual numbers with these figures.

### Seasonal supplementary figures and legends

All three PNGs preserve the previous style and figure dimensions, use the same seasonal run as the new main figures, and leave earlier files unchanged:

- **Supplementary Fig. S6-1:** [Green10/20/30 overlay, seasonal v2](../../figures/figure6/scenario_comparison/figure6_supplement_green_overlay_seasonal_20261112_v2.png).
- **Extended Data Fig. 3:** [Target10/20/30 overlay, seasonal v2](../../figures/figure6/scenario_comparison/figure6_supplement_target_overlay_seasonal_20261112_v2.png). Retains the approved numbering; do not reinstate the retired S6-2 label.
- **Supplementary Fig. S7:** [Five-indicator Green30/Target30 maps, seasonal v2](../../figures/figure7/png/fig7_ab_supplement_all_five_seasonal_20261112_v2.png). Map-only supplement; no additional transition panels.

Generator: [`plot_seasonal_supplement_figures.R`](../../code/health_assessment/plot_seasonal_supplement_figures.R), called with the seasonal run folder. It reuses the overlay renderer in `code/compare_figure6_scenarios.R`, the seasonal input adapter in `code/health_assessment/plot_seasonal_manuscript_figures.R` and the map renderer in `code/equity-health-fig7-production.Rmd`. Thirty saved seasonal GAM curves are used without refitting. The existing SVI source checksum, five-indicator map percentages and preservation of earlier figures are checked. `--refresh` replaces only these three new PNGs. No old annual uncertainty draws are used.

**Supplementary Fig. S6-1 | Neighbourhood vulnerability and modelled seasonal health benefits under general greening.** Dotted, dashed and solid green curves represent Green10, Green20 and Green30. Panels show population shares aged <5 and ≥75 and ethnic-minority share (top row), and income deprivation and four-dimension SVI (bottom row). Curves are full fitted predictions, including the intercept, from separate robust scaled-t GAMs fitted by REML across 4,835 London LSOAs. Benefits are all-cause deaths averted per million residents per modelled June–August summer relative to baseline land use under WorldClim-aligned SSP370, 2041–2060 conditions, with 2021 population and mortality fixed and borough deaths allocated by population. Shading shows pointwise 95% model-conditional intervals. Predictors are standardized z-scores; dotted vertical lines mark citywide means, arrows indicate vulnerability direction, and grey densities show predictor distributions on an independent scale. All LSOAs enter the fits; curves display P1–P99. Y-axis ranges differ between indicators but are shared across scenarios within each panel. Associations are unadjusted for location; intervals exclude spatial dependence and upstream uncertainty.

**Extended Data Fig. 3 | Neighbourhood vulnerability and modelled seasonal health benefits under targeted greening.** Dotted, dashed and solid purple curves represent Target10, Target20 and Target30. Panel order, population and mortality inputs, seasonal outcome, climate reconstruction and GAM specification match Supplementary Fig. S6-1. Curves show full fitted all-cause deaths averted per million residents per modelled summer; shading indicates pointwise 95% model-conditional intervals. Grey densities show predictor distributions independently of mortality units, and arrows indicate vulnerability direction. All 4,835 LSOAs enter the fits; curves display P1–P99. Y-axis ranges differ between indicators but are shared by the three scenarios within each panel. Associations are unadjusted for location; intervals exclude spatial dependence and upstream uncertainty.

**Supplementary Fig. S7 | Spatial distribution of modelled seasonal health benefits across vulnerability indicators.** **a,b,** Bivariate maps under Green30 (a) and Target30 (b) for overall SVI, income deprivation, population aged ≥75, ethnic-minority share and population aged <5. Benefits are all-cause deaths averted per million residents per modelled summer under the same climate reconstruction and fixed 2021 population and mortality as Fig. 7. Metric-specific vulnerability tertiles and seasonal Green30 benefit thresholds (3.5062 and 5.0539) are held fixed across both scenarios for all 4,835 LSOAs. Blue indicates high vulnerability and relatively low benefit. Age indicators describe neighbourhood composition, not age-specific mortality; SVI is related to targeting design, not an independent validation measure.

1. Review the seasonal estimates and diagnostic equity figures before adopting them. The principal Green–Target conclusions persist, but Green30's age-related wording should avoid monotonic or physiological claims, and its spatially adjusted deprivation association should not be described as clearly negative.
2. If adopted, update health numbers and units consistently across the Abstract, Results, Table 2, Figures 4–7 and health supplements. Use **per modelled summer**, and **per million residents per modelled summer** for rates. Do not present the annual-to-seasonal difference as a single-method sensitivity.
3. Retain the assumptions and uncertainty qualifications above. Energy and work-capacity outputs were not rerun here; check their temperature-input and reporting-period consistency before replacing combined co-benefit figures. The existing manuscript, figures and production outputs remain unchanged pending review.

## Population accounting and reproducibility checks

- Shared health footprint: **14,258,588 cells, 8,834,005 residents**. Exactly 4,835 LSOAs and 33 boroughs are represented.
- Health cells outside the generalized LSOA outline follow the existing nearest-polygon rule, with ties assigned to the smallest stable id: **6,823 cells**, maximum distance **21.43 m**. No cells are dropped from the zonal sum.
- The saved manuscript LSOA population lookup totals **8,832,325**, because its pixel-centre allocation does not use the same edge assignment. For matched equity comparisons that denominator is retained, not silently changed. The alternative health-mask population is exported per LSOA; the maximum local discrepancy is **5.38%**, although the citywide difference is only 0.019%. Recomputing Figure 7 classes with the matched-mask denominator gives **identical high-vulnerability low-benefit counts for all five indicators**. Population remains 2021 in both cases.
- All **300** citywide rows reconcile with **1,450,500** LSOA rows and **9,900** borough rows; maximum discrepancies are below **10⁻¹² deaths**.
- Direct daily arithmetic agrees with the exact time-integration algorithm to **1.43 × 10⁻¹⁷ deaths** on the sampled checks. An additional mixed-sign synthetic test passes at 10⁻¹² tolerance.
- All 45 primary rasters are finite on the shared mask; their Float32 sums agree with Float64 totals within 10⁻⁵ deaths. Baseline is exactly zero; warming harms are retained. Recorded source hashes remain unchanged.

The [coverage/consistency audit](climate_coverage_consistency_audit.md) and [corrected-temperature report](corrected_temperature_family.md) remain authoritative for upstream changes. Full Tmax transfer, assumed UHI strength, regional-to-pixel response transfer, England seasonal occurrence fractions applied to London annual registrations and fixed baseline mortality remain limitations. Numerical verification does not remove them.

## Files and scripts

New output folder relative to the configured data root:

```text
2_postprocess_intermediate/UCM_official_runs/diagnostics/seasonal_health_all_scenarios_2026-11-12_v1/
```

- `citywide_results.csv`, `borough_results.csv`, `lsoa_results.csv`: all computed reconstruction/SSP/coefficient combinations.
- `rasters/`: 45 primary central-coefficient seasonal deaths-averted rasters, named by scenario and outcome. Units are deaths per pixel per modelled summer.
- `population_coverage.csv`, `population_denominator_sensitivity.csv`: denominator accounting and classification sensitivity.
- `previous_annual_vs_revised_summer.csv`, `green_target_ranking.csv`: comparisons with explicit estimands.
- `numerical_checks.csv`, `final_accounting_checks.json`, `manifest.json`: calculation, conservation, input hashes and run settings.
- `equity_comparison/`: matched GAMs, Figure 7 thresholds/transitions and all-five-outcome high-vulnerability summaries.
- `comparison_figures/`: new diagnostic PNGs only, not replacement manuscript figures.

Generating scripts:

1. [`run_seasonal_health_corrected.py`](../../code/health_assessment/run_seasonal_health_corrected.py): health rasters and summaries. Exact time integration evaluates daily risks before ensemble averaging; no temperature binning or monthly-mean risk shortcut.
2. [`audit_seasonal_health_run.py`](../../code/health_assessment/audit_seasonal_health_run.py): preservation, completeness, raster totals and previous-result comparison.
3. [`compare_seasonal_health_equity.R`](../../code/health_assessment/compare_seasonal_health_equity.R): matched descriptive GAMs, focused spatial check and vulnerability summaries.
4. [`check_seasonal_population_denominator.py`](../../code/health_assessment/check_seasonal_population_denominator.py): denominator sensitivity.
5. [`plot_seasonal_health_comparison.py`](../../code/health_assessment/plot_seasonal_health_comparison.py): diagnostic PNGs from saved comparisons.

Python scripts use the existing InVEST environment; the equity script uses the existing R installation with `sf`, `mgcv` and `dplyr`. New GIS processing uses GDAL/OGR and Shapely; no packages were installed. All output destinations are separate from production.
