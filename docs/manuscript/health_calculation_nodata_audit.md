# Health calculation and NoData audit

12 November 2026. Read-only investigation of production code, existing diagnostic rasters and published coefficient sources. Existing model outputs and Word files are unchanged. No production reruns or push. Age-specific modelling is excluded by the author's data constraints.

## Decisions needed first

**Daily-mean/canopy check completed:** see the [validation findings and literature justification](daily_mean_canopy_validation.md). The available spatial data do not validate a 50% or 100% cooling transfer: canopy associations depend on adjustment and adding canopy did not improve the tested geographic holdout predictions. Existing health estimates remain diagnostic; do not adopt them as revised manuscript totals yet.

### London all-cause response pilot (12 November): completed, not adopted

**Recommendation:** prefer a London all-cause response over an endpoint-mixed transferred coefficient for the next health-method revision, but do not replace manuscript totals until the daily-mean cooling assumption is agreed. This test preserves every production output and existing figure.

**Selected source.** Arbuthnott et al. (2020), *Years of life lost and mortality due to heat and cold in the three largest English cities*, [DOI](https://doi.org/10.1016/j.envint.2020.105966), [open published paper](https://discovery.ucl.ac.uk/id/eprint/10106443/12/Heaviside_1-s2.0-S0160412020319218-main.pdf). Methods 2.1–2.2 (pp.2–3) and visually checked Table 2 (p.5) specify Greater London **all-cause, all-age mortality**, RR **1.039 (1.035–1.043) per °C above 18.9°C**, heat lags **0–2 days**, and daily mean defined as **(daily maximum + daily minimum)/2**. The observational period is 1996–2013; exposure came from the central conurbation grid cell. The threshold was selected by model fit, not inferred from our July map. This endpoint and geography are a closer match than Yang's pooled disease-related Oceanic estimate. It does not require age-stratified modelling; it is not evidence that historical susceptibility remains unchanged in 2050.

#### Controlled comparison

All new rows below use the **same corrected baseline, Green30 and Target30 rasters**, fixed 2021 population and 2021 ONS borough monthly registrations. Baseline has **zero deaths averted by definition**, not zero mortality; the full June–August baseline denominator is 11,671 registered deaths. The old annual production row is shown only for orientation and differs in duration, formula and NoData treatment.

| Calculation | Green30 deaths averted | Target30 deaths averted | Interpretation |
|---|---:|---:|---|
| Previous annual production | 387.37 | 495.54 | Existing result, unchanged; not directly comparable to summer rows |
| Corrected summer pilot, Yang RR 1.020 on every day | 80.08 | 100.49 | Reproduced previous seasonal pilot within 0.001 case |
| Replace coefficient only with London RR 1.039 | 154.10 | 193.08 | Isolates slope change; omitting its threshold is diagnostic only |
| Yang coefficient with London threshold/lag convention | 19.89 | 23.93 | Isolates domain restriction; not Yang's published response |
| London response with threshold; full cooling transfer | **38.29** | **45.99** | Conditional daily-response pilot |
| London response with threshold; half cooling transfer | **20.51** | **24.97** | Illustrative daily-mean cooling sensitivity, not a lower confidence bound |

The stronger coefficient alone raises estimates by approximately 92%; restricting benefits to above-threshold days more than offsets this. Target30 remains higher in all tested citywide comparisons: its advantage is approximately 20% with full transfer, versus 25% in the unrestricted seasonal pilot. This does **not** verify every borough, cause-specific endpoint, SVI curve or Figure 7 transition. No equity figures were regenerated.

For full transfer, coefficient-endpoint sensitivity is 34.45–42.10 (Green30) and 41.40–50.55 (Target30); for half transfer, 18.45–22.56 and 22.47–27.47. These are **not total uncertainty intervals** and exclude exposure, threshold, lag, transfer and baseline-death uncertainty.

#### What was actually tested

- Quality-controlled MIDAS St James's Park station 00697 data supply May 30–August 31, 2021, including the two lag-history days. All 94 days have 24 retained hourly observations after version/QC filtering and latest-record deduplication. The daily proxy is `(hourly maximum + hourly minimum)/2`; hourly extrema are not exact observed daily extrema or the paper's original gridded series. UTC day boundaries are used.
- A single station baseline series is assigned across London. The pilot uses a three-day mean (lags 0–2) before thresholding; **26 of 92** summer days exceed 18.9°C. Equal lag averaging is an explicit application convention, not a reconstruction of the original fitted distributed-lag coefficients. Confirm the exact source implementation before production adoption; thresholding before averaging can differ near the threshold.
- The corrected 25/5°C UCM cooling contrast is assumed constant across all days, including the pre-June lag history, and transferred to daily mean at factors **1.0 or 0.5**. Neither factor is empirically calibrated. This combines historical 2021 weather with an assumed warm-condition cooling field: it is **neither an actual 2021 impact estimate nor a 2050 projection**. No future warming offset was invented.
- Monthly registered deaths are spread evenly across days within each month and allocated spatially using the existing population proportions. Registration dates are not occurrence dates; this is a smooth baseline-risk approximation, not an observed daily mortality series. Full borough population denominators precede masking; minimum retained population coverage is 99.9979%, without renormalizing missing cells.
- For baseline daily expected deaths `D`, exposure `E` and threshold `h`, use `q(E)=max(E-h,0)` and `avoided = D * [1 - exp(log(RR) * (q(E_scenario)-q(E_baseline)))]`. Cooling that crosses the threshold earns benefit only down to the threshold; below it this heat-only model gives zero benefit. Do not multiply by three for the lag or subtract two attributable fractions from the same baseline death count. Cold effects and mortality displacement are not estimated.

#### Reproduction, checks and next decision

Script: [`pilot_london_heat_response.py`](../../code/health_assessment/pilot_london_heat_response.py). Use the InVEST 3.20.2 Python environment with arguments `DATA_ROOT NEW_OUTPUT_DIRECTORY`. It refuses an existing output directory. It checks zero/threshold-crossing integration against direct arithmetic, non-negative cooling, daily completeness, aligned grids, all 33 boroughs, and agreement with the earlier seasonal pilot. Exact population-weighted prefix sums are used; no cooling bins or pixel sampling are introduced.

Outputs: `2_postprocess_intermediate/UCM_official_runs/diagnostics/london_heat_response_2026-11-12/` under the external data root: `citywide_comparison.csv`, `borough_comparison.csv`, `station_daily_temperature.csv`, `manifest.json`. The manifest records source and script hashes, assumptions and coverage. No model or figure rerun is needed to inspect this pilot.

**Most feasible next step:** retain this as an all-cause sensitivity while deciding the reporting target. A conditional warm-condition intervention analysis can transparently report assumed daily-mean transfer and its sensitivity; a projected future summer death count needs a defensible daily baseline temperature series and daily-mean cooling estimates. Do not simply substitute 1.039 for 1.020 in the old annual calculation or apply 18.9°C to the Tmax map. Resolve that exposure choice before updating all nine scenarios and health figures. Cause-specific endpoints remain separately unresolved; the London all-cause result does not validate them.

### Source-method and seasonal-cause follow-up (12 November)

**Updated after Zotero review:** no production calculation changed. The main PDF and supplementary DOCX are now available and Methods were checked (details below). A usable England-level monthly ICD dataset was located and inspected; an equivalent public London-borough dataset was not found in this search. This is an availability finding, not proof that the data do not exist.

#### Yang 2024: what is and is not established

| Item | Evidence status | Consequence |
|---|---|---|
| Exposure contrast | Accessible abstract distinguishes per-1°C high-temperature associations from heatwave-event associations | The selected per-degree coefficient is not a heatwave yes/no risk ratio |
| Oceanic RR 1.020 (1.014–1.026), K=10 | Verified from author's Figure 3 screenshot | Retain the extraction; do not substitute the overall pooled estimate |
| Temperature metric | Section 2.2, pp.2–3 explicitly pools varied metrics; Table S1 includes daily mean, maximum and apparent temperature | Oceanic RR is not established as a Tmax-only coefficient |
| Reference/threshold and standardization | Section 2.2 describes high-temperature episodes relative to percentile thresholds and standardizes contrasts to 1°C assuming log-linearity | No single London threshold is supplied; not justification for application on every summer day |
| Lag selection | Section 2.2, p.3 selects the maximum-risk lag when several are reported | No common fixed lag; possible upward selection relative to a prespecified lag; do not multiply risks by lag length |

Access checks: publisher full text returned HTTP 403; Scite reported no readable full text; SciSpace returned the abstract and explicitly reported that requested Methods/supplement details were unavailable. Targeted searches did not resolve them. No methods claim was inferred from another paper. Source: Yang et al. (2024), *Heat exposure impacts on urban health: A meta-analysis*, [DOI](https://doi.org/10.1016/j.scitotenv.2024.174650); [SciSpace record](https://scispace.com/papers/heat-exposure-impacts-on-urban-health-a-meta-analysis-3f52amknh1).

**Access resolved through local Zotero:** parent `N5L6KBNA`, PDF attachment `DJE6H9UE`, supplement `ZRJPWE3D`. Files are under `~/Library/CloudStorage/GoogleDrive-yingjiel@stanford.edu/My Drive/lib/MEGAsync/`, named `Yang et al. - 2024 - Heat exposure impacts on urban health A meta-analysis.pdf` and `Yang et al. - 2024 - Heat exposure impacts on-mmc1.docx`. No library files edited. The earlier access failures above are history, not a current request for the author to upload them.

Main Methods sections 2.1–2.2 (pp.2–3) and supplementary Table S1 were read. S1 lists exposure metrics, seasons and outcomes, but has no dedicated threshold/lag or climate-zone column; it does not directly identify and reconstruct the ten Oceanic effect estimates. The paper calls its pooled outcome **disease-related mortality**, and combines disease endpoints; it must not automatically be relabelled a pure all-cause risk estimate. Numerical extraction of 1.020 is correct, while endpoint transfer to the all-cause denominator remains an additional assumption needing verification. Table S5 is a common-effect/leave-one-out analysis, not a replacement for the main random-effects Oceanic coefficient. No coefficient was silently changed.

Implication: the constant-July summer pilot is a **seasonal scaling sensitivity**, not a validated expected summer burden. The source supports neither a universal threshold nor a single compatible daily lag model. A definitive daily estimate needs a documented endpoint-matched response and compatible temperature metric. Keep Yang as an explicitly qualified transferred coefficient for this diagnostic; do not claim ONS seasonal counts alone repair the exposure-response design.

#### Newly verified monthly cause source

[ONS ad hoc release 2155](https://www.ons.gov.uk/peoplepopulationandcommunity/birthsdeathsandmarriages/deaths/adhocs/2155deathsbymonthofoccurrenceagegroupcauseandsexengland2014to2023) provides **England residents, month of occurrence, 2014–2023, three-character ICD causes, sex and age groups**. The downloaded workbook has all-person/male/female sheets, but no borough dimension. Notes warn of undercount from registration delays and use May 2024 NSPL boundaries. The web release is dated 25 June 2024; its cover says April 2024. Record both rather than invent a single version date.

Read-only inspection script: `code/health_assessment/inspect_ons_monthly_causes.py`; run with the downloaded workbook path using Python with `openpyxl`. It sums all ages in the all-person sheet (not all three sex sheets), checks unique month/code records and numeric counts, and includes the separately coded `Neonate` category in overall counts only. It writes no model inputs. Source archived in the seasonal diagnostic directory as `ons_england_monthly_cause_2014_2023.xlsx`; SHA256 `7037844d5e4583f512772ef39a8e35733ce90184974e49ffe0f066f5c6b657f0`.

| England 2021 occurrence counts | Annual | June–August | Summer fraction |
|---|---:|---:|---:|
| All cause, including separate neonate category | 551,772 | 121,382 | 22.00% |
| Cardiovascular I00–I99 | 126,534 | 29,793 | 23.55% |
| Respiratory J00–J99 | 51,930 | 11,826 | 22.77% |
| Mental/behavioural F00–F99 | 38,834 | 8,806 | 22.68% |
| Intentional self-harm X60–X84 | 4,730 | 1,238 | 26.17% |

These are **not London counts**, and must not replace the observed borough all-cause summer counts. The fractions differ by cause, demonstrating why one common seasonal fraction is an assumption. Differences from London's 20.50% all-cause registration share also mix geography and occurrence/registration timing; they cannot be attributed solely to cause mix. X60–X84 is the pipeline's intentional-self-harm definition, not the broader official suicide definition that can include undetermined intent.

The [2025 ONS response](https://www.ons.gov.uk/aboutus/transparencyandgovernance/freedomofinformationfoi/deathsbymonthsexandage2013to2025) points to this source and bespoke services. An [earlier response specifically addressing month, cause and area](https://www.ons.gov.uk/aboutus/transparencyandgovernance/freedomofinformationfoi/deathsintheukfromjanuary2015toseptember2020byareaageandgender) says a custom analysis is needed, subject to disclosure, resources and cost. These establish a request route, not guaranteed 2021 borough availability or turnaround. OHID's revised excess-mortality reporting starts in 2023 and is not a drop-in source for 2021 baseline deaths ([methodology](https://www.gov.uk/government/statistics/excess-mortality-in-england--2/changes-to-ohids-reporting-of-excess-mortality-in-england)). Excess deaths must not be substituted for total baseline deaths.

#### Feasible recommended route, in order

1. **Resolve response transfer first:** Zotero review is complete; Yang pools metrics and selects maximum-risk lags, without supplying one London-ready threshold/lag. Keep the existing summer pilot conditional. For a defensible daily estimate, select an independently documented London/UK all-age, endpoint-matched response; do not fabricate missing parameters. A daily-mean response also requires compatible daily-mean cooling, not an unexplained transfer from Tmax.
2. **Keep all-cause as the first validated endpoint:** use actual borough seasonal counts, fixed 2021 population and the corrected baseline-risk formula. If an above-threshold function is required, apply the same response to baseline and intervention daily temperature histories and sum their mortality differences. Monthly deaths divided by days can approximate daily baseline risk, explicitly; handle lags once and include the required days outside seasonal boundaries. This additional daily model is not implemented here.
3. **Request the smallest useful cause table, if cause-specific results must remain confirmatory:** contact `Health.Data@ons.gov.uk` for 2021 usual-residence boroughs E09000001–E09000033; both sexes/all ages; underlying causes I00–I99, J00–J99, F00–F99, X60–X84 plus all cause; month of occurrence preferred, registration counts separately labelled if offered; disclosure flags, extract date and annual reconciliation totals. Ask whether June–August totals by borough/cause can be supplied if monthly cells are suppressed, and ask price/turnaround before commissioning. No message sent and no cost authorized.
4. **Deadline fallback:** England cause-specific summer fractions × existing annual borough cause counts can be an explicitly labelled geographic/temporal-transfer sensitivity, not observed London seasonal deaths. Preserve cause-specific borough patterns but acknowledge that the national fraction is shared across boroughs. Do not use this to claim verified neighbourhood equity robustness. The simpler primary route remains all-cause; unresolved cause-specific findings can be withheld from confirmatory conclusions rather than presented as repaired by scaling alone. Age-65+ cardiovascular/respiratory coefficient mismatch still requires separate resolution.

**Next action:** review the response-transfer caveats and the conditional fallback below. The [ONS enquiry](ons_seasonal_deaths_request.md) is drafted only; send if borough cause-specific results are essential. More climate months alone do not resolve the response or mortality-timing issues.

#### Alternative estimate completed without waiting for ONS

`code/health_assessment/pilot_seasonal_causes.py` applies England's cause-specific occurrence fractions to each borough's existing annual registered-death allocation, using corrected baseline/Green30/Target30 temperatures and the baseline-risk formula. Existing coefficients are retained **only to isolate the scaling effect**, not endorsed as endpoint/age compatible. It compares annual, national cause-specific summer, and borough all-cause summer-fraction assumptions on one common footprint. Output: `UCM_official_runs/diagnostics/seasonal_cause_sensitivity_2026-11-12/conditional_cause_comparison.csv` and `manifest.json`. The manifest embeds the national fraction values/source hash and records temperature, mortality and config hashes. No production file was changed.

Reproduction: run `inspect_ons_monthly_causes.py` on the archived ONS workbook and capture its JSON stdout; pass data root, that JSON and a **new** output directory to `pilot_seasonal_causes.py`. The inspection requires `openpyxl`; the calculation uses the existing InVEST 3.20.2 Python environment. Both scripts are diagnostic, not a revised production health pipeline.

| Exploratory summer scaling; avoided cases | Green30 | Target30 | Additional qualification |
|---|---:|---:|---|
| All-cause denominator × England all-cause fraction | 85.29 | 108.39 | Yang pooled-disease coefficient transfer unresolved |
| Mental/behavioural | 6.81 | 8.51 | Climate-subgroup versus exact endpoint compatibility unresolved |
| Cardiovascular | 35.42 | 45.00 | Retains age-65+ coefficient on all-age deaths solely for diagnostic comparison |
| Respiratory | 13.11 | 16.63 | Same age mismatch; not confirmatory |
| Intentional self-harm | 0.49 | 0.62 | Registration/occurrence transfer and response timing unresolved |

Baseline avoided cases are zero for every method. **Do not sum the all-cause row with cause-specific rows.** The observed London-borough all-cause seasonal calculation remains 80.08/100.49; it is preferable for the denominator to the national-fraction alternative, but shares the response-transfer limitations. The difference between these two all-cause estimates combines national versus local seasonality, occurrence versus registration timing and a very small annual-source discrepancy; it is not a clean geography-only effect. Target exceeds Green in these aggregate diagnostics, not proof of unchanged LSOA equity patterns.

Recommendation: use these files now to evaluate magnitude and methodological sensitivity, while prioritizing a single compatible all-cause response for the manuscript. Do not wait for ONS to assess the main issue, and do not present the scaling calculations as validated expected summer deaths. If the deadline prevents a compatible response model, narrow the health claim to an explicitly conditional scenario analysis and acknowledge this substantive limitation rather than relabelling the estimates as repaired.

### Seasonal-health pilot completed (12 November)

**Conditional diagnostic, not replacement manuscript estimates.** Runner: `code/health_assessment/pilot_seasonal_health.py DATA_ROOT NEW_OUTPUT_DIRECTORY`. Results are in `UCM_official_runs/diagnostics/seasonal_health_2026-11-12`: `citywide_comparison.csv`, `borough_comparison.csv`, `denominator_qa.csv` and `manifest.json` (input paths/hashes and assumptions). Existing rasters, figures and Word files are untouched. The runner refuses an existing output directory.

For baseline-state deaths D and scenario-minus-baseline temperature difference ΔT, the tested formula is **D × [1 − exp(log(RR) × ΔT)]**. Zero-change, warming and cooling unit checks passed. The Oceanic RR remains 1.020 (1.014–1.026). Its numerical extraction is verified, but temperature metric, threshold/domain and lag compatibility remain unresolved. No daily threshold, invented hot-day fraction or age-specific response was added. Applying this contrast assumes 2021 mortality represents the baseline risk under the modelled climate; this is a fixed-risk scenario assumption, not a projected future mortality baseline.

| Sequential calculation; avoided deaths | Green30 | Target30 |
|---|---:|---:|
| Production temperatures, annual production deaths, legacy formula | 387.37 | 495.54 |
| Corrected temperatures, annual production deaths, legacy formula | 390.94 | 498.65 |
| Corrected temperatures, annual production deaths, baseline-risk formula | 387.69 | 492.70 |
| Same, annual ONS monthly-summed deaths | 387.68 | 492.69 |
| Same, June–August ONS deaths | 80.08 | 100.49 |
| Same, July-only ONS deaths | 27.50 | 34.31 |

Baseline avoided deaths are zero by definition, not zero baseline mortality. These are paired calculations on the same positive-population/common-temperature footprint, not an assertion that the first row exactly reproduces every published rounded output. The NoData correction adds 3.57/3.11 cases; the formula correction then subtracts 3.26/5.94. Changing the annual to seasonal denominator dominates. Target30 remains larger in every citywide comparison; its summer advantage is 20.41 cases (25.5% relative to Green30), with larger benefits in 22/33 boroughs. LSOA equity curves and all six greening tiers have not been retested.

Summer results evaluated at RR limits are 56.29–103.67 (Green30) and 70.68–130.02 (Target30). These are **coefficient-endpoint sensitivities**, not full model uncertainty intervals. They exclude uncertainty in the response transfer, UHI, seasonal cooling and mortality timing.

The calculation assumes the July-parameterised cooling field persists throughout June–August and applies the log-linear response throughout that period. Therefore it addresses the annual-versus-seasonal denominator concern but **does not establish a validated seasonal heat-attributable burden**. If the source slope applies only above a heat threshold or to a different temperature metric, a compatible daily-response calculation is required before adopting these numbers.

Coverage: 14,258,588 common positive-population cells, representing 8,834,005.13 residents. Borough population coverage is at least 99.9979%. Seasonal deaths were allocated using each borough's full population denominator before masking; missing temperature cells were not compensated by redistributing deaths. Thus 11,670.977 of the 11,671 summer deaths enter the calculation. Population and mortality grids match temperature grids exactly; no resampling was used.

### Annual versus monthly mortality sources: causes and reconciliation

The author's existing Nomis files **do distinguish causes**. For example, `data/health-data/35915561408648160_data.tsv` includes mental/behavioural disorders, respiratory disease, ischaemic heart disease, cerebrovascular disease and intentional self-harm. However, the time field is a calendar year, not a month. [Nomis MORTSA](https://www.nomisweb.co.uk/datasets/mortsa) confirms annual registrations by age, sex, underlying cause and residence.

| Source | Time | Geography | Cause detail | Appropriate role |
|---|---|---|---|---|
| Existing Nomis MORTSA exports | Annual | Borough (and other levels; filter E090 explicitly) | Detailed underlying ICD-10 causes | Preserve cause definitions and annual totals |
| Downloaded final ONS 2021 monthly workbook | Monthly | All 33 London boroughs | All-cause only in this workbook | Direct borough summer denominator for all-cause pilot |
| ONS Monthly mortality analysis | Monthly | Published leading-cause summaries for England/Wales | Selected leading-cause groups | Potential temporal sensitivity only; not verified as matching borough/ICD data |

The inspected monthly workbook has only `Cover Sheet`, `Notes` and `1`; its data table contains area codes/names and January–December columns, with no cause dimension. Do not confuse it with the separate [monthly mortality analysis](https://www.ons.gov.uk/peoplepopulationandcommunity/birthsdeathsandmarriages/deaths/bulletins/monthlymortalityanalysisenglandandwales/december2021), which does publish leading causes. The latter does not by itself establish availability of the required borough × month × exact ICD groups.

The annual production allocation represents 56,945 registrations versus 56,944 from the final monthly workbook. Borough differences (production minus monthly sum) are Brent +1, Bromley +1, Greenwich −1, Lewisham +1 and Westminster −1. These are source differences, separate from the small mask losses; their precise release/revision explanation is not established. Using the ONS annual sum changes pilot estimates by less than 0.01 case. Preserve both sources and record the discrepancy rather than silently forcing equality. The Nomis export contains repeated records and multiple geographic levels: deduplicate identical records and filter the intended borough/age/sex/cause before summing.

**Low-effort route for reviewer concerns:** prioritize all-cause health with actual borough seasonal counts, conditional on resolving response applicability. Retain annual cause-specific data as provenance, but do not assume all causes share the all-cause summer fraction. For confirmatory seasonal cause-specific results, seek/request 2021 month-of-occurrence (or clearly labelled registration) × borough × matching ICD-group counts, both sexes/all ages, including disclosure rules. No request has been sent. If unavailable by submission, move unresolved cause-specific results out of confirmatory reporting; a national cause-specific seasonal fraction would be a labelled sensitivity requiring matching ICD groups and an explicit geographic-transfer assumption, not an equivalent substitute. Cardiovascular/respiratory age-65+ coefficient compatibility remains a separate issue. A non-pandemic-period sensitivity is useful later, but should not silently replace the agreed 2021 baseline.

### Follow-up verification on 12 November

The isolated correction experiment uses current production baseline and revised Green30 at 25/5°C, not the historical Green30 scenario. Each is run twice with identical valid land cover: NoData0 and NoData255. Private biophysical tables exclude the declared missing-value keys. The runner refuses an existing output directory, checks input hashes after each run, and leaves production paths untouched. Results belong under `UCM_official_runs/diagnostics/nodata_correction_2026-11-12`. Runner: `code/Urban_Cooling_Modeling_Runs/test_nodata_correction.py`.

**Completed result:** all four runs succeeded. Corrected NoData0 and NoData255 have identical valid masks and exactly zero maximum absolute difference for shade, cooling capacity and final air temperature, in both baseline and Green30. All have 15,946,858 valid cells. Thus the lookup correction removes the sentinel dependence in this test. This is a numerical invariance test, not empirical climate validation.

| Quantity on 15,733,893 common London cells | Existing production | Corrected diagnostic |
|---|---:|---:|
| Baseline area-weighted temperature, °C | 28.560576 | 28.492241 |
| Green30 area-weighted temperature, °C | 28.269617 | 28.193774 |
| Paired Green30 area-weighted cooling, °C | 0.290959 | 0.298467 |
| Paired Green30 mean cooling across boroughs, °C | 0.336379 | 0.341836 |

The corrected paired cooling increases by 0.007508°C (about 2.58%) area-weighted, or 0.005457°C (about 1.62%) across borough means. Correcting both baseline and intervention partly cancels their absolute changes. This is different from the earlier historical-Green30-only comparison; do not apply its 0.076973°C change to all paired production effects. The Target30 extension below completes the temperature-ranking check. Health effects remain untested. Inspect external-buffer coverage before treating any boundary convention as physically validated. Model warnings concerned polygon/multipolygon GeoPackage typing; runs completed successfully.

`comparison.json` records these results; `manifest.json` records source hashes and four workspaces. `compare_corrected.py` is preserved alongside them and uses the repository's existing `valuation_accounting.py` for a common London mask. No existing output was overwritten.

### Corrected Target30 comparison

Two additional Target30 runs completed in `UCM_official_runs/diagnostics/nodata_target30_2026-11-12`, using the same correction and 25/5°C setting. Corrected Target30 temperature masks and values are exactly identical for sentinel 0 and 255. Comparison with corrected baseline/Green30 and existing production uses the same 15,733,893 London cells; no resampling or changed aggregation footprint.

| Cooling relative to matched baseline, °C | Existing production | Corrected diagnostic |
|---|---:|---:|
| Green30, London area-weighted | 0.290959 | 0.298467 |
| Target30, London area-weighted | 0.319249 | 0.324216 |
| Target advantage, London area-weighted | 0.028289 | 0.025749 |
| Green30, unweighted borough mean | 0.336379 | 0.341836 |
| Target30, unweighted borough mean | 0.377965 | 0.381402 |
| Target advantage, unweighted borough mean | 0.041586 | 0.039565 |

**Ranking unchanged:** Target30 cools more under both summaries. Target30 is cooler in 19 of 33 boroughs in each version (this count does not by itself establish identical borough membership). The area-weighted advantage narrows by 0.002541°C. This supports stability of the 30% aggregate cooling comparison to this specific correction, not all-tier, health-equity or empirical robustness. Corrected absolute Target30 temperature is 28.168025°C versus corrected baseline 28.492241°C. Existing production and figures remain unchanged.

Scripts: `code/Urban_Cooling_Modeling_Runs/test_nodata_correction.py --scenarios target30` (after production and new-output directory arguments); `code/post_processing_layers/compare_nodata_corrected.py` (data root, baseline/Green diagnostic directory, Target diagnostic directory). The latter saves `three_scenario_comparison.json` in the Target diagnostic folder and refuses overwrite. The runner preserves input checksums; all six corrected air-temperature runs show zero mask/value difference between sentinel encodings.

### Most feasible next steps

1. Retain the Oceanic RR extraction as verified. Obtain the source Methods/supplement only to settle temperature metric, heat range/threshold and lag, not to re-open the numerical extraction.
2. Before approving a new full production version, settle the seasonal all-age calculation and baseline-risk formula in a small baseline/Green30/Target30 pilot using corrected temperatures. Use verified borough seasonal deaths rather than a single London fraction. Keep 2021 population and avoid age-specific expansion.
3. Treat a constant July-parameterised summer contrast as a stated scenario assumption; a monthly/seasonal denominator fixes duration but not response-domain compatibility. If the source requires daily above-threshold exposure, use a compatible daily series/response; a hot-day multiplier is not a validated lower bound. The aim is one defensible primary calculation, not multiple new model families.
4. After author approval of that design, regenerate the affected scenarios into a new versioned production directory and refresh dependent summaries together. The correction should be applied consistently to baseline and every intervention, including energy/work outputs if recomputed. Do not combine corrected Green with uncorrected Target, or infer unchanged health curves from stable aggregate cooling.
5. Preserve existing versions for comparison; do not replace manuscript numbers until revised health and all-scenario checks are complete. No full production rerun is authorized or performed by this diagnostic.

**Seasonal data verified by downloading and reading the final ONS 2021 workbook:** sheet `1` includes all 33 distinct E090 London borough codes and January–December counts. Their annual sum is 56,944; June–August sum 11,671 (20.4956%); July sum 4,013 (7.0473%). Borough summer shares range from 17.5879% to 27.2727%. These are all-cause registrations by usual residence, not occurrence-dated or cause-specific seasonal mortality. This range shows why applying borough-specific seasonal fractions need not preserve equity results. No seasonal health calculation has been substituted into production.

Source: [ONS monthly deaths, final 2021 edition](https://www.ons.gov.uk/peoplepopulationandcommunity/birthsdeathsandmarriages/deaths/datasets/monthlyfiguresondeathsregisteredbyareaofusualresidence). Download filename `deathsregisteredmonthly2021finalised.xlsx`; SHA256 `1f237a9cc42cc77569efb11ef96defd6023020b802e5ba9ea25055791de52cb1`. The downloaded workbook and `ons_seasonal_2021_boroughs.csv` are preserved in the diagnostic folder. The annual sum still needs reconciliation with the exact Nomis cause definition and release used by production before adoption. Earlier non-pandemic editions are listed by ONS but have not been downloaded or selected here.

**All-cause numerical extraction verified from the author's supplied Figure 3 image:** Yang 2024, Climate zone (Köppen classification), Group C (Oceanic), K=10, reports 1.020 (95% CI 1.014–1.026), exactly matching production. The abstract's 1.021 is a different pooled estimate, not evidence of an extraction error. Retain the Oceanic coefficient and record the climate-subgroup rationale. The screenshot does not identify temperature metric, thresholds, lags or temporal applicability; those still require the source Methods/supplement before converting summer exposure to mortality counts. [Publication record](https://pubmed.ncbi.nlm.nih.gov/38986701/).

**Claude addendum adjudication:** [Nomis](https://www.nomisweb.co.uk/datasets/mortsa) confirms annual age categories (under 1, 1–4, five-year bands through 85–89, then 90+). This supports the possibility of age-informed allocation, but does not establish ready-to-use seasonal age-specific borough rates, matched LSOA geography, or age-specific heat response. Small subregional counts are rounded. No age-stratified export or model was undertaken; that extension remains out of scope. “Summer daytime contrast parameterised with July” is a suitable conditional description, not evidence that every summer day experiences 5°C UHI. Multiplying summer deaths by a hot-day fraction is not automatically a defensible lower bound: thresholds, exposure changes across the threshold, daily baseline mortality and lags must be considered. Equal sentinel use across production does not cure the lookup collision.

Two calculation issues precede submission: seasonal exposure is applied to annual deaths, and NoData code 0 collides with a biophysical-table class. Neither is resolved by changing labels. Keep existing results as an archived version, not as a validated final release pending these checks.

## Health implementation

`code/health_assessment/prepare_mortality_population_weighted.py` allocates each borough's full calendar-year registered deaths as `D_b * pop_i / sum(pop_b)`. This is population-conserving; population allocation itself is not the temporal problem.

`code/health_assessment/health_modeling_v2.py` lines 256–272 calculates `deltaT = T_scenario - T_baseline`, `beta = log(RR)`, and `excess = D * (1-exp(-beta*deltaT))`. There is no season fraction, daily integration, temperature threshold or lag. The equal-area example configuration explicitly points to annual population-weighted 2021 death rasters and the 25/5°C UCM rasters. WorldClim June/August files would not automatically enter this calculation.

### Reference-risk convention

For a direct scenario/baseline risk ratio `R=exp(beta*deltaT)` and deaths D under baseline land use, the direct difference is `D*(R-1)`; avoided deaths are `D*(1-R)`. The implemented `D*(1-1/R)` instead uses an attributable fraction with a different reference denominator. This is inconsistent with the current baseline-deaths description unless a different risk-state interpretation is established.

Example with D=1,000, RR per °C=1.02 and cooling of 1°C: the current code returns 20 avoided deaths; the direct baseline-to-scenario contrast gives 19.6078. For warming of 1°C, current excess is 19.6078 versus 20 directly. This algebra issue is distinct from, and likely smaller than, the duration issue. Do not silently swap formulas: define the two states first, then test warming, cooling and zero-change cases.

### Coefficient verification

| Outcome | Current input | Source check and required action |
|---|---|---|
| All cause | 1.020 (1.014–1.026) | Verified against author's Figure 3 screenshot: Group C (Oceanic), K=10. Retain this climate-specific extraction; exposure metric/window and thresholds remain to be verified. |
| Cardiovascular | 1.0344 (1.0310–1.0378) | Exact match to Bunker et al. mortality estimate for adults 65+. Not an all-age estimate. Seek an all-age coefficient; if unavailable, withhold this endpoint from confirmatory reporting rather than invent age-specific inputs. |
| Respiratory | 1.036 (1.0318–1.0402) | Same confirmed 65+ applicability issue. |
| Mental disorders | 1.026 (1.013–1.040) | Liu et al. Table 2, p.10: Oceanic-climate mortality subgroup. It is mortality, not morbidity, but is not explicitly the F00–F99-specific pooled estimate (that row is 1.031). Verify endpoint compatibility; do not automatically replace it with the overall estimate. |
| Suicide/self-harm | 1.012 (1.003–1.021) | Liu Table 2 matches the X60–X84 mortality row. Exposure-metric/window transfer still requires justification. |

Sources: [Yang et al.](https://doi.org/10.1016/j.scitotenv.2024.174650), [Bunker et al.](https://pubmed.ncbi.nlm.nih.gov/27211569/), [Liu et al., Table 2](https://researchmgt.monash.edu/ws/files/382374807/381823556_oa.pdf).

## A feasible summer analysis without age-specific modelling

July can illustrate **warm summer conditions**, but July alone is not a measured June–August average. WorldClim Tmax is a monthly temperature summary, not the hottest individual day; future monthly values are averaged over 20 years ([WorldClim](https://worldclim.org/data/cmip6/cmip6climate.html)). Adding June/August improves seasonal climate description but does not repair mortality duration or threshold compatibility by itself.

Recommended staged route:

1. Retain fixed 2021 population and all-age modelling. Resolve the all-cause coefficient, metric and risk-state convention first.
2. Obtain 2021 monthly all-cause deaths for each borough and sum June–August, with July-only as a sensitivity. [ONS monthly registrations](https://www.ons.gov.uk/peoplepopulationandcommunity/birthsdeathsandmarriages/deaths/datasets/monthlyfiguresondeathsregisteredbyareaofusualresidence/2022) documents borough-level availability; locate and validate the 2021 edition before use. Registration month is not occurrence month; prefer occurrence data when available and otherwise disclose the timing limitation.
3. Allocate seasonal deaths using the existing population-conserving method. If using a July cooling contrast throughout summer, explicitly call it a constant-cooling summer scenario. It is an assumption, not a demonstrated seasonal estimate.
4. For a defensible heat-attributable count, use a daily summer series and a compatible all-age heat-response function/threshold, then sum daily counterfactual differences. Published-response application does not require individual or age-specific mortality microdata. Monthly death totals can supply an assumed daily baseline within each month; record this approximation. A daily-mean response cannot simply be paired with a Tmax cooling contrast.
5. Reuse UCM cooling fields only with an explicit temporal-transfer assumption; adding climate data alone does not calibrate daily cooling. Do a bounded pilot before all-scenario recomputation.

A monthly constant-response calculation is a lower-effort, narrower scenario approximation, defensible only if coefficient applicability supports that interpretation. If it does not, daily threshold-aware calculation is needed; relabelling is insufficient. Dividing current results by 12 or 4 is not a validated correction. Annual deaths times 31/365 or 92/365 is at most an explicitly assumed uniform-daily-mortality sensitivity, not a final estimate.

Uniform positive scaling would preserve curve shapes, rankings and fixed-Green thresholds if thresholds are rescaled too. Borough-specific seasonal fractions, corrected risk functions and threshold crossings need not; no claim of unchanged conclusions is verified here. Cause-specific seasonal counts may require separate data; do not silently apply all-cause season fractions to every cause.

## NoData mechanism now identified

NoData means unknown/outside coverage, not a real land-cover class. Historical Green30 uses sentinel 255; the harmonized input uses sentinel 0, with identical valid land-cover values and original masks. However, the shared biophysical CSV contains `lucode=0, Description=Nothing` with shade, Kc, albedo and green-area all zero. It does not contain a matching 255 row.

Installed InVEST 3.20.2 passes the entire biophysical mapping to `utils.reclassify_raster`; installed pygeoprocessing explicitly maps input NoData to a valid output if that sentinel occurs in the value map. Thus 0-NoData becomes a valid zero-valued surface, whereas 255-NoData remains missing. This is a mapping collision, not evidence of a climate effect or necessarily an InVEST version bug.

Read-only stage comparison of existing `archive/diagnostic_runs/green30_health_invest3202` and `green30_health_invest3202_nodata_harmonized`, at 25/5°C:

| Stage | Valid cells with 255 | Valid cells with 0 | Finding |
|---|---:|---:|---|
| Aligned LULC | 15,946,858 | 15,946,858 | Identical valid values and mask |
| Shade/Kc/albedo/green area | 15,946,858 | 26,332,560 | 10,385,702 missing cells become valid |
| Cooling capacity | 15,946,858 | 23,089,980 | ET coverage limits expansion; common-cell values still identical |
| Park cooling | 15,946,858 | 23,089,980 | 5,559,102 common cells now differ |
| Final air temperature | 15,946,858 | 23,089,980 | 8,972,401 common cells differ; mean harmonized-minus-original +0.07615475°C over common full-grid valid cells |

The first divergence is reclassification. Zero-valued additional cells enter spatial averaging and green-area calculations, then change park cooling, heat mitigation and air mixing on genuinely valid London cells. The final common-grid mean above is not the London-only statistic. The prior [London-mask comparison](figure3_ranking_input_audit.md) reports a 0.076973°C reduction in Green30 cooling (0.049786°C across borough means). Those earlier summary values were not regenerated here.

Recommended fix, not implemented: use a missing-value sentinel absent from every biophysical lookup, or explicitly exclude that sentinel from each lookup. Preserve any genuinely valid class-0 pixels if they exist; do not indiscriminately delete valid data. Inspect original spatial coverage and whether missing surroundings need real buffered land-cover inputs rather than zero-filling. Test that valid reclassified masks equal valid aligned-LULC masks and that equivalent sentinel encodings produce identical common-cell outputs within tolerance. First test baseline and Green30 in a new diagnostic directory, then assess all affected production configurations. Do not mix corrected intervention temperatures with an uncorrected baseline.

The revised production scenario configuration uses NoData0 and the same lookup convention, so its exposure to this collision must be checked before asserting numerical robustness. The effect size cannot be transferred from Green30 to other scenarios or health outcomes without testing. Existing results remain unchanged; a controlled correction may require downstream updates.

### Reproduction and boundaries

Health source: `code/health_assessment/health_modeling_v2.py`, `prepare_mortality_population_weighted.py`, `health-analysis-revised-equal-area.example.json`. Existing ranking-audit generator: `code/post_processing_layers/audit_figure3_ranking_inputs.py`. This audit creates no figures.

Installed source: `/opt/homebrew/Caskroom/miniforge/base/envs/urban-cooling-invest-3.20.2/lib/python3.12/site-packages/natcap/invest/urban_cooling_model/urban_cooling_model.py` (reclassification around lines 680–704); `natcap/invest/utils.py` around line 600; `pygeoprocessing/geoprocessing.py` around lines 2356–2432. Current woodland rows have Shade=1, Kc=0.7; the Supplement's Shade=0 is a reporting mismatch, not the NoData mechanism.

The reusable stage probe is `code/post_processing_layers/audit_nodata_stages.py`; pass the external `UCM_official_runs/archive/diagnostic_runs` directory as its sole argument. It prints results without writing rasters. The initial probe read existing rasters only; subsequent isolated correction runs and the conditional seasonal-health pilot are documented above. All-cause numerical subgroup extraction is resolved; daily-response applicability remains unresolved. Age-specific modelling, manuscript edits, production replacement and pushing remain out of scope.
