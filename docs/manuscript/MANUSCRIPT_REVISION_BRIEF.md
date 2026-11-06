# London manuscript revision brief

Original review: 13 October 2026. Status pointers updated: 4 November 2026.
Discussion window for the scientific recap: 13 September–13 October 2026.

For the current whole-manuscript organization recommendation, use the
[5 November three-reviewer review](MANUSCRIPT_ORGANIZATION_REVIEW.md).
It supersedes the provisional section/figure-placement recommendations below;
this brief retains its historical scientific recap.

November follow-up: see the [SVI comparison and Figure 6 spatial analysis](SVI_SPATIAL_RESULTS.md)
for the current results; use the [Figure 2/6/7 review checklist](../../TODO.md#manuscript-review-checklist--figures-2-6-and-7)
for remaining decisions. Spatial fitting and publication of the report's 12
illustrations are complete; manuscript insertion is not.
Figure 2 now uses July 2021 baseline temperature by author decision; future
temperature and change remain comparison outputs. Historical selection
evidence and the current reproduction instructions are maintained in the
[Figure 2 guide](../../code/README.md#figure-2-temperature-input).

The strongest manuscript framing remains a comparison of **where an equivalent amount of new canopy is allocated, who receives the health benefits, and which vulnerability groups lose relative benefit**. Figures 4–5, their common setup and the October Figure 2/6 statistical additions are committed revisions. The Figure 2 temperature input was recovered locally on 3 November; submission still depends on reviewing that panel, resolving Figure 6 spatial inference, and reconciling the manuscript with the revised outputs. Use the current [TODO](../../TODO.md) for priority and ownership; this brief preserves the October scientific recap.

This brief draws on the accessible **Nature Health-cowork**, **MS-edit**, and **London-MS-edit** discussions, current repository history, and saved output tables. The original **Nature Health Revision Tips** reference could not be reopened; its supplied preview provides historical context only. Earlier September 11–12 model and report work is identified below as inherited context, not work newly completed after September 13. September 13 evening UTC discussions occurred on September 14 in Shanghai; dates below group that review phase together.

## What changed and what needs review

| Workstream | Completed change | Review status |
|---|---|---|
| Shared R setup | Common packages, helpers, paths and fixed 2021 baseline centralized in `code/ucm_analysis_setup.R`; external root configured through `URBAN_COOLING_DATA_ROOT` | Began in commit `945bddb`, 13 September; October refinement committed as `9ccc90a`. |
| Figure 4 | Signed value labels, outcome-specific precision, health labels clear of intervals, common scenario order and compact publication typography | Commit `063cacf`, 13 September. Review all nine scenarios and the revised units/caption. |
| Figure 5 | Restored AllBuilt, TreeRisk and TreeOpp alongside Green10/20/30 and Target10/20/30; two scenario blocks and common outcome-specific zero-centred scales | Commit `e3ad67f`, 13 September. Recorded validation covered 33 boroughs × 9 scenarios × 3 outcomes = 891 records. |
| Manuscript framing and structure | Proposed a health-equity narrative, revised outline, title, abstract, Introduction framing, and cover-letter argument | Drafted in the September discussions; not inserted into the Word manuscript by this assistant. |
| Health Methods | Drafted fixed-population design, population-weighted allocation, matched-climate comparison, equations and uncertainty description | Review before insertion, including the annual interpretation and applicability of exposure–response coefficients. |
| Figure 2 statistics | Added n, medians/IQRs, Kruskal–Wallis statistics, effect sizes, BH adjustment, Dunn comparisons, outcome selection, and invalid-input checks | Committed as `2c9d7c2`. On 3 November the original LSOA climate CSVs were recovered, the temperature cache rebuilt (4,831/4,835 finite areas), and the temperature notebook completed; review the panel and spatial-inference caveat. |
| Figure 6 statistics | Official-code joins; all valid LSOAs; trimmed sensitivity; GAM diagnostics and percentile contrasts; now reported per million | Spatial sensitivity completed in November; see the linked report. Residual dependence remains, so retain descriptive interpretation. |
| Imported chat context | Presentation framing and co-author credit preserved locally in `docs/conversation-imports/LONDON_MS_EDIT_CONTEXT.md` | Imported October 8. This is a context note, not a new manuscript revision. |

Historical snapshot (not current Git status): at the 1 November repository check, local `main` and the locally cached
`origin/main` both point to `8eb215d`; the three October topic commits are
`9ccc90a`, `2c9d7c2` and `8eb215d`. No remote refresh was performed for this
status note. The seven binary modifications noted then are no longer present
at the 4 November check. Most generated outputs remain local-only; the 12
illustrated spatial-report PNGs are now explicitly versioned with attribution.

## Materials to open first

The external files in items 2, 3 and 7 are under
`$URBAN_COOLING_DATA_ROOT/2_postprocess_intermediate/UCM_official_runs/revised_equal_area_invest3202_2026-09-11_valuations/`.
`URBAN_COOLING_DATA_ROOT` denotes the Wellcome Trust Project Data folder.

1. The last supplied manuscript is `London urban cooling-ms-0806.docx` in the author's OneDrive submission folder. Its file modification date is September 16; this recap does not re-audit its current text. Treat the September language findings as a checklist to verify, since you may already have corrected them.
2. Open `summary/figure4/figure4_citywide_cobenefits.png`, its source table `summary/figure4/figure4_citywide_cobenefits_data.csv`, and [aggregation and caption guidance](../../code/post_processing_layers/FIGURE4_METHODS.md).
3. Open `summary/figure5/figure5_borough_cobenefits.png` and `summary/figure4/extended_data_citywide_vs_borough.png`.
4. [Figure 6 candidate review pack](FIGURE6_REVIEW_DRAFTS.md): current PNGs, legends, Results and links to allocation-specific reporting. Use this pack rather than older centred-curve exports for the main figure decision.
5. [Figure 7 maps](../../figures/figure7/png/fig7_ab_upgraded.png), [paired comparison panels](../../figures/figure7/png/fig7_cd_upgraded.png), and [headline results](../../figures/figure7/csv/fig7_quantitative_headline_results.csv). These provide the central distributional result.
6. [Figure 2 tree-cover statistics](../../figures/figure2_statistical_reporting/figure2_TCC_kruskal_wallis.csv) and [revised notebook](../../code/equity-temp-tcc.Rmd). Temperature results have been recovered; review the regenerated panels.
7. Open `urban_cooling_original_vs_revised_interactive_comparison_report_offline.html`. This inherited September 11 report explains original/revised workflows, canopy area/pixel/tree-equivalent comparisons, and alternative aggregation methods. Its existence was checked during the October 13 review; it was not rebuilt or visually revalidated in that recap.

Use [code/README.md](../../code/README.md) for canonical scripts,
[REPRODUCIBILITY.md](../../REPRODUCIBILITY.md) for reproduction,
[health method decisions](../../code/health_assessment/DATA_SELECTION_AND_METHOD_DECISIONS.md)
for provenance, and the refreshed [TODO](../../TODO.md) and
[Figures 6–7 review](../../code/health_assessment/FIGURES6_7_REVIEW.md) for
current action status. Official-code joins and trimming changes have been
implemented, as has the spatial sensitivity analysis. Author interpretation
and manuscript reconciliation remain outstanding.

## Findings to carry into the manuscript

The following values were checked against saved production CSVs on October 13. They are model estimates under the documented scenario assumptions.

| Quantity | Green30 | Target30 | Interpretation |
|---|---:|---:|---|
| Annual deaths averted, point estimate | 387.4 | 495.5 | Target30 adds 108.2, or 27.9%. |
| 95% exposure–response interval | 271.8–500.3 | 347.5–640.3 | These intervals do not cover all model/input uncertainty. |
| Avoided energy costs, £ million/month | 16.30 | 22.37 | Citywide total across unique buildings. |
| Modelled heavy-work capacity gain, percentage points | 0.85 | 1.00 | Area-weighted capacity, not employment-weighted output. |
| High-SVI LSOAs with relatively low benefit | 45.5% | 4.5% | 733/1,612 versus 73/1,612. |
| High-income-deprivation LSOAs with relatively low benefit | 41.6% | 6.1% | 669/1,608 versus 98/1,608. |
| High ethnic-minority-share LSOAs with relatively low benefit | 51.4% | 3.5% | 828/1,612 versus 56/1,612. |
| High-age-75+ LSOAs with relatively low benefit | 28.7% | 59.0% | 461/1,608 versus 949/1,608; this group receives 26.8 fewer modelled deaths averted under Target30. |

The age-75+ row describes all-cause mortality benefits in neighbourhoods with a high older-resident share. It is not a direct estimate of deaths averted specifically among people aged 75 or older.

The health results document also reports Monte Carlo means of 388.4 and 496.9, and a mean paired difference of 108.5 (95% interval 75.7–140.0). Those differ slightly from deterministic point estimates because of the uncertainty transformation. Choose one reporting convention consistently; do not silently mix the two.

Keep these design statements explicit:

- Population and registered mortality stay at **2021** levels in every scenario; this is not a demographic projection for 2050.
- Land-use effects compare each intervention with baseline land cover under the **same mid-century climate**. Future-versus-2021 climate comparisons answer a separate question.
- Green scenarios use **proximity-based** expansion. “Uniform planting” is inaccurate.
- Green–Target matching concerns **realized added-canopy area**, not necessarily equal financial cost, tree counts, or feasibility. Include the canopy-budget audit with the scenario table.
- Figure 4 uses citywide energy totals, area-weighted heavy-work capacity, and population-allocated citywide mortality totals. Equal-weight borough means are complementary spatial summaries.
- Figure 7 holds Green30 benefit thresholds fixed for Target30 and pairs the same LSOAs. “Blue” means high vulnerability and relatively low deaths averted per 100,000, not low tree cover.
- The 28°C health run leaves temperature differences unchanged when both baseline and intervention shift by 3°C. It checks consistency, not sensitivity of absolute heat risk to a hotter baseline.

## Manuscript framing and structure to decide

The recommended central claim is: **At a matched canopy budget, planting location changes aggregate health benefits and their distribution; targeting socioeconomic vulnerability does not automatically protect every vulnerable group.**

The September targeted literature review identified precedents for canopy–mortality benefits, configuration effects and greening inequity. The proposed novelty is their integration in a controlled allocation comparison with neighbourhood benefit incidence and competing vulnerability priorities. This was not a systematic review and does not establish a “first study” claim.

Revisit the comparators identified then: Iungman et al. (2023), Bosch et al. (2021), Zhou et al. (2021), McDonald et al. (2024), Taylor et al. (2024), Eyni et al. (2025), and He et al. (2026). Use the original September literature-review response for its linked papers and drafted Introduction/cover letter; bibliographic details and comparative claims need checking against the papers before submission.

The proposed Results sequence is: baseline heat/canopy inequality → scenario temperature effects → citywide health and supporting co-benefits → vulnerability gradients → Green30–Target30 redistribution and age-related trade-offs. Lead with health when interpreting the co-benefit figure.

The proposed five main figures are current Figures 2, 3, 4, 6 and 7. Move current Figure 1 (climate background) and Figure 5 (borough maps) to Extended Data; place scenario details and exposure–response tables in the Supplement. Figure 6's final position depends on the spatial analysis. This arrangement remains a recommendation, not an adopted renumbering.

Current [Nature Health Article guidance](https://www.nature.com/naturehealth/content), checked October 13, permits a 150-word abstract, 4,000-word main text excluding abstract/Methods/references/legends, and six main figures/tables combined. It requests no Introduction heading or Discussion subheadings and recommends approximately 60 references. The September estimates of manuscript length and tracked-change counts are historical, not a fresh count of the current file.

## Methods and wording to review

The September health Methods draft is in this chat. Its essentials are population allocation within each borough, `D0_i = D_b × P_i / sum(P_b)`, matched-climate temperature difference, `AF = 1 − exp(−log(RR) × ΔT)`, sign convention for cooling/avoided deaths, and 2,000 shared cause-specific exposure–response draws with seed 20260908. Review the exact implementation and RR age/cause applicability before treating this text as final. Also explain what annual mortality applied to the modelled temperature contrast represents.

Resolve the inconsistent GCM count/list reported in September, and align the InVEST version/citation with production **3.20.2**. Describe all nine scenarios in Methods and tables. The earlier InVEST, canopy, population and mortality corrections predate this review window but still govern the current manuscript.

Keep **modelled heavy-work capacity**, as you requested. Define ethnic-minority composition precisely using the source aggregate: Asian, Black, Mixed or multiple, and Other Census categories, excluding White categories. This is a heterogeneous administrative aggregate; neighbourhood associations are not individual-level effects. Internal `pBAME` fields can retain their provenance names while publication-facing labels use the defined term.

Check whether these September edits are already complete: `LOSA` → `LSOA`; remove “and” from “and while generating”; `neighborhood -scale` → `neighborhood-scale`; “may improve…yet failing” → “may improve…without proportionately benefiting”; “uniform” → “proximity-based” where describing Green scenarios. No explicit “first study” claim was found in that review; broad claims of absent London evidence still needed qualification.

The old £0.53 million/450–680-household analogy cannot accompany the revised £22.4 million **monthly** result. The prior discussion suggested roughly 500 annual household electricity bills for £0.53 million under the bill assumptions examined then. Revisit the analogy only after matching value, time period, tariff and household-bill basis. Do not multiply a representative warm-month estimate by 12 without a seasonal analysis. The currency is pounds; exact tariff dates/sources across building types still need a complete record.

Leave the Data Availability wording as accepted for now, per your instruction. Dataset permissions, attribution and a versioned release remain eventual submission tasks.

## Remaining technical work

1. **Review the recovered Figure 2 panel.** The October all-NA cache came from running the borough default before an LSOA join. The original LSOA CHELSA and WorldClim/GCM zonal CSVs were found under `data/02_preprocess/` on 3 November; their `borough` field actually holds numeric LSOA IDs. The original baseline–future calculation was retained, the cache rebuilt with 4,831/4,835 finite LSOAs, and the temperature notebook completed. See the [code guide](../../code/README.md#figure-2-temperature-input) for exact paths and rerun instructions. Four source-data gaps remain. Conventional rank-test p-values for both temperature and tree cover assume independence and should not be presented as spatially corrected evidence.
2. **Figure 6 spatial sensitivity.** Completed; use the [illustrated report](SVI_SPATIAL_RESULTS.md) for the analysis and its interpretation.
3. **Review Figure 6 diagnostics and interpretation.** The diagnostic seed and standalone spatial runner are implemented. Review age-indicator k diagnostics and keep the Gaussian-ML AIC comparison distinct from the primary scaled-t GAM. Follow the current TODO checklist for decisions; conditional smooth bands are not exposure–response uncertainty.
4. **Complete figure/text reconciliation.** Build one table mapping each Abstract/Results/legend number to its exact CSV row, unit, point-estimate convention and uncertainty type. Include scenario counts, canopy budgets, Figure 7 denominators and the age-75+ trade-off.
5. **Prepare the submission archive.** The illustrated report is published
   (commit `a39a1f6`); archive the final approved figure versions, configuration
   and permitted supporting data after the manuscript review. Machine-readable
   derivative permissions remain a separate task in the release review.

## Items deliberately deferred

Keep the detailed Target provenance requests on hold: road extract/date and retained classes, building and 1 m canopy sources/buffers, UTCI provenance/scale, and the SVI input/ranking definition. The collaborator's point-generation notebook was never supplied; record it as unavailable, not an expected deliverable. The current reproducible boundary starts from the preserved screened/ranked point dataset. See [Target method audit](../../code/lc_scenarios/TARGET_OPPORTUNITY_METHOD_AUDIT.md).

Retain the future work-capacity improvement separately from this revision: pilot worker-exposure weighting for construction, using workplace employment, occupational intensity/outdoor shares, hourly WBGT, and worker-hours. Test work-rest/adaptation assumptions and validate against observed disruption before expanding or monetizing. The implementation instructions are preserved in [the future work-capacity method](../methods/FUTURE_WORK_CAPACITY_METHOD.md).

Public-release provenance and permissions are maintained in
[DATA_LICENCE_AND_REDISTRIBUTION.md](../../DATA_LICENCE_AND_REDISTRIBUTION.md).
The rendered report figures have been released under the report provision;
machine-readable derivatives and weather-extract provenance remain separate
open tasks.

## Your next working session

1. Review Figure 7's age-related trade-off and adopt or revise the proposed central claim and figure order.
2. Open Figure 4 beside the manuscript and reconcile the abstract's scenario count, health estimate/interval, energy period and capacity units. Remove any surviving placeholder.
3. Review the recovered Figure 2 temperature panel, its four source-data gaps, and the rank-test inference caveat.
4. Revise the Figure 6 legend using the [spatial sensitivity report](SVI_SPATIAL_RESULTS.md).
5. Insert the agreed Methods and terminology changes, then finalize the Introduction and cover-letter framing after the scientific results are settled.

The inactivity reminder is active in this thread. It checks daily at 10:00 Asia/Shanghai, reminds after more than 48 hours without observable human project activity, and sends at most one reminder per inactive period. Automated checks do not reset the clock. Offline work may not be visible.
