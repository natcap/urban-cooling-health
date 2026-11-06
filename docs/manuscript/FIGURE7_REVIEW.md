# Figure 7: numerical and reporting review

Reviewed and revised 5 November 2026. **Numerical checks pass; the requested
display revisions are implemented. Author review remains pending.** The
initial audit was read-only; the subsequent authorized revision regenerated
Figure 7 PNGs and reporting tables from unchanged health inputs. No Word edits,
UCM/health-model reruns or GitHub push were performed.

## Current deliverables

Files are organized by type in [the Figure 7 folder index](../../figures/figure7/README.md):
PNGs in `png/`, numerical tables in `csv/`. The production script and reporting
test have been checked against this layout.

The preferred main candidate now follows the manuscript's map-led layout:
two SVI maps, complete low/moderate/high benefit bars for high-SVI LSOAs, and
a compact four-indicator low-benefit comparison. Earlier expanded versions
are preserved unchanged. The supplementary version includes population aged <5.
SVI evaluates the intended distribution of targeting, not independent
validation of the targeting rule; both age indicators describe neighbourhood
composition, not age-specific deaths.

- [Supplementary all-five-indicator PNG](../../figures/figure7/png/fig7_complete_supplement_all_five.png)
- [Supplementary comparisons](../../figures/figure7/png/fig7_cd_supplement_all_five.png)
- [Combined Figure 7 PNG](../../figures/figure7/png/fig7_complete_upgraded.png)
- [Map panels a–b](../../figures/figure7/png/fig7_ab_upgraded.png)
- [Comparison panels c–d](../../figures/figure7/png/fig7_cd_upgraded.png)
- [Entry/exit counts](../../figures/figure7/csv/fig7_low_benefit_entries_exits.csv)

The map legend and reporting tables use per-million rates. In the earlier
expanded exports linked above (not the preferred v4 candidate), panel d uses
labelled, zero-based bars for remained low, left low benefit and newly low
benefit across all high-vulnerability neighbourhoods. Neighbourhoods with
moderate/high benefit in both scenarios are explicitly excluded from the bars
but retained in the entry/exit table. Panel c no longer displays non-spatial
bootstrap whiskers; those sensitivity estimates remain in the headline CSV.
PNG exports were visually checked and the consistency test now independently
checks all four transition groups and their reconciliation. Existing PDF/SVG
files are legacy exports and were not refreshed; use the PNGs above.

## Map-led candidate and reviewer synthesis

[Open the preferred map-led Figure 7 v4 PNG](../../figures/figure7/png/fig7_main_mapled_SVI_20261105_v4.png).
Version 4 centers each panel letter and title on the same horizontal line.
Maps, key, colored key text and panel c use one named palette; c uses the
high-SVI column (low to high benefit, bottom to top), without a separate legend.
All d percentages sit below their points, with scenario names above overall SVI.
Earlier versions, including v3, are preserved. These are presentation-only changes.
Version 3 builds on the author's manual scenario titles and shortened footer.
Gray subtitles share the title's left anchor. Panel d has a single gray
connector with a small matching arrowhead, percentages for all four indicators,
and scenario names beside the overall-SVI points only; its caption is removed.
Two independent AI reviews checked the code and rendered layout. The final
spacing adjustment raises the small external 4.5% label in c above the baseline;
bar values are unchanged. Numerical reporting checks also pass.
The [v2 layout](../../figures/figure7/png/fig7_main_mapled_SVI_20261105_v2.png)
is also preserved.
The [first map-led version](../../figures/figure7/png/fig7_main_mapled_SVI_20261105_v1.png)
is preserved. Version 2 places the shared map key below b, gives c–d the full
right column, removes transition counts from the image, and adds Green30-to-Target30
arrows in d with point-value labels only for SVI. Panel titles are left-aligned
beside enlarged bold letters. Transition counts remain available in the tables
and expanded figures; they are not necessary for interpreting the main composition bars.

Benefit classes are defined from all paired London LSOAs under Green30, not
recomputed within high SVI or within Target30: low ≤33.20943, moderate
>33.20943 to ≤51.94653, and high >51.94653 deaths averted per million residents
(cutpoints shown here rounded; classification uses full precision). The bars
then describe the same high-SVI subset, so they need not contain equal thirds.

The v2 review export is 180 mm wide at 600 dpi, with sans-serif body labels
predominantly 6–7 pt. [Nature Health's final-formatting guidance](https://www.nature.com/naturehealth/submission-guidelines/aip-and-formatting)
specifies maximum 180 mm width, minimum 300 dpi, and 5–7 pt sans-serif labels.
The requested PNG is a review copy: editable, unflattened artwork remains a
separate final-production requirement. No vector/PDF export was requested here.
Its [distribution table](../../figures/figure7/csv/fig7_mapled_SVI_distribution.csv)
records all three classes for the same 1,612 high-SVI LSOAs. Low/moderate/high
shares are 45.5/29.4/25.1% under Green30 and 4.5/10.9/84.6% under Target30.
The new candidate was visually inspected; the distribution test recomputes
all six bar segments from paired input records. All six previous PNG exports
match their saved SHA-256 hashes and remain unchanged.

Three independent AI reviewers assessed the design (health equity, statistics,
and simulated senior-editor/visual roles; not external journal peer review).
All favoured the original two-map structure and neutral legend arrows rather
than thumbs. The statistical reviewer favoured complete SVI benefit distributions;
the other two prioritised keeping the age-related reversal visible in a compact
cross-indicator panel. The current candidate combines both recommendations;
paired transition counts remain in the supporting tables rather than beneath
the bars. Earlier layouts remain available for comparison; the following legend
applies to the v4 map-led candidate.

## Proposed revised legend

**Fig. 7 | Redistribution of modelled health benefits under targeted greening.**
**a,b,** Bivariate maps of neighbourhood vulnerability and all-cause heat-related
deaths averted under Green30 (a) and Target30 (b), relative to baseline land use
under identical mid-century climate conditions. Maps show the corrected
four-dimension Social Vulnerability Index (SVI). The same 4,835 London LSOAs
are paired across scenarios. Vulnerability is
classified into metric-specific tertiles. Benefit classes are defined using
Green30 tertile cutpoints across all 4,835 paired LSOAs: low ≤33.21,
moderate >33.21 to ≤51.95, and high >51.95 deaths averted per million residents
(cutpoints rounded for display; unrounded values used for classification).
These thresholds are held fixed for Target30. Rates use fixed 2021 population; borough
mortality is allocated in proportion to population. Blue denotes high
vulnerability and relatively low benefit, not low canopy cover. **c,** Complete
benefit distribution among the same 1,612 high-SVI LSOAs; bars sum to 100%.
Segments run from low to moderate to high benefit, bottom to top, using the
same colours as the high-SVI column of the shared bivariate map key.
Benefit thresholds are not recomputed within the high-SVI group. **d,** Percentage
receiving low benefits within the high-vulnerability group for overall SVI,
income deprivation, population aged ≥75 and ethnic-minority population share.
Open circles denote Green30 and filled circles Target30; arrows indicate the
direction from Green30 to Target30. Percentages appear below all points;
scenario names appear above the overall-SVI points only. Percentages describe
neighbourhoods, not individual deaths, and vulnerability groups overlap.
Age composition does not identify age-specific mortality outcomes.
Panels c–d are descriptive; non-spatial
bootstrap composition sensitivity is reported separately in the supporting table.

## Proposed Results paragraph

#### Targeting redistributes benefits across vulnerability dimensions

Targeted greening altered these distributional patterns. Across Target10,
Target20 and Target30, all five 90th–10th percentile contrasts reversed direction
relative to the corresponding Green scenarios (Supplementary Fig. S2). Benefits
generally increased with income deprivation, ethnic-minority population share,
under-five population share and overall SVI, but declined with older-population
share. The SVI contrast increased from +34.0 to +58.0 deaths averted per million
residents from Target10 to Target30, compared with −4.2 to −9.8 under the
corresponding Green scenarios.

The paired neighbourhood comparison quantified this redistribution for Green30
and Target30 (Fig. 7). Using fixed Green30 benefit thresholds, the
proportion of high-SVI neighbourhoods receiving low benefits fell from 45.5%
under Green30 to 4.5% under Target30, while the proportion receiving high benefits
rose from 25.1% to 84.6%. In contrast, among
neighbourhoods with the highest proportions of residents aged ≥75 years, the
low-benefit share increased from 28.7% to 59.0%. Thus, targeting improved alignment
with composite vulnerability while revealing a trade-off for neighbourhoods with
older populations; these all-cause outcomes describe neighbourhood-level
benefits, not age-specific mortality responses.

*This section follows the [Green Results](FIGURE6_REVIEW_DRAFTS.md#proposed-results).
The first paragraph summarizes the unadjusted GAM contrasts; the second reports
fixed-threshold neighbourhood shares, not the same estimand. If the combined
Green30–Target30 Figure 6 candidate is selected, also cite Figure 6 in the first
paragraph. Numerical contrasts and supplementary curves are maintained in the
[Figure 6 review](FIGURE6_REVIEW_DRAFTS.md#six-scenario-comparison-and-supplementary-figures).*

### Supplementary figure legend

**Supplementary Fig. | Redistribution of health benefits across all five
vulnerability indicators.** **a,b,** Maps under Green30 and Target30 for overall
SVI, income deprivation, population aged ≥75, ethnic-minority share and population
aged <5. **c,** Low-benefit shares within each high-vulnerability group.
**d,** Counts remaining low, leaving low benefit and newly entering low benefit;
LSOAs with moderate/high benefit in both scenarios are not drawn. Definitions,
fixed Green30 benefit thresholds and paired LSOAs are identical to Figure 7. Age indicators
refer to neighbourhood composition; the outcome remains all-cause deaths
averted across all residents, not age-specific mortality. SVI is related to
the targeting design and is not an independent validation measure.

For manuscript placement, see the [three-reviewer recommendations](MANUSCRIPT_ORGANIZATION_REVIEW.md).
For execution instructions, use the [canonical workflow](../../code/health_assessment/FIGURES6_7_REVIEW.md#figure-7-review).
The [TODO](../../TODO.md) tracks implementation; this document records findings.

## What was checked

The production source is `code/equity-health-fig7-production.Rmd`, rather than
disabled legacy chunks in `equity-health.Rmd`. Its saved
[maps](../../figures/figure7/png/fig7_ab_upgraded.png)
and [comparison panels](../../figures/figure7/png/fig7_cd_upgraded.png)
were visually inspected.

- All 4,835 LSOAs match uniquely across health, vulnerability and population
  inputs, with positive population and finite outcomes and vulnerability metrics.
- The composite SVI uses the corrected `overall_score`; it is now in the main
  panels, with under-five results included in the all-five-indicator supplement.
- Green30 benefit tertiles are held fixed for Target30. Vulnerability classes
  also remain fixed, so changes are not caused by scenario-specific re-ranking.
- All headline counts, population-weighted shares, 45 transition cells and
  citywide totals were independently reproduced from the saved inputs.
- The existing read-only test
  `code/tests/test_figure7_reporting_consistency.R` passed. Rescaling rates
  from per 100,000 to per million changes no classifications or counts.
- The paired citywide Monte Carlo summary was reproduced from 2,000 matching,
  unique draw IDs. This verifies summary arithmetic and ID pairing, not the
  upstream provenance of common random draws, which was not re-audited here.

This audit does not revalidate raw rasters, rerun InVEST/health models, or
independently establish equal realized canopy budgets. Use the existing canopy
audit before making a matched-budget claim; equal area is not equal cost or
equal tree numbers.

## Main findings to retain

These percentages use **high-vulnerability LSOAs as the denominator** and show
the share receiving relatively low deaths averted per resident, under fixed
Green30 benefit thresholds. They are not shares of individual deaths or risks.

| High-vulnerability group | Green30 low-benefit share | Target30 low-benefit share | Change (percentage points) |
|---|---:|---:|---:|
| Income deprivation | 41.6% | 6.1% | −35.5 |
| Ethnic-minority population share | 51.4% | 3.5% | −47.9 |
| Population aged 75 years or older | 28.7% | 59.0% | +30.3 |
| Composite SVI | 45.5% | 4.5% | −40.9 |

Population-weighted summaries have the same directions. They describe the
share of residents living in these classified neighbourhoods, not individual
exposure or an age-specific mortality response. Vulnerability groups overlap:
do not add their deaths-averted totals together. SVI is related to the targeting
rule, so an improved SVI distribution is not independent validation of that rule.

Citywide point estimates are 387.37 deaths averted for Green30 and 495.54 for
Target30: an additional 108.17, or 27.9%. The paired Monte Carlo difference has
a mean of 108.49 and a 2.5th–97.5th percentile interval of 75.69–140.03. Keep
point estimates separate from Monte Carlo means and describe the uncertainty
source, rather than calling this a complete model-uncertainty interval.

Authoritative numerical sources:
[headline results](../../figures/figure7/csv/fig7_quantitative_headline_results.csv),
[transitions](../../figures/figure7/csv/fig7_paired_transition_matrix.csv),
[city totals](../../figures/figure7/csv/fig7_citywide_tradeoff.csv),
and [citywide uncertainty](../../figures/figure7/csv/fig7_citywide_uncertainty.csv).

## Initial audit findings and revision rationale

The recommendations below record the original issues and their rationale;
the implementation status and current legend are given above.

### 1. Make the transition denominator explicit

Panel d follows only high-vulnerability LSOAs classified as low benefit under
Green30. It does not show all high-vulnerability neighbourhoods or those newly
entering the low-benefit category under Target30.

For the older-population group, 247 of the original 461 low-benefit LSOAs leave
that category, but 735 other LSOAs enter it: **461 − 247 + 735 = 949**. Thus,
improvement within the initial low-benefit cohort coexists with a worsening
overall distribution. This is a selection/denominator issue, not a calculation
error.

Recommended revision: explicitly title panel d as outcomes among
“high-vulnerability LSOAs with low benefits under Green30”, and show entries
alongside exits, or provide the complete transition table in the supplement.
Keep panel c as the primary summary of the overall distributional change.

### 2. Align units with Figure 6 without changing the analysis

This initial recommendation is completed: rates and cutpoints were converted
from per 100,000 to per million in labels and reporting outputs. Absolute
deaths, percentages and percentage-point differences were unchanged; no model
rerun or reclassification was required. See the current legend above.

### 3. Separate descriptive redistribution from inference

The paired-LSOA bootstrap resamples neighbourhoods, preserves each scenario
pair and holds classifications fixed. It does not use spatial blocks or
propagate threshold, temperature, population or mortality-model uncertainty.
London's LSOAs are the study population, not an independent random sample.

Keep the main figure descriptive. If retained, label these bootstrap intervals
as non-spatial composition sensitivity and place them in supplementary reporting.
Do not equate them with the citywide risk-coefficient Monte Carlo intervals or
with the spatially adjusted associations examined for Figure 6. Stronger spatial
inference would require separately approved analysis.

### 4. Improve readability in the final PNG export

- Define blue as high vulnerability with relatively low mortality benefit,
  not low canopy cover. Give the fixed benefit thresholds in the legend.
- Use “Ethnic-minority population share” and “Population aged ≥75 years”
  rather than labels that could imply individual vulnerability or mortality.
- Enlarge the bivariate legend; keep map area distinct from LSOA/population shares.
- Improve text contrast on dark transition bars and supply exact counts for
  small segments currently lacking labels.
- Check crowded titles and the rightmost percentage tick in panel c; assemble
  panels a–d at final publication size before visual approval.
- Prefer PNG for the requested review deliverable. The runner currently also
  exports PDF/SVG; do not regenerate these merely to change display units.

## Recommended conclusion and next decision

Target30 redistributes modelled health benefits towards income-deprived and
high ethnic-minority-share neighbourhoods and increases the citywide total,
but does not improve the distribution for every vulnerability dimension.
Neighbourhoods with higher proportions of older residents show an important
trade-off. These are neighbourhood-level modelled outcomes, not evidence of
individual causal effects or age-specific effectiveness.

Figure 7 therefore adds something distinct to Figure 6: paired geographical
redistribution under common thresholds, rather than another set of smooth
associations. Retain it in the proposed Target Results subsection. Next, review
the revised PNGs and approve the legend and targeting interpretation; Word
edits and GitHub publication remain separate decisions.
