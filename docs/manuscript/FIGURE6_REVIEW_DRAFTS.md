# Figure 6 candidates and manuscript text

Prepared 5 November 2026. Drafts for author review; the Word manuscript has not
been edited. Both candidates use the same revised Green30 cooling inputs,
four-dimension SVI, fixed 2021 population, all 4,835 valid LSOAs, rates per
million and full GAM predictions including the intercept. Allocation remains
an author choice; population allocation is recommended for the main analysis.
For decision rationale and controlled comparisons, use the
[V0–V1 record](V0_V1_HEALTH_AND_FIGURE6_CHANGES.md), not a second methods inventory.

The rendered candidate and comparison figures accompany this non-commercial
research report under the existing [figure release review](../../DATA_LICENCE_AND_REDISTRIBUTION.md#rendered-report-figures-review-4-november-2026).
The [source attribution and reuse notice](SVI_SPATIAL_RESULTS.md#figure-attribution-and-reuse)
applies to these figures too; the code licence does not override input-data rights.
Numerical exports, raw inputs and historical image archives remain local unless
separately released. Links to those local outputs are reproduction references,
not promises that those files are available on GitHub.

## Candidate A population allocation

[Open the 600-dpi PNG](../../figures/figure6/manuscript/equity_health_All_cause_gam_Green30_20261105_10_per_million_population_full_prediction.png).

### Proposed legend

**Fig. 6 | Neighbourhood vulnerability and modelled health benefits of greening
across London.** Curves show separate univariable generalized additive model
(GAM) predictions of all-cause heat-related deaths averted per million residents
under Green30 across 4,835 Lower Layer Super Output Areas (LSOAs). Benefits are
estimated relative to baseline land use under identical mid-century climate
conditions, holding 2021 population and registered mortality fixed. Borough
mortality is allocated to pixels in proportion to population. Vulnerability
metrics are standardized z-scores; the vertical dotted line marks the citywide
mean. Arrows indicate less to more vulnerability. SVI denotes the
four-dimension Social Vulnerability Index (Methods).
Blue lines are full fitted predictions, including the intercept, from robust
scaled-t GAMs fitted by REML; shaded bands are pointwise 95% conditional
confidence intervals. Grey densities show predictor distributions, vertically rescaled
for display rather than expressed in mortality units. All valid LSOAs are
fitted; displayed predictor ranges span the 1st–99th percentiles. These are
unadjusted citywide associations. Intervals do not account for spatial
dependence or uncertainty in upstream model inputs.

### Proposed Results

Under Green30, modelled per-resident health benefits varied nonlinearly with
neighbourhood vulnerability (Fig. 6). Fitted benefits at the 90th versus 10th
percentile were lower by 8.8 deaths per million for income deprivation, 14.9
for the proportion aged <5, 16.5 for ethnic-minority population share and 9.8
for overall SVI. Benefits were 5.7 deaths per million higher for the proportion
aged ≥75, although the curve levelled off at higher values. Spatial adjustment
attenuated these contrasts, with the under-five association becoming near zero
and changing sign across sensitivity settings. Substantial residual spatial
dependence remained, so these patterns should not be interpreted as independent
causal effects of vulnerability.

The spatial sentence is supported by the existing population-allocation
[spatial report](SVI_SPATIAL_RESULTS.md). It is not a result for area allocation.
The age variables describe neighbourhood composition, not age-specific deaths.

## Candidate B uniform area allocation

[Open the 600-dpi PNG](../../figures/figure6/manuscript/equity_health_All_cause_gam_Green30_20261105_10_per_million_area_full_prediction.png).

This is a controlled V0-style reconstruction on the revised grid, not the
historical V0 result. It can serve as a Supplementary figure; a final number
should be assigned only after the main/SM choice.

### Proposed legend

**Fig. Sx | Sensitivity of neighbourhood health-benefit patterns to uniform-area
mortality allocation.** Curves show separate univariable GAM predictions of
all-cause heat-related deaths averted per million residents under Green30 across
4,835 London LSOAs. Borough 2021 registered deaths are distributed uniformly
across equal-area pixels, preserving borough totals. Revised cooling fields,
exposure–response parameters, population denominators and neighbourhood data
are otherwise unchanged from the population-allocation analysis. This is a
controlled reconstruction of the earlier allocation assumption, not a rerun
of the complete historical pipeline. Vulnerability metrics are standardized
z-scores; the vertical dotted line marks the citywide mean. Arrows indicate
less to more vulnerability. Blue lines are
full predictions, including the intercept, from robust scaled-t GAMs fitted
by REML; bands are pointwise 95% conditional confidence intervals.
Grey densities indicate predictor distributions
and are rescaled independently of mortality units. All LSOAs are fitted, with
P1–P99 displayed. Associations are not spatially adjusted, and intervals omit
spatial dependence and upstream input uncertainty.

### Proposed Results

Under uniform-area mortality allocation, the fitted P90–P10 contrasts retained
the same directions as the population-allocation analysis: −13.2 deaths per
million for income deprivation, −20.1 for the proportion aged <5, −15.6 for
ethnic-minority population share, −12.8 for SVI and +9.9 for the proportion
aged ≥75. The older-age curve increased more consistently, while deprivation
and SVI contrasts were more negative. Thus, broad directions persisted, but
their magnitudes and nonlinear shapes depended on the mortality-allocation
assumption. Residual spatial autocorrelation remained substantial; these
unadjusted associations are descriptive, not independent effects of vulnerability.

Spatially adjusted area-allocation models are **on hold**; see the
[Figure 6 TODO](../../TODO.md). Focus review on the main citywide patterns.
Candidate A's spatially adjusted conclusions do not apply to B.

## Suggested supplementary selection

1. Retain the population-allocation spatial sensitivity (k = 50, 100 and 200),
   with its residual-dependence caveat. The curves there remain centered
   vulnerability contributions; they are not the full predictions above.
2. Use Candidate B or the allocation overlay as one allocation-sensitivity
   figure. Avoid including both if they duplicate the same message.
3. Keep the side-by-side normalization comparison in the local decision record;
   add it to the Supplement only if needed to explain the estimand choice.

## Figure 7 consistency review

The read-only [consistency test](../../code/tests/test_figure7_reporting_consistency.R)
passed on 5 November 2026: 4,835 paired LSOAs, the stored four-dimension SVI,
the same revised population-allocated health input and 2021 population lookup,
and all saved high-vulnerability/low-benefit counts reproduced.

- Benefit classes use **rates per resident**, not raw LSOA death totals.
  Green30's tertile thresholds are reused for Target30. Figures 6 and 7 use
  different summaries of vulnerability (continuous GAM versus tertile groups),
  not different SVI constructions.
- Saved Figure 7 thresholds are still per 100,000. A read-only test confirms
  multiplying rates and thresholds by ten preserves every classification.
  The corresponding thresholds are 33.20943 and 51.94653 per million.
  **Next Figure 7 display edit:** rescale rate fields and labels together,
  leaving percentages, counts and absolute deaths unchanged. No Figure 7
  outputs have been replaced in this step.
- Retain the age-75+ trade-off: the high-age-75, low-benefit share rises from
  28.7% to 59.0%, with 26.8 fewer deaths averted in those neighbourhoods under
  Target30. This concerns neighbourhood age composition, not age-specific
  mortality, and rules out claiming improvement for every vulnerability group.
- Label the paired-LSOA bootstrap as composition sensitivity; it is not
  spatially corrected inference or propagated health-model uncertainty.
- Review the realized canopy-budget audit before final equal-budget wording;
  equal canopy area does not establish equal costs or numbers of trees.

Evidence: [Figure 7 headline table](../../figures/equity_map_biscale_revised_equal_area/fig7_quantitative_headline_results.csv)
and [production code](../../code/equity-health-fig7-production.Rmd).

## Current figure presentation

Both candidate PNGs retain full predictions and confidence bands, with no
P10/P90 markers or marker legend. Numerical percentile contrasts remain in
the Results and reporting tables. The original association/outcome/scenario
heading is restored; allocation is identified by filename and the legends
above. Arrows start beside z = 0, point to −2/+2, with labels at −1.5/+1.5
above the density strip. Density heights are display-only, not deaths.
The in-figure caption is retained as commented code; uncertainty qualifications
remain in the manuscript legends above.

## File organization and reproduction

- `figures/figure6/manuscript/`: both dated, allocation-labelled candidates;
  earlier versions are preserved.
- `figures/figure6/statistical_reporting/population_allocation/` and
  `area_allocation/`: contrasts, model diagnostics, residual checks, curve data,
  input checksums, session information and candidate manifests.
- `figures/figure6/allocation_normalization_comparison/`: controlled comparisons
  and side-by-side review images, separate from candidate figures.
- This document owns the proposed legends and Results. The V0–V1 record owns
  the method-change rationale; TODO owns outstanding decisions.

`Rscript code/run_figure6_candidates.R 20261105` runs only the two canonical
production chunks from `equity-health.Rmd`, once per allocation, and checks
contrasts against the controlled comparison. It refuses to replace completed
candidate outputs. No model equations are duplicated in the runner. Both
candidate contrast checks and the Figure 7 consistency test passed. No Word
manuscript changes were made. Git publication does not imply final author
selection of a candidate.

For annotation-only updates, append `--redraw` to the command. This reuses
saved curves and contrasts, updates only the two candidate PNGs, and verifies
that all other existing figure and reporting files remain unchanged.
