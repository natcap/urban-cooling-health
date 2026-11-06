# Figures 6–7 analysis review and production plan

## Status at a glance

| Figure | Current entry point | Status | Use in final manuscript now? |
|---|---|---|---|
| 6 | `../equity-health.Rmd` + `../run_figure6_spatial.R` | Per-million reporting and spatial sensitivity completed; residual dependence remains | Descriptive reporting after author review, not causal or fully spatially corrected inference |
| 7 | `../run-fig7.R` → `../equity-health-fig7-production.Rmd` | Reproducible paired Green30–Target30 workflow | Yes, after inserting the latest revised-equal-area outputs and caption |

The older Figure 7 sections in `../equity-health.Rmd` are disabled audit code.
They must not be used to regenerate the manuscript figure.

Updated 5 November 2026. Two intercept-inclusive Figure 6 allocation candidates
and their proposed legends/Results are in the
[review pack](../../docs/manuscript/FIGURE6_REVIEW_DRAFTS.md). The current author decisions and remaining actions
are maintained once in the [manuscript checklist](../../TODO.md#manuscript-review-checklist--figures-2-6-and-7).
Numerical spatial results and interpretation are in the
[illustrated report](../../docs/manuscript/SVI_SPATIAL_RESULTS.md).

## Figure 6 review

### What the current notebook does

The current primary Figure 6 section selects Green30 all-cause deaths averted
per million residents for reporting (fits retain the original per-100,000
scale), joins health, vulnerability and fixed-2021 population
one-to-one by `LSOA11CD`, and fits separate marginal GAMs to all valid LSOAs.
It saves 1st–99th percentile trimmed fits as a labelled sensitivity, curve
data, P10–P90 contrasts, model diagnostics and residual Moran's I. The saved
primary residual Moran's I values are 0.891–0.941 (999 permutations,
two-sided p = 0.001), so conventional smooth p-values are not final inference.

### Required improvements before reporting

1. **Keep the revised official-code inputs.** The active section now reads
   `health_lsoa_fig7_revised_equal_area_population_weighted_2021.csv`,
   `fig7_vulnerability_lsoa11_fig7_revised_equal_area.gpkg` and
   `lsoa_population_2021_by_lsoa11cd.csv`, joined one-to-one by `LSOA11CD`.
   Preserve these checks in any production replacement.
2. **Keep the pre-specified estimand.** The primary response is all-cause
   deaths averted per million residents using fixed 2021 population. Add an
   absolute-deaths sensitivity only if it answers a distinct manuscript
   question; raw LSOA totals are strongly influenced by population size.
3. **Keep the scenario question focused.** Use Green30 for the primary Figure 6
   association because it represents the general greening strategy. Repeat the
   same analysis for Target30 as an Extended Data sensitivity. Do not put all
   six scenarios into the main curve grid; that would obscure the
   vulnerability relationship and duplicate Figures 4–5.
4. **Review the completed spatial sensitivity.** Spatial GAMs at k = 50,
   100 and 200 are now fitted and documented in the linked report. Residual
   dependence remains substantial; preserve the distinction between citywide
   patterns and adjusted associations. Label bands as conditional model
   intervals and avoid causal or fully corrected significance claims.
5. **Retain trimming as sensitivity.** The primary fit now uses all valid
   observations and the 1st–99th percentile fit is labelled separately.
   Preserve retained `n` and observed predictor support in final exports.
6. **Separate marginal and adjusted questions.** Univariable curves show
   marginal associations. A multivariable model answers a different question
   and can be unstable because the vulnerability indicators are correlated.
   Keep the main figure marginal and, if adjusted curves are needed, pre-specify
   a small non-redundant predictor set and report collinearity diagnostics.
7. **Correct the uncertainty language.** A smooth-term band is not uncertainty
   in temperature, mortality, population or the exposure-response function.
   State exactly which uncertainty is represented and avoid interpreting
   smooth-term p-values as causal evidence.
8. **Finish reproducible outputs.** Current candidate exports are PNG-only,
   with curve data, model diagnostics, input checksums and session information.
   Random `k.check` diagnostics now have an
   explicit seed and spatial sensitivity outputs/configuration are saved.
   Review low k-check p-values for age indicators before manuscript use.

### Unambiguous correction made in the exploratory code

The earlier caption described a centered effect while predictions included the
intercept. An intermediate revision used centered terms. Following author
review, the production chunks now use full response predictions with matching
labels and intercept-inclusive uncertainty. The scalar F/chi-square extraction
repair remains. Spatial-sensitivity plots still use centered contributions;
these displays answer distinct presentation questions, as explained in the
linked review pack. Neither display resolves residual spatial dependence.

## Figure 7 review

### Strengths retained

- official `LSOA11CD` joins with duplicate and completeness checks;
- identical LSOAs paired between Green30 and Target30;
- Green30 benefit thresholds reused for Target30;
- fixed 2021 population and 2021 registered mortality stated explicitly;
- population-normalized benefit used for the distributional classification;
- a fixed bootstrap seed and paired citywide exposure-response draws; and
- quantitative tables, run metadata and session information exported beside
  the figure;
- one canonical revised-equal-area output directory; and
- 600-dpi PNG exports and an input/output SHA-256 manifest. The current runner
  intentionally does not refresh legacy PDF/SVG files.

### Current Figure 7 display (5 November 2026)

The preferred map-led candidate is `fig7_main_mapled_SVI_20261105_v4.png`;
the original map-led PNG remains unchanged.
Images are saved in `figures/figure7/png/`, numerical tables and the manifest
in `figures/figure7/csv/`, and session information at `figures/figure7/`.
`URBAN_COOLING_FIG7_OUTPUT_DIR` sets this parent directory, not a file-type subfolder.
To regenerate only this candidate and its distribution CSV, preserving all
earlier PNGs and their manifest, run from the repository root:

```sh
URBAN_COOLING_FIG7_MAPLED_ONLY=1 Rscript code/run-fig7.R
```

It combines two SVI maps, the full benefit distribution within high-SVI LSOAs,
and the four-indicator low-benefit comparison. The expanded layouts below are
retained alternatives/supplementary outputs, not overwritten by this mode.

Run `Rscript code/run-fig7.R` from the repository checkout. The same output
directory's `png/` subfolder contains `fig7_ab_upgraded.png` (maps), `fig7_cd_upgraded.png`
(shares and entries/exits), and `fig7_complete_upgraded.png` (combined review).
Benefit is now per million residents; classifications are unchanged.
Main panels include SVI, income deprivation, population aged ≥75 and
ethnic-minority share. The corresponding `fig7_*_supplement_all_five.png`
exports add population aged <5, using the same renderer and thresholds.
In these expanded alternatives, panel d covers all high-vulnerability LSOAs and separates remained low, left
low and newly low, with neither-low counts retained in
`fig7_low_benefit_entries_exits.csv`. The main figure is descriptive; non-spatial
bootstrap sensitivity remains in the headline CSV, not plotted as confidence
whiskers. See the [review and proposed legend](../../docs/manuscript/FIGURE7_REVIEW.md)
for interpretation. Validate with `Rscript code/tests/test_figure7_reporting_consistency.R`.

### Remaining improvements

1. Make the equal-canopy-budget validation CSV a required reviewed input to the
   final manuscript release, rather than a separate optional check.
2. Add spatial-block bootstrap or another spatial sensitivity if the Figure 7
   intervals are interpreted inferentially. The current paired LSOA bootstrap
   measures sensitivity to neighbourhood composition but assumes independent
   resampling and does not propagate UCM or input-data uncertainty.
3. Review the four-indicator main figure and all-five-indicator supplement.
   SVI measures the intended targeting distribution, not independent validation;
   age-composition associations must not be described as age-specific mortality.

## Recommended manuscript roles

- **Figure 6:** unadjusted citywide associations under Green30; the combined
  Green30–Target30 candidate remains an author choice. Supplementary S1–S2
  compare all three intensities within each scenario family.
- **Figure 7:** direct paired distributional comparison of Green30 and Target30
  using fixed Green30 benefit thresholds. Confirm the realized-canopy audit
  before final matched-budget wording.

Keep Green and Target in separate Results sections regardless of the Figure 6
candidate selected. Figure 7 adds neighbourhood class shares rather than
repeating GAM curve findings. Use the linked manuscript drafts for wording.
