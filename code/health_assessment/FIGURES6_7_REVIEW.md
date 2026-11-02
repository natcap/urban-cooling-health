# Figures 6–7 analysis review and production plan

## Status at a glance

| Figure | Current entry point | Status | Use in final manuscript now? |
|---|---|---|---|
| 6 | `../equity-health.Rmd` | Official-code/per-100,000 analysis and trimming sensitivity implemented; spatial inference unresolved | No; complete the spatial review gate below first |
| 7 | `../run-fig7.R` → `../equity-health-fig7-production.Rmd` | Reproducible paired Green30–Target30 workflow | Yes, after inserting the latest revised-equal-area outputs and caption |

The older Figure 7 sections in `../equity-health.Rmd` are disabled audit code.
They must not be used to regenerate the manuscript figure.

## Figure 6 review

### What the current notebook does

The current primary Figure 6 section selects Green30 all-cause deaths averted
per 100,000 residents, joins health, vulnerability and fixed-2021 population
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
   deaths averted per 100,000 residents using fixed 2021 population. Add an
   absolute-deaths sensitivity only if it answers a distinct manuscript
   question; raw LSOA totals are strongly influenced by population size.
3. **Keep the scenario question focused.** Use Green30 for the primary Figure 6
   association because it represents the general greening strategy. Repeat the
   same analysis for Target30 as an Extended Data sensitivity. Do not put all
   six scenarios into the main curve grid; that would obscure the
   vulnerability relationship and duplicate Figures 4–5.
4. **Address spatial dependence.** Residual autocorrelation is already shown
   to be substantial. Pre-specify a spatial smooth, spatial-block resampling
   or another justified approach; report remaining residual dependence and
   the change in curves/effect sizes. Until validated, label current bands as
   model-based conditional intervals and avoid causal language.
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
8. **Finish reproducible outputs.** Current exports include PNG/PDF, curve
   data, model diagnostics, input checksums and session information. Add SVG
   if needed for the journal, seed random `k.check` diagnostics explicitly,
   review low k-check p-values for age indicators, and record the final
   spatial-model configuration and sensitivity outputs beside the figure.

### Unambiguous correction made in the exploratory code

The notebook comment said that Figure 6 plotted a centered GAM term, but the
code predicted the response scale. The code now explicitly requests
`type = "terms"`, matching the zero-centred y-axis and caption. Extraction of
the GAM test statistic was also corrected to select the scalar F or chi-square
entry rather than the entire summary matrix. These repairs do not resolve the
estimand or spatial-inference review gates above.

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
- 600-dpi PNG plus PDF/SVG exports and an input/output SHA-256 manifest.

### Remaining improvements

1. Make the equal-canopy-budget validation CSV a required reviewed input to the
   final manuscript release, rather than a separate optional check.
2. Add spatial-block bootstrap or another spatial sensitivity if the Figure 7
   intervals are interpreted inferentially. The current paired LSOA bootstrap
   measures sensitivity to neighbourhood composition but assumes independent
   resampling and does not propagate UCM or input-data uncertainty.
3. Report the main three vulnerability dimensions in the figure and keep age
   under 5 and composite SVI as supplementary checks, as the production script
   currently does.

## Recommended manuscript roles

- **Figure 6:** marginal association between vulnerability and population-
  normalized mortality benefit for Green30; Target30 sensitivity in Extended
  Data.
- **Figure 7:** direct paired distributional comparison of Green30 and Target30
  at the equal 30% realized-canopy budget.

This separation avoids asking Figure 6 to compare scenario strategies and
keeps Figure 7 focused on the equity effect of targeting.
