# Figures 6–7 analysis review and production plan

## Status at a glance

| Figure | Current entry point | Status | Use in final manuscript now? |
|---|---|---|---|
| 6 | `../equity-health.Rmd` | Exploratory analysis with unresolved estimand and spatial-inference choices | No; complete the review gates below first |
| 7 | `../run-fig7.R` → `../equity-health-fig7-production.Rmd` | Reproducible paired Green30–Target30 workflow | Yes, after inserting the latest revised-equal-area outputs and caption |

The older Figure 7 sections in `../equity-health.Rmd` are disabled audit code.
They must not be used to regenerate the manuscript figure.

## Figure 6 review

### What the current notebook does

The active Figure 6 section selects all-cause deaths averted under Green30,
standardizes each vulnerability indicator, removes its lowest and highest 1%
of observations, fits a separate robust univariable GAM for each indicator and
plots the fitted relationship with a density trace. It also writes a model
comparison table.

### Required improvements before reporting

1. **Use the revised official-code inputs.** Replace the historical zonal table
   and row-order `id` join with
   `health_lsoa_fig7_revised_equal_area_population_weighted_2021.csv`,
   `fig7_vulnerability_lsoa11_fig7_revised_equal_area.gpkg` and
   `lsoa_population_2021_by_lsoa11cd.csv`, joined one-to-one by `LSOA11CD`.
2. **Pre-specify the estimand.** The recommended primary response is all-cause
   deaths averted per 100,000 residents using fixed 2021 population. Absolute
   LSOA deaths averted are strongly influenced by population size and should be
   retained only as a sensitivity analysis. If counts are preferred, model
   them with an explicit population offset and a count-compatible likelihood.
3. **Keep the scenario question focused.** Use Green30 for the primary Figure 6
   association because it represents the general greening strategy. Repeat the
   same analysis for Target30 as an Extended Data sensitivity. Do not put all
   six scenarios into the main curve grid; that would obscure the
   vulnerability relationship and duplicate Figures 4–5.
4. **Address spatial dependence.** The current GAM intervals treat LSOAs as
   independent. Test residual spatial autocorrelation and, if material, add a
   spatial smooth of LSOA centroids or use another pre-specified spatial model.
   Until then, label intervals as model-based conditional intervals and avoid
   causal language.
5. **Make trimming a sensitivity, not the default.** The current 1st–99th
   percentile trimming creates a different sample for every indicator. Fit the
   primary model to all valid observations, show the observed support, and
   report the trimmed result as a robustness check with retained `n` recorded.
6. **Separate marginal and adjusted questions.** Univariable curves show
   marginal associations. A multivariable model answers a different question
   and can be unstable because the vulnerability indicators are correlated.
   Keep the main figure marginal and, if adjusted curves are needed, pre-specify
   a small non-redundant predictor set and report collinearity diagnostics.
7. **Correct the uncertainty language.** A smooth-term band is not uncertainty
   in temperature, mortality, population or the exposure-response function.
   State exactly which uncertainty is represented and avoid interpreting
   smooth-term p-values as causal evidence.
8. **Record reproducible outputs.** The production replacement should export
   PNG/PDF/SVG, the exact curve data, model coefficients/diagnostics, retained
   sample counts, input checksums, package versions, scenario, outcome,
   population year and random seed where relevant.

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
