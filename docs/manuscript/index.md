# Manuscript revision index

Use this page as the entry point for the current manuscript package. It links
to one authoritative document for each figure group and keeps audits,
diagnostics and historical comparisons in supporting roles.

## Current figures and copy-ready text

| Figures | Open first | What it contains |
|---|---|---|
| 1 | [Historical-reference and Figure 1 report](figure1_historical_reference_comparison.html) | Selected main/supplementary versions, legend, Results and climate-reference decisions |
| 2 | [Current figure package](../../figures/manuscript_current/readme.md) | Selected July 2021 baseline-temperature and corrected-SVI tree-cover figure; source script is `code/equity-temp-tcc.Rmd` |
| 3 | [Climate/UHI audit](figure3_climate_uhi_audit.html) | Selected v03, conditional-setting interpretation, legend, Results and provenance |
| 4–5 | [Energy and work-capacity candidate report](energy_capacity_candidate_results.html) | Current Figure 4 v05/Figure 5 v06 candidates, JJA capacity decision, Results, legends and methods |
| 6–7 | [Seasonal-health report](seasonal_health_rerun_comparison.md) | Current seasonal figures, supplementary figures, legends, Results and spatial qualifications |

All currently selected PNGs and confirmed supplementary candidates are copied
into the single [current manuscript figure folder](../../figures/manuscript_current/).
Its manifest records original source paths, generators, versions and hashes.
Extended Data Fig. 1 is not yet in that folder because its existing image
predates the seasonal-health update; its cause-specific seasonal version must
be confirmed before final submission packaging.

## Scientific decisions and supporting audits

- [Current priority list](../../TODO.md): outstanding author decisions and
  release gates.
- [Health calculation and NoData audit](health_calculation_nodata_audit.md):
  health formula and missing-data diagnosis.
- [Corrected seasonal-health comparison](seasonal_health_rerun_comparison.md):
  authoritative health estimates and equity findings.
- [Monthly outcome alignment](monthly_outcome_alignment.html): why JJA health
  and capacity were selected and why energy remains conditional.
- [Figure 6 spatial sensitivity](SVI_SPATIAL_RESULTS_offline.html): descriptive
  versus spatially adjusted associations.
- [Figure 7 review](FIGURE7_REVIEW.md): historical layout decisions and
  interpretation details; use the seasonal-health report for current numbers.
- [Pre-submission review synthesis](reviews/pre_submission_action_plan_20261112.md):
  consolidated reviewer concerns and their disposition.

## Methods and reproducibility

- [Run-version index](../methods/RUN_VERSION_INDEX.md): production, health and
  valuation run relationships.
- [Figure 4–5 methods](figure4_methods.html): figure-to-script and aggregation
  reference; current candidate values are in the candidate report above.
- [Figure 7 standalone reproduction](../../REPRODUCIBILITY.md): tracked-input
  workflow for the established Figure 7 pipeline.
- [Code and figure index](../../code/README.md): generators for every main
  figure.

## Historical material

Files not listed as current above remain analytical evidence, diagnostics or
earlier versions. Do not infer manuscript status from their modification date.
In particular, uppercase Figure 6–7 review files preserve earlier decisions;
the seasonal-health report and current-figure manifest identify the latest
selected outputs.
