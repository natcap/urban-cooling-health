# Project documentation index

Start with the repository [README](../README.md) for orientation. This index
points to one primary document for each workflow or decision; detailed
procedures are not repeated here.

## Current work

- [Project TODO](../TODO.md): live priorities at the top; the completed
  model-rebuild ledger below is retained as historical evidence, not as a
  second method guide.
- [Full workflow map](WORKFLOW.md): ordered source-to-figure stages and links
  to the authoritative procedures.
- [Manuscript revision brief](manuscript/MANUSCRIPT_REVISION_BRIEF.md): recent
  changes, review materials, deferred questions and submission steps.
- [V0–V1 health and Figure 6 changes](manuscript/V0_V1_HEALTH_AND_FIGURE6_CHANGES.md):
  health-input revisions, population normalization and prediction-display
  decisions, distinguishing implemented changes from the agreed next update.
- [Figure 6 candidate figures and draft text](manuscript/FIGURE6_REVIEW_DRAFTS.md):
  both mortality-allocation versions, proposed legends and Results, suggested
  supplementary selection and the Figure 7 consistency review.
- [SVI and spatial sensitivity report](manuscript/SVI_SPATIAL_RESULTS.md):
  authoritative interpretation and figure attribution. Its 12 rendered images
  live in [`figures/figure6/spatial_sensitivity/`](../figures/figure6/spatial_sensitivity/)
  and [`figures/svi_spatial_report/`](../figures/svi_spatial_report/),
  separate from narrative documents and numerical analysis outputs. The local
  offline HTML is a generated, Git-ignored report export; edit the Markdown,
  not the HTML, and regenerate it after changes.
- [Figure 7 reproduction guide](../REPRODUCIBILITY.md): standalone procedure
  using tracked derived inputs.
- [Code guide](../code/README.md): canonical scripts and workflow entry points.
- [Data directory guide](../data/README.md): local data roles, preservation,
  tracking boundaries and the remaining inventory work.

## Methods and future upgrades

- [Future work-capacity method](methods/FUTURE_WORK_CAPACITY_METHOD.md): staged
  construction-sector pilot using workplace employment, hourly heat exposure
  and capacity-weighted worker-hours. This is a proposal, not a current result.
- [Figure 4 methods](../code/post_processing_layers/FIGURE4_METHODS.md):
  current citywide estimands and the borough sensitivity comparison.
- [Health data and method decisions](../code/health_assessment/DATA_SELECTION_AND_METHOD_DECISIONS.md):
  mortality source, population allocation and model assumptions.
- [Data licence and redistribution review](../DATA_LICENCE_AND_REDISTRIBUTION.md):
  source attribution and release restrictions.

The imported `London-MS-edit` chat context is kept locally in
`docs/conversation-imports/`. That folder is ignored by Git; it is background
context rather than a required reproducibility input.
