# Project documentation index

Start with the repository [README](../README.md) for orientation. This index
points to one primary document for each workflow or decision; detailed
procedures are not repeated here.

## Current work

- [Current manuscript index](manuscript/index.md): selected Figures 1–7,
  copy-ready text, authoritative reports and supporting audits. Start here for
  manuscript revision rather than searching by file date.
- [Current manuscript figure folder](../figures/manuscript_current/): one
  review folder containing the selected main figures and confirmed
  supplementary candidates, with a source/script/hash manifest.
- [Reproducibility guide](reproducibility.md): verifies the curated manuscript
  package separately from a full restricted-data rebuild.
- [Project TODO](../TODO.md): current priorities. The separate
  [historical ledger](archive/todo_history_20261118.md) preserves earlier
  milestones and technical tasks, not a second current task list.
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
- [Figure 4 methods](manuscript/figure4_methods.md):
  current citywide estimands and the borough sensitivity comparison.
- [Health data and method decisions](../code/health_assessment/DATA_SELECTION_AND_METHOD_DECISIONS.md):
  mortality source, population allocation and model assumptions.
- [Data licence and redistribution review](../DATA_LICENCE_AND_REDISTRIBUTION.md):
  source attribution and release restrictions.

The imported `London-MS-edit` chat context is kept locally in
`docs/conversation-imports/`. That folder is ignored by Git; it is background
context rather than a required reproducibility input.

Local Word review copies are grouped in
`manuscript/reviews/word_review_copies/`, with a local index and hash-verified
move manifest. They are ignored by Git and are not the author's current Word
sources. Canonical Markdown/HTML remains in `manuscript/`; do not copy review
text into multiple authoritative documents.
