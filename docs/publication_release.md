# Separate figure-reproduction release

Prepared and released 19 November 2026:

- public repository: [`natcap/urban-heat-health`](https://github.com/natcap/urban-heat-health);
- initial commit: [`693d0e7`](https://github.com/natcap/urban-heat-health/commit/693d0e7);
- release: [`v0.1.0`](https://github.com/natcap/urban-heat-health/releases/tag/v0.1.0).

The public [data guide](https://github.com/natcap/urban-heat-health/blob/main/docs/data.md)
is the reader-facing source and attribution inventory. Keep source URLs there
rather than duplicating the full list in this release record.

The author approved the packaged summary inputs and a focused 13-figure release.
An archived DOI remains pending connection to an external archival service.

## Local package

The review copy is the sibling folder `../urban-heat-health` beside this repository.
Start with its `readme.md`. Release checks and future update instructions are in
its ignored `local_review/maintenance.md`; reader-facing data information is in
`docs/data.md`. Its manifests are the
authoritative file-level inventory; do not maintain another inventory here.

It contains selected final figure renderers, small summary-level inputs, 13
publication PNGs and concise reproduction/attribution instructions. It excludes
historical comparisons, manuscript drafts, peer reviews, credentials, raw land
cover, scenario rasters, UCM rasters and large building datasets. Data and
reference images used for pixel comparisons remain local and ignored by Git.
`code/` and generated `figures/` are flat folders. The former `outputs/` folder
has been replaced by `figures/`; reference PNGs and maintenance records remain
local-only. The public verification command does not require either local folder.

The package redraws final results; it does not rerun InVEST or refit the health
models. Full production remains in this research repository. The public README
states that boundary explicitly and records the tested R, Python, spatial-library
and upstream InVEST versions.

## Clarifications found during preparation

- Root `REPRODUCIBILITY.md` describes the **earlier annual-health** Figure 7,
  not the selected seasonal manuscript figure. Its title and README links now
  state that distinction; historical commands/data are retained.
- Figure 2's original July 2021 temperature CSV has four non-finite LSOA means:
  temperature uses 4,831 finite observations, tree cover 4,835. The package
  preserves all 4,835 score/quintile assignments and does not impute missing
  temperatures. Check caption sample sizes before submission; Word was not edited.
- Extracting only the production plotting chunks removes dependencies on
  exploratory notebook sections and private cloud paths without changing the
  scientific inputs. Figure 6 redraws saved full-response predictions rather
  than refitting GAMs.
- Figure 2's initial package rendering difference is resolved: retain the
  original 300-dpi panel exports before the 600-dpi assembly and run its renderer
  in an isolated R process. Preserve pooled display-percentile interpolation.
  The selected reference PNG and all scientific values are unchanged; consult
  the package's validation report for the pixel-comparison results.
- Figure numbering was reconciled with `London urban cooling-ms-1118.docx` and
  `London urban cooling-SM-1118.docx`: Green and Target overlays are Extended
  Data Figs. 2 and 3, and the combined climate maps are Supplementary Fig. 12.

## Released scope and maintenance

The release includes the reviewed summary-level inputs needed for the 13 indexed
figures. Source-specific attribution and third-party terms remain in the public
data guide; the GPL code licence does not replace those terms. The
[existing licence review](../DATA_LICENCE_AND_REDISTRIBUTION.md) remains the
main-repository record for raw and upstream data that were not released.

The ignored `local_review/maintenance.md` in the sibling package defines how to
refresh code, inputs, checksums and reference comparisons after future manuscript
revisions. Do not publish its reference PNGs or maintenance records. After each
approved update, rerun validation, create a new tagged release and update this
record. The original research worktree and its unrelated pending changes remain
preserved.
