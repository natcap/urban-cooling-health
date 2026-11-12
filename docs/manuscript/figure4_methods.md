# Figure 4 aggregation and uncertainty decision

[Copy-friendly HTML](figure4_methods.html) · [Review text](figure4_5_audit.md) · [Methods and script index](figure4_methods.md)

## Decision

Figure 4 is described as a comparison of **citywide co-benefits across
London**. Its primary results therefore use a London-wide estimand rather than
an unweighted mean of 33 borough changes.

| Outcome | Primary Figure 4 estimand | Unit | Uncertainty shown |
|---|---|---|---|
| Energy | Scenario-minus-baseline avoided cost summed once across unique building records | £ million/month, using late-2025 input-price assumptions | None; deterministic model output |
| Heavy-work capacity | Scenario-minus-baseline continuous Hothaps workability, averaged over common valid 10 m cell centres inside the London borough union | Percentage points | None; deterministic model output |
| Health | Total heat-related deaths averted using population-weighted allocation of 2021 registered mortality | Annual deaths | 95% interval from paired exposure-response draws |

The three panels do not need the same mathematical weight because their
quantities are different. They do need a consistent geographic interpretation:
each primary estimate represents an aggregate or area-wide London outcome.

## Borough sensitivity

**Boundary/accounting correction, 10 November 2026.** All scenarios now use the
same London cell-centre mask (15,733,956 cells), excluding the surrounding model
domain. Borough capacity means use that same exclusive grid; weighting them by
cell count reproduces the London mean. Differences are calculated before rounding.
Energy includes buildings with positive London overlap, excludes wholly outside
buildings, and allocates cross-borough building totals by normalized intersection
area. Borough sums therefore reproduce the unique-building London total. A
boundary-crossing building's full energy estimate is retained and divided among
its London intersections; this is an explicit allocation convention, not a new
building-energy simulation. Original full-domain/intersection figures remain QA
fields, not current manuscript estimates.

The earlier manuscript scripts subtract the matched baseline within each
borough and then average the 33 borough changes without weights. This remains
useful because it:

- matches the borough maps;
- gives each borough equal influence; and
- helps reveal whether the citywide conclusion is dominated by larger or more
  building-dense boroughs.

It is not a London total and is not weighted by land area, population,
buildings or workers. It should be labelled **unweighted mean across 33
boroughs** and reported as a sensitivity or distributional result. The
production script exports borough boxplots rather than calling borough spread
a model confidence interval.

For the revised scenarios, the citywide and equal-weight borough approaches
agree on the qualitative conclusion: Target exceeds Green at the 10%, 20% and
30% tiers for avoided energy cost and heavy-work capacity. Their magnitudes
should not be interchanged because the estimands differ.

## Important productivity limitation

The study uses a modified work-capacity component: a continuous Hothaps function
of InVEST-derived WBGT replaces the built-in threshold-based work-loss indicator.
This modification is implemented in separate post-processing; InVEST's cooling,
temperature, WBGT and energy calculations remain unchanged. It is not a
demonstrated improvement in predictive accuracy for London workers.

The citywide productivity panel is weighted by valid modeled land-area pixels,
not by workers or workplaces. It is therefore a modeled heavy-work capacity
index rather than a direct estimate of London-wide economic productivity. A
future worker-weighted sensitivity would require a defensible workplace or
worker-location raster. Until such data are reviewed, do not describe the
current value as worker-weighted.

## Error-bar rule

The older zonal plotting function computes a standard error from variation
among borough values. That interval estimates uncertainty around an average
borough under strong independence assumptions; it is not parameter or model
uncertainty around a citywide total. Consequently:

- retain the paired exposure-response interval for health;
- omit uncertainty bars for deterministic energy and productivity results; and
- show borough variation with distributions, such as boxplots, explicitly
  labelled as spatial variation rather than confidence intervals.

## Manuscript text

Use the [review document](figure4_5_audit.md) for the current legends and Results.
This document owns methodological decisions, reproducibility and the figure-to-script index.

## Recommended nine-scenario layout

Use the same order in all three panels:

`AllBuilt, TreeRisk, TreeOpp | Green10, Target10, Green20, Target20, Green30, Target30`.

The first block retains the three original manuscript counterfactuals, without
a vertical separator. Target-versus-Green differences are adjacent within each
matched canopy budget and the 10% to 30% progression is preserved. Use all nine
canonical scenario colours from `code/func_colors.R`, including distinct shades
for each Green and Target tier. A horizontal zero line is required because AllBuilt and TreeRisk can
produce negative co-benefits. Do not connect bars with lines, which could imply
a continuous trajectory between categorical scenarios.

The legacy AllBuilt, TreeRisk and TreeOpp results were generated with InVEST
3.14.1. They must not be combined with the revised InVEST 3.20.2 Green/Target
results. The production workflow reruns their original LULC rasters with the
same 3.20.2 model settings, inputs and post-processing used for the other six.

The three established source notebooks now create the individual panels:

- `code/invest_result_zonal_viz_2_energy.Rmd` — Figure 4a;
- `code/invest_result_zonal_viz_3_pd_NEW.Rmd` — Figure 4b; and
- `code/health_assessment/health-modeling-output-plot-city.Rmd` — Figure 4c.

They share scenario ordering, labels and styling through
`figure4_panel_helpers.R`. `plot_figure4_citywide.R` assembles the final
three-panel figure and records a single provenance manifest.

## Copy-ready Methods text

> We quantified land-use co-benefits relative to the baseline land-use
> configuration under the same mid-century climate condition. Figure 4 reports
> London-wide estimands selected for each outcome. Avoided monthly energy cost
> was summed once across unique building records. Heavy-work capacity was
> calculated with the continuous Hothaps relationship and summarized as the
> area-weighted mean change across common valid 10 m cell centres inside the
> London borough union; this is a
> modeled capacity index and is not weighted by worker location.
>
> Gridded excess deaths, estimated using population-weighted allocation of deaths
> registered in 2021 and fixed 2021 population, were summed to obtain borough-
> and city-level totals for each cause and scenario. Equity analyses expressed
> deaths averted per million residents using the same 2021 population denominator.

Equal-weight borough summaries and boxplots have been generated and retained
for optional Priority 3 review, not confirmed for the final submission. Their
methods remain documented above; no claim of their inclusion is made in the
copy-ready Methods text.

### Artwork policy

Main Figures 4–5 omit overall titles and captions; explanatory material belongs
in their legends. Figure 4 has no redundant scenario colour key. Figure 5 uses
single-line scenario labels, aligned letter/service row headings and a compact
three-row layout. Sans-serif text, high-contrast labels and an orange–blue palette
follow [Nature's figure specifications](https://research-figure-guide.nature.com/figures/preparing-figures-our-specifications/)
and [panel-sizing guidance](https://research-figure-guide.nature.com/figures/building-and-exporting-figure-panels/)
(checked 10 November 2026). PNGs are author-review exports as requested; Nature's
editable-artwork requirement remains a final-submission task, not a claim that
PNG alone satisfies final artwork requirements.

## Reproducible outputs

### Document maintenance

These lowercase Markdown files are authoritative. `figure4_5_audit.md` owns
copy-ready text and the audit; this file owns methods and the script index.
The former `code/post_processing_layers/FIGURE4_METHODS.md` is a symlink, not
a duplicate. On case-insensitive macOS, the old uppercase audit path resolves
to the lowercase file directly; no self-referencing alias is needed.

Regenerate the white-background HTML copies from the repository root:

```sh
pandoc docs/manuscript/figure4_5_audit.md --standalone --embed-resources --toc --css docs/manuscript/figure_review.css --metadata title="Figures 4–5 review" -o docs/manuscript/figure4_5_audit.html
pandoc docs/manuscript/figure4_methods.md --standalone --embed-resources --toc --css docs/manuscript/figure_review.css --metadata title="Figures 4–5 methods" -o docs/manuscript/figure4_methods.html
```

Use lowercase names for new project files. Unrelated existing files are not
renamed by this focused reorganization because script and data references need
their own migration check.

### Current review artwork: `20261110_v2`

Scripts keep their original names. New PNGs, source CSVs and manifests carry
`_20261110_v2`; the preceding unsuffixed files are retained for comparison.
The suffix denotes styling only: estimates and uncertainty are unchanged.
Figure 4 and citywide-versus-borough supplementary bars use `alpha = 0.7`, matching Figure 3; this transparency update
retains the current `20261110_v2` filename and leaves the data unchanged.

| PNG stem (append `_20261110_v2.png`) | Generating script |
|---|---|
| `figure4_citywide_cobenefits` | [plot_figure4_citywide.R](../../code/post_processing_layers/plot_figure4_citywide.R) |
| `figure4_borough_sensitivity` | [plot_figure4_citywide.R](../../code/post_processing_layers/plot_figure4_citywide.R) |
| `extended_data_citywide_vs_borough` | [plot_figure4_citywide.R](../../code/post_processing_layers/plot_figure4_citywide.R) |
| `figure5_borough_cobenefits` | [plot_figure5_borough_cobenefits.R](../../code/post_processing_layers/plot_figure5_borough_cobenefits.R) |

Extended Data Fig. 1 is generated by `plot_figure4_citywide.R` with the additional
option `--cause-specific true`. Its separate stem is
`extended_data_fig1_mortality_by_cause_20261110_v1` (PNG, CSV and manifest).
It requires all five mortality categories and 2,000 finite draws per scenario;
the all-cause points and intervals must equal Figure 4 exactly. The old embedded
Word image is preserved as `extended_data_fig1_previous_manuscript_0806.png`.
The [audit and copy-ready legend](figure4_5_audit.md#extended-data-figure-1--verified-update)
are maintained in the manuscript review document, not duplicated here.

Both builders read [func_colors.R](../../code/func_colors.R). Figure 5 v3
(`figure5_borough_cobenefits_20261110_v3.png`) replaces the v2 filled banners
with bold coloured text on white, without boxes. Pale scenario colours are
darkened only for text to reach at least 4.5:1 contrast on white; the canonical
fill palette is unchanged. The v2 figure remains available for comparison.
Map colours still
encode benefit magnitude. All four plots omit overall titles, subtitles,
captions and redundant scenario keys, retaining panel headings and units.
Figure 4 labels are regular-weight grey, and the vertical separator is removed.
Manifests record the generator path/checksum and palette source.

The borough boxplot legend should state: boxes show the interquartile range,
centre lines the median, whiskers extend to the most extreme values within
1.5 IQR of the box, and points denote outliers across 33 equally weighted
boroughs. This is spatial variation, not model uncertainty.

Run `plot_figure4_citywide.R` after the citywide valuation summary and reviewed
health scenario set are complete. The output stems below receive the version
suffix described above (before `_data`/`_manifest` for companion files):

- `figure4_citywide_cobenefits.png`;
- `figure4_citywide_cobenefits_data.csv`;
- `figure4_citywide_cobenefits_manifest.json`; and
- when borough input is supplied,
  `figure4_borough_sensitivity.png`, its source CSV, and
  `extended_data_citywide_vs_borough.png` with its source CSV.

The manifest records the input paths, energy-summary checksum, temperature,
price-basis label, estimands and uncertainty treatment.

## Figure typography and direct labels

Figure 4 uses regular-weight grey signed value labels following the convention established
by `func_plot_change_point()`. Precision is matched to the estimand: one decimal
place for energy, two for productivity and whole deaths for health. Health
labels sit beyond the corresponding 95% interval rather than on the bar end.
The plotting code uses an 11-point base font and produces PNG output for review.
PNG is raster artwork; check legibility at its intended final size. Before submission, verify
that the effective text remains within the 5–7 point range in the
[Nature research figure guide](https://research-figure-guide.nature.com/figures/building-and-exporting-figure-panels/)
at the final typeset width and that no labels overlap after any publisher-side
resizing.
