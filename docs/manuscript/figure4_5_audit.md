# Abstract and Figures 4–5: audit and draft text

[Copy-friendly HTML](figure4_5_audit.html) · [Review text](figure4_5_audit.md) · [Methods and script index](figure4_methods.md)

Audited 10 November 2026 against the saved `London urban cooling-ms-0806.docx`
and `London urban cooling-SM.docx` (accepted-change text), embedded Figure 4
images and the production Figure 4 CSV. All 21 checked input/table checksums
matched its manifest. Word remains author-managed; no new runs are implied.

**Current sequence:** clarify [run versions](../methods/RUN_VERSION_INDEX.md)
first, agree the Figure 4/5 source package, then revise the figures and prose
together. Drafts below are proposals, not final author approval.

## Completed and held items

- Saved Word confirms separate Green/Target Results, Figure 7 before Discussion,
  updated fixed-threshold findings, Target overlay as Extended Data Fig. 3,
  Green overlay S6-1 and map-only S7; S6-2 is no longer present as a caption.
- Health Methods comparator/units and GCM-count corrections: author reports
  completed after the audit; not rechecked in Word since that message.
- Discussion generalization across Green/Target: on hold by author.
- Figure 2 sample-size review: set aside at author's request.
- Figure 5 and cause-specific figures: coordinate with Figure 4, not a separate update.
- Figure 6 uncertainty qualification: Priority 3.

## Verified discrepancies

| Location | Current at audit | Production evidence / revision |
|---|---|---|
| Abstract health range | 347–640 deaths across greening scenarios | This is Target30's 95% exposure–response interval, not a scenario range. Green/Target point estimates span 135–496 annual deaths. |
| Abstract / Results AllBuilt | 1,078 additional deaths | 1,110; 95% interval 782–1,427 additional deaths. |
| Abstract energy | £22.4 million without period | £22.4 million/month; modelled warm-month conditions, not an annual total. |
| Abstract capacity | up to 2.2% productivity gain | Up to 1.42 percentage points in London-boundary, area-weighted modelled heavy-work capacity (corrected follow-up below). |
| Abstract targeting | quantitative placeholder | High-SVI low-benefit share: 45.5% → 4.5%. |
| Figure 4 artwork / Results | Old six-scenario outputs, energy −1.51/+0.53, capacity −7.40/+2.19, health −1,078/+377 | Replace with the consistent nine-scenario package, not only the caption. |
| Figure 4 legend | Six scenarios; generic 95% CIs | Nine scenarios; health-only exposure–response intervals. Energy/capacity deterministic. |
| Figure 5 legend | Six scenarios | Check artwork and update jointly with Figure 4. |

Abstract temperature extrema +1.05°C/−0.37°C were **not validated** by this
Figure 4 audit. Check Figure 3 sources before retaining them. Do not mix a
borough mean, citywide raster mean and local maximum.

## Pre-boundary-correction values (historical audit, superseded)

These are deterministic points, not Monte Carlo means. Negative deaths averted
mean additional deaths. All values below are rounded from the saved CSV.

| Scenario | Avoided energy (£m/month) | Capacity change (percentage points) | Annual deaths averted |
|---|---:|---:|---:|
| AllBuilt | −49.59 | −4.66 | −1,110 |
| TreeRisk | −4.63 | −0.25 | −105 |
| TreeOpp | 2.80 | 0.23 | 56 |
| Green10 | 5.77 | 0.31 | 135 |
| Target10 | 8.76 | 0.38 | 188 |
| Green20 | 10.94 | 0.59 | 258 |
| Target20 | 15.59 | 0.70 | 345 |
| Green30 | 16.30 | 0.85 | 387 |
| Target30 | 22.37 | 1.00 | 496 |

Source: valuation `summary_citywide_cobenefits_data.csv` and
its colocated manifest; roots are defined in the run index. See
[aggregation decisions](figure4_methods.md)
for unique-building totals, area-weighted capacity, borough sensitivity and
uncertainty. Do not annualize monthly energy by multiplying by 12.

## Corrected London-boundary results — 10 November follow-up

**Author-selected Figure 5, 10 November 2026:** use `_20261110_v3.png` (bold
coloured text, no banner boxes; v2 retained for comparison only). Figure 4 and supplementary
plots retain `_20261110_v2.png`. These styling-only updates preserve
all numbers. [Output-to-script mapping and style rules](figure4_methods.md#current-review-artwork-20261110_v2)
are maintained in one place; scripts themselves are not versioned by filename.

The initial audit above reproduced the then-current full-domain capacity outputs.
The subsequent authorized correction uses 15,733,956 common valid London cell
centres rather than 23,089,980 cells across the wider model domain. The corrected
Green30 and Target30 capacity gains are **1.19 and 1.42 percentage points**;
equal-borough means are **1.39 and 1.68**. AllBuilt becomes **−6.36** points.
Energy now excludes 70 wholly outside buildings and allocates cross-borough
building totals by intersection-area shares; 2,223,411 buildings contribute.
Citywide Green30/Target30 savings remain **£16.29m/£22.37m per month** at two
decimals. Health totals and intervals are unchanged.

Current figures and `figure45_aggregation_comparison.{csv,md}` are pooled in
the valuation `summary/` folder. Use these, not the historical table above.
Previous summary outputs are preserved in
`archive/summary_before_london_boundary_20261110/`.

## Revised Abstract — draft for review

Urban greening can reduce heat exposure, but its health benefits may be unevenly
distributed. We combined land-use scenarios, climate projections and spatial
models of cooling, mortality, energy costs and heavy-work capacity across London.
Under identical mid-century climate conditions, Green and Target scenarios
averted 135–496 modelled annual deaths, holding 2021 population and mortality
fixed. Avoided energy costs reached £22.4 million per month, and area-weighted
heavy-work capacity increased by up to 1.42 percentage points. Benefits under
proximity-based greening were generally lower in neighbourhoods with greater
income deprivation and ethnic-minority population shares. Compared with Green30,
Target30 reduced the share of high-SVI neighbourhoods receiving relatively low
health benefits from 45.5% to 4.5%, but increased that share in neighbourhoods
with older populations. These findings suggest that planting location shapes
both aggregate benefits and equity trade-offs, requiring explicit choices
between vulnerability priorities.

This draft omits the unverified temperature extrema and emphasizes the
Green/Target comparisons. It does not imply demographic projections to 2050.

## Figure 4 Results — proposed replacement

Across Green and Target scenarios, modelled annual heat-related deaths averted
ranged from 135 to 496, with larger citywide benefits under Target at each canopy
tier across all three outcomes (Fig. 4). Target30 averted 496 deaths (95%
exposure–response uncertainty interval, 347–640), compared with 387 (272–500)
under Green30. Their respective avoided energy costs were £22.4 million and
£16.3 million per month, and area-weighted heavy-work capacity gains were 1.42
and 1.19 percentage points. Conversely, AllBuilt resulted in 1,110 additional
annual deaths, £49.6 million higher monthly energy costs and a 6.36-percentage-point
capacity loss. These comparisons isolate land-use effects under identical
mid-century climate conditions, holding 2021 population and mortality fixed.

## Figure 4 legend — proposed replacement

**Fig. 4 | Citywide co-benefits of alternative land-use scenarios under
mid-century climate conditions.** Bars show scenario-minus-baseline changes in
**a,** total avoided energy costs (£ million/month); **b,** London-boundary,
area-weighted mean modelled heavy-work capacity (percentage points); and **c,**
annual all-cause heat-related deaths averted. Nine scenarios (Table 1) are
evaluated under identical climate conditions, holding 2021 population and
registered mortality fixed. Colours identify scenarios; labels give estimates.
Negative values indicate losses. Health error bars show 95% exposure–response
uncertainty intervals; energy and work-capacity estimates are deterministic.
Energy values represent modelled warm-month conditions, not annual savings;
work capacity is area-weighted, not worker-weighted. See Extended Data Fig. 1
for cause-specific mortality results.

Extended Data Fig. 1 has now been regenerated from the same revised health
outputs and checked against Figure 4 (audit below). The Word image still needs
author replacement; the cross-reference can be retained with that replacement.

## Extended Data Figure 1 — verified update

Checked the image embedded as `word/media/image39.png` and its legend in
`London urban cooling-ms-0806.docx` on 10 November 2026. It showed only six
scenarios, including Green30 all-cause deaths averted of 377 and AllBuilt of
−1,078, rather than current deterministic values of 387 and −1,110. The old
notebook also summarized draw means; current Figure 4 uses deterministic RR
point estimates. Differences therefore should not be attributed to mortality
allocation alone. The old figure did not display uncertainty intervals.

The updated five-panel figure includes all nine scenarios, retains the
point-and-stem design, uses the canonical scenario colours and adds 95%
exposure–response intervals. The old legend's reference to temperature/UHI
strata does not describe this matched-climate figure and is removed. Selected
cause-specific outcomes are not an exhaustive or additive decomposition of
all-cause mortality.

- Current: `extended_data_fig1_mortality_by_cause_20261110_v1.png`.
- Previous: `extended_data_fig1_previous_manuscript_0806.png`, extracted unchanged
  from Word for comparison; the Word document itself is untouched.
- Both reside in the valuation `summary/` folder. The new source CSV and
  manifest record all 45 estimates, intervals and source-file checksums.
- Validation: nine scenarios × five outcomes; 2,000 finite draws each; sign
  reversal before interval calculation; all-cause estimates and interval limits
  match Figure 4. Script/command details are in
  [Figure 4 Methods](figure4_methods.md#reproducible-outputs).

### Revised Extended Data Figure 1 legend

**Extended Data Fig. 1 | Modelled heat-related deaths averted by mortality
category under alternative land-use scenarios.** Points show annual London-wide
deaths averted for **a,** all causes; **b,** cardiovascular diseases; **c,** mental
disorders; **d,** respiratory diseases; and **e,** self-harm. Each scenario is
compared with baseline land use under identical mid-century climate conditions,
holding 2021 population and registered mortality fixed. Positive values indicate
deaths averted and negative values additional deaths. Points and labels give
deterministic estimates; error bars show 95% intervals from 2,000 exposure–response
uncertainty draws. Colours identify scenarios as in Fig. 4. Panels use different
y-axis scales; cause-specific estimates should not be summed to obtain the
all-cause total.

## Figure 5 Results — proposed replacement

Citywide gains masked substantial borough-level variation (Fig. 5). Under
Target30, annual deaths averted ranged from 0.29 to 29.75 per borough, avoided
energy costs from £0.08 million to £2.53 million per month, and heavy-work
capacity gains from 0.30 to 3.61 percentage points. The largest gains occurred
in different boroughs: Newham for health, Hillingdon for energy and Barking and
Dagenham for work capacity. Although Target30 exceeded Green30 citywide for all
three outcomes, it delivered greater health and energy benefits in 22 of 33
boroughs for each outcome, and greater work-capacity gains in 21. Targeting
therefore increased aggregate benefits without improving every borough's
outcomes relative to Green30.

Verification: calculated directly from the 891 rows (33 boroughs × nine
scenarios × three outcomes) of `figure5_borough_cobenefits_20261110_v3_data.csv`
(numerically unchanged from v2).
The sets of 22 boroughs need not be identical across outcomes. Decimal deaths
are modelled expectations, not observed counts. Counts compare point estimates,
not statistically significant differences.

## Figure 5 legend — corrected maps

**Fig. 5 | Borough-scale changes in cooling-related co-benefits.** Maps show
scenario-minus-baseline changes under identical mid-century climate conditions:
**a,** total avoided energy costs (£ million/month); **b,** area-weighted mean
modelled heavy-work capacity (percentage points); and **c,** annual all-cause
heat-related deaths averted across 33 London boroughs. Population and registered
mortality are held at 2021 levels. Cross-borough building energy is apportioned
by intersection area, conserving the London total. Bold coloured labels identify
scenarios using the Fig. 4 colour families, darkened where needed for legibility.
Within each row, a shared symmetric pseudo-logarithmic
(asinh) scale shows losses in orange, zero in white and gains in blue; colourbar
labels retain original units. Maps depict spatial variation, not uncertainty.

## Next author review — Figures 4–5

**Author-confirmed completed, 10 November 2026** (reported by the author;
Word has not been re-audited after these updates):

- [x] Figure 4 v2 numerical wording and Results reviewed.
- [x] Figure 5 v3 Results and legend reviewed; v3 selected.
- [x] Updated Extended Data Fig. 1 adopted.
- [x] Word integration and associated Abstract/cross-reference updates completed.

The revised health aggregation wording is maintained once in the
[copy-ready Methods](figure4_methods.md#copy-ready-methods-text).
Figure 3 temperature extrema remain outside this audit's validation scope.

### Priority 3 — optional supplementary comparisons

- [ ] Review `extended_data_citywide_vs_borough_20261110_v2.png` and its
  comparison table later; retained in the valuation summary folder.
- [ ] Review `figure4_borough_sensitivity_20261110_v2.png` (borough boxplots)
  later; also retained in that folder.

Neither figure is currently committed to the final submission. Decide on
inclusion and supplementary numbering only if revisited; the draft legend
below is retained for that purpose, not an outstanding immediate requirement.

No further model run is required for these figure updates. Git publication
remains on hold pending explicit approval.

## Supplementary aggregation comparison legend

**Supplementary Fig. [number to assign] | Sensitivity to geographic aggregation.**
Panels compare London-wide energy totals and London-boundary, area-weighted
heavy-work capacity with equal-weight means of the corresponding changes across
33 boroughs. Energy is expressed in £ million/month for London and £ million/month
per borough for the borough mean; capacity changes are percentage points. Panels
use different y-axis scales and answer different questions. All scenarios are
compared against matched-climate baseline land use. Estimates are deterministic;
neither capacity summary is worker-weighted. The accompanying comparison table
also retains the superseded full-model-domain capacity values to explain the
boundary correction, not as an alternative primary estimate.
