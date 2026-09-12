# Figure 4 aggregation and uncertainty decision

## Decision

Figure 4 is described as a comparison of **citywide co-benefits across
London**. Its primary results therefore use a London-wide estimand rather than
an unweighted mean of 33 borough changes.

| Outcome | Primary Figure 4 estimand | Unit | Uncertainty shown |
|---|---|---|---|
| Energy | Scenario-minus-baseline avoided cost summed once across unique building records | £ million/month, using late-2025 input-price assumptions | None; deterministic model output |
| Heavy-work productivity | Scenario-minus-baseline continuous Hothaps workability, averaged over common valid 10 m pixels | Percentage points | None; deterministic model output |
| Health | Total heat-related deaths averted using population-weighted allocation of 2021 registered mortality | Annual deaths | 95% interval from paired exposure-response draws |

The three panels do not need the same mathematical weight because their
quantities are different. They do need a consistent geographic interpretation:
each primary estimate represents an aggregate or area-wide London outcome.

## Borough sensitivity

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

## Recommended Figure 4 legend

> **Fig. 4 | Citywide co-benefits of alternative land-use scenarios under
> mid-century climate conditions.** Bars show changes in **a**, total avoided
> monthly energy costs summed across unique London building records; **b**,
> mean modeled heavy-work capacity across valid London land-area pixels; and
> **c**, total heat-related deaths averted using 2021 population and registered
> mortality, under nine land-use scenarios. Values are calculated relative to
> the baseline land-use configuration under identical mid-century climate
> conditions, isolating the modeled contribution of land-use change. Health
> error bars show 95% intervals from paired exposure-response uncertainty
> draws. Energy and productivity estimates are deterministic. Borough-level
> distributions and equal-weight borough summaries are provided in the
> Extended Data.

## Recommended nine-scenario layout

Use the same order in all three panels:

`AllBuilt, TreeRisk, TreeOpp | Green10, Target10, Green20, Target20, Green30, Target30`.

The first block retains the three original manuscript counterfactuals. A light
separator distinguishes them from the second block, where Target-versus-Green
differences are adjacent within each matched canopy budget and the 10% to 30%
progression is preserved. Keep the historical colours for AllBuilt, TreeRisk
and TreeOpp, plus one common Green colour and one common Target colour across
panels. A horizontal zero line is required because AllBuilt and TreeRisk can
produce negative co-benefits. Do not connect bars with lines, which could imply
a continuous trajectory between categorical scenarios.

The legacy AllBuilt, TreeRisk and TreeOpp results were generated with InVEST
3.14.1. They must not be combined with the revised InVEST 3.20.2 Green/Target
results. The production workflow reruns their original LULC rasters with the
same 3.20.2 model settings, inputs and post-processing used for the other six.

The three established source notebooks now create the individual panels:

- `../invest_result_zonal_viz_2_energy.Rmd` — Figure 4a;
- `../invest_result_zonal_viz_3_pd_NEW.Rmd` — Figure 4b; and
- `../health_assessment/health-modeling-output-plot-city.Rmd` — Figure 4c.

They share scenario ordering, labels and styling through
`figure4_panel_helpers.R`. `plot_figure4_citywide.R` assembles the final
three-panel figure and records a single provenance manifest.

## Copy-ready Methods text

> We quantified land-use co-benefits relative to the baseline land-use
> configuration under the same mid-century climate condition. Figure 4 reports
> London-wide estimands selected for each outcome. Avoided monthly energy cost
> was summed once across unique building records. Heavy-work capacity was
> calculated with the continuous Hothaps relationship and summarized as the
> area-weighted mean change across common valid 10 m land pixels; this is a
> modeled capacity index and is not weighted by worker location. Heat-related
> deaths averted were summed across London using population-weighted allocation
> of deaths registered in 2021 and fixed 2021 population. For sensitivity
> analysis, energy and productivity changes were also calculated separately
> within each of the 33 London boroughs and summarized with equal weight per
> borough. The citywide and equal-weight borough summaries answer different
> questions: total or area-wide benefit across London versus the response of an
> average borough. We therefore use the citywide estimands in Figure 4 and
> report borough distributions and maps separately.

## Recommended Figure 5 legend

> **Fig. 5 | Borough-scale changes in co-benefits under alternative land-use
> scenarios.** Maps show scenario-minus-baseline changes in avoided monthly
> energy cost, modeled heavy-work capacity and heat-related deaths averted for
> each London borough under Green and Target scenarios with matched 10%, 20%
> and 30% canopy budgets. Colour scales are held constant across the six
> scenarios within each outcome row. Energy and productivity changes are
> calculated relative to the matching borough baseline. Health values are
> deterministic borough sums of the population-weighted 10 m mortality-change
> rasters. Map variation represents spatial heterogeneity, not model
> uncertainty.

## Recommended Extended Data legend

> **Extended Data Fig. X | Sensitivity of energy and productivity summaries to
> geographic aggregation.** The left panels show the citywide estimands used in
> Figure 4; the right panels show unweighted means of scenario-minus-baseline
> changes across 33 London boroughs. The panels use their stated units and
> independent y-axis scales because a London total and a mean borough are not
> numerically interchangeable. Both approaches preserve the qualitative
> Target-versus-Green comparison. Health is not included because a reviewed
> borough-level uncertainty workflow is not currently available.

## Reproducible outputs

Run `plot_figure4_citywide.R` after the citywide valuation summary and reviewed
health scenario set are complete. The script writes:

- `figure4_citywide_cobenefits.{png,pdf,svg}`;
- `figure4_citywide_cobenefits_data.csv`;
- `figure4_citywide_cobenefits_manifest.json`; and
- when borough input is supplied,
  `figure4_borough_sensitivity.{png,pdf,svg}`, its source CSV, and
  `extended_data_citywide_vs_borough.{png,pdf,svg}` with its source CSV.

The manifest records the input paths, energy-summary checksum, temperature,
price-basis label, estimands and uncertainty treatment.
