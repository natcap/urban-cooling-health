# Independent simulated review 1: narrative, organization and claim–evidence alignment

## Scope and recommendation

This is a simulated academic peer review for a Nature Health submission, not an assessment by journal staff. I read the full accepted-change manuscript and Supplementary Information exported to `/tmp/london_manuscript_review_20261111.md` and `/tmp/london_supplement_review_20261111.md`. I formed the concerns below before consulting `docs/manuscript/figure3_climate_uhi_audit.md` and `docs/manuscript/figure4_5_audit.md`. Those internal audits are secondary project evidence, not independently reproduced analyses. I did not inspect another review, execute models, inspect image pixels, or edit the sources. Embedded figure captions were reviewed; rendered artwork, Word layout, and tracked-change fidelity were not independently verified. No full reference-validity or claim-support audit was conducted; no external claims are certified here.

**Recommendation to the editor: major revision; not ready for submission in its present form.** The potentially publishable contribution is the equal-canopy comparison of allocation strategies, linking citywide benefits to neighborhood redistribution and an explicit older-population trade-off. The current manuscript dilutes this contribution with broad background and inconsistently presents conditional modeling as calibrated future health prediction. Several issues require analytic justification, not copyediting alone.

## Strengths

- The progression from exposure inequalities, through cooling and co-benefits, to neighborhood distribution and targeted allocation is intelligible and worth retaining.
- Matching Green and Target on added canopy is a strong starting design. The Results acknowledge that higher aggregate benefits do not improve every borough.
- Figure 7's fixed Green30 thresholds and paired LSOAs make redistribution interpretable; reporting that percentages count neighborhoods rather than people is especially useful.
- The current Figure 4/6 legends distinguish limited exposure–response uncertainty, area-weighted work capacity, fixed population, and noncausal ecological associations. These qualifications are substantive strengths, not missing items.
- The explicit result for older-population neighborhoods prevents a simplistic claim that one targeting approach improves every dimension of equity.

## Major concerns

### N1. The main contribution should be the allocation trade-off, not another demonstration that trees cool cities

**Anchor:** Introduction, “fewer translate cooling into multiple human outcomes within a unified framework”; Discussion, “three overarching insights.” **Urgency: high, before submission.**

The introduction motivates heat, nature, three endpoints, attribution and equity sequentially, but the question distinguishing this study appears late. The strongest design is not simply integration of familiar model components: it is the comparison of proximity-based versus heat-and-vulnerability-targeted allocation at the same canopy increment, with both aggregate gains and distributional losers. The paper should state this test directly in the final Introduction paragraph and identify which uncertainty it resolves. A London case study with an established cooling model needs a more precise novelty claim than a “transferable template.” Do not assert unprecedented novelty without a focused literature comparison.

**Feasible change preserving structure:** retain all main sections, shorten general heat/economic background, and end the Introduction with three linked questions: how much modeled benefit each allocation produces, who receives it, and which vulnerability dimensions trade off. State that transferability refers to the comparison workflow, not transfer of London coefficients, feasibility, or effect sizes. Bring Target into Table 1 and the main scenario description rather than leaving the pivotal intervention primarily in Supplementary Methods.

### N2. Climate context and conditional land-use experiments are narrated as one forecast

**Anchors:** Abstract, “integrated ... climate projections”; Results, “Across all future periods”; Figure 3 legend, “95% confidence intervals for model estimates”; Methods, “Climate effects were estimated by comparing future and 2021 climate conditions.” **Urgency: critical.**

Figure 1 uses a July 2021 baseline and future WorldClim climatologies, whereas Supplementary Tables 1/3 prescribe reference-temperature/UHI combinations. The narrative does not explain how those two temperature systems are linked. The UHI parameter changes from 2°C to 5°C; this is not itself an observed future climate trajectory. “Mid-century conditions” presently encourages a stronger interpretation than justified by the parameter provenance. The Figure 3 bars are also presented as model uncertainty although the supplied audit identifies geographic-summary intervals.

**Post-review audit distinction:** the Figure 3 audit already documents a completed matched-condition nine-scenario figure and proposed text; it explicitly says Word was not changed. The current Word export still has the old caption, calendar framing and 1.05/0.37°C claims. Its revised London-area-weighted contrasts are +1.11°C for AllBuilt and −0.32°C for Target30, not interchangeable with old equal-borough means. Integrating that existing package is an author-facing update, not a request to repeat its completed production work. The unresolved UHI applicability and baseline-product comparability remain substantive after the new figure is inserted.

**Feasible change:** introduce Figure 1 as climate context and Figures 3–7 as conditional land-use experiments under explicitly prescribed warm conditions. Replace outdated temperature claims together with the new figure/caption and retain a clear geographic aggregation definition. Only restore exact SSP/year forecast language if a defensible coupling is demonstrated.

### N3. The headline health quantity is more certain and temporally specific than the described methods justify

**Anchors:** Abstract, “averting 135–496 deaths ... per year”; Health outcomes, “summer (July) temperature increases” combined with “calendar year 2021” registered deaths. **Urgency: critical.**

The manuscript converts a representative warm-month temperature contrast into an annual death count without explaining the exposure duration or annualization. Readers will interpret 496 as a defensible estimate of annual preventable mortality, not an index obtained by applying a July contrast to annual registered deaths. A high-impact health paper cannot leave this temporal bridge implicit. Neither fixed-population disclosure nor coefficient uncertainty resolves it. The counterfactual mortality denominator also needs clarification (separate methods review).

**Feasible change:** resolve the estimand and temporal mapping before finalizing the Abstract. If it remains a standardized scenario calculation, state that plainly rather than describing a forecast of deaths actually prevented. Do not simply divide by 12: seasonal mortality and nonlinear heat exposure require justification. The main text should distinguish hypothetical modeled intervention effects from observed intervention effectiveness.

### N4. Equity success is partially design-defined, and the main conclusion suppresses an important trade-off

**Anchors:** Supplementary Methods, “Priority score = UTCI_norm × SVI”; Abstract, “redistributed benefits toward ... social vulnerability”; Results, older-population low-benefit share “28.7% to 59.0%”; Discussion, “reversing the pattern.” **Urgency: high.**

Improvement along the targeting score is informative implementation performance, but not independent validation that targeting delivers equity in a broader sense. Supplementary Fig. S7 already says this; the main narrative should carry the same qualification. The observed loss for older-population neighborhoods is central, not a caveat to bury: the targeting objective weights dimensions that are not spatially aligned. Moreover, Target changes the planting domain to street-based sites, not only the ranking of Green's candidates, so the comparison tests strategy packages rather than isolating the causal effect of vulnerability weighting.

**Feasible change:** add one Abstract clause acknowledging the older-area trade-off. In the existing Discussion equity section, distinguish composite-score alignment, distribution across dimensions, resident-weighted welfare, and age-specific mortality. Say that equal canopy controls quantity, not site type or cost. Present the two strategies as alternative allocation packages unless a common-candidate sensitivity is available. Do not call the score a universally preferable definition of equity.

### N5. Feasibility and realized economic benefits are overstated

**Anchors:** Abstract, “feasible planting strategies”; Table 1, “maximum plausible co-benefits”; Discussion, “measurable work productivity benefits” and “conservative ... estimate”; Supplementary Methods, 1,819,000 Target30 sites. **Urgency: high.**

Road offsets, canopy/building buffers and canopy-area matching are useful screening constraints, but do not establish practical feasibility, mature canopy survival, comparable implementation costs, or freedom from underground-service/access conflicts. An area-weighted heavy-work capacity index is not measured productivity or income. Selecting the most heat-sensitive intensity does not establish conservatism. Energy estimates are conditional modeled demand effects, not realized household savings, especially with negligible residential AC and commercial category omissions.

**Feasible change:** replace “feasible” with “spatially screened” where appropriate; distinguish mature-canopy scenarios from planting plans; delete the unsupported “conservative” label. Use “modeled heavy-work capacity” consistently. Put a compact implementation limitation in the existing limitations section. Keep stewardship proposals, but label them as policy implications beyond the tested model rather than evidence generated here.

## Minor and editorial concerns

1. **Scenario inventory:** Methods says “six LULC scenarios”; Table 1 has six, whereas Figure 4 says nine and the supplement lists nine. Add Target10/20/30 and baseline clearly. Distinguish relative canopy increases from percentage-point changes.
2. **Outcome units:** Results still says “49.6 million £” without a period, “productivity ... 6.37%,” and gains “1.42%.” Use £49.6 million/month and percentage points; the internal audit reports 6.36 points for AllBuilt. Its corrected 1.42-point maximum and 135–496 health range are already in the current Abstract and should not be flagged as absent.
3. **Temperature terminology:** reconcile “mean temperature of the warmest month,” “average maximum temperature,” July Tmax and “peak summer temperatures.” Monthly mean daily maxima are not daily heatwave peaks.
4. **Discussion balance:** retain its headings but substantially reduce repeated global heat/energy statistics. Use the space for the older-area trade-off, design dependence of targeting, temporal health estimand and model applicability. The concluding paragraph currently largely repeats the Green pattern despite the distinct Target result.
5. **Evidence language:** “identification logic” should mean model-controlled comparison, not causal identification in observed London. “Marginal gains” is ambiguous for absolute differences between discrete scenarios.
6. **Limitations:** “likely understate the full suite of benefits or trade-offs” is not a coherent direction-of-bias claim. Omitted benefits and omitted harms do not establish that net value is underestimated.
7. **Internal citation checks:** reference 5's listed title concerns benefits of nature exposure, yet it accompanies heat-vulnerability claims; reference 42 concerns acute nature-related mental-health benefits but accompanies temperature–mortality effect-size discussion. These are apparent fit problems based on the supplied bibliography, not completed source audits. Reference 61 explicitly concerns elderly outcomes but is used with all-age mortality (methods concern).
8. **Copy consistency:** remove the heading style applied to the full first Target Results paragraph, duplicate full stops, the repeated crown-size sentence in Supplementary Methods, empty headings, and incorrect LULC color-key cross-references to Supplementary Fig. 2 (a temperature plot). Standardize supplemental numbering.

## What is already addressed versus still open

The current text incorporates important Figure 4/5 audit changes: nine-scenario wording, corrected health range, monthly Abstract energy units, 1.42 percentage points, limited health uncertainty, borough ranges and the “not every borough” qualification. Do not reopen those as missing. Image adoption cannot be certified from this text-only review. Figure 3 integration, some body-text units, full Target/SVI reproducibility, temporal mortality interpretation, defensible forecast framing and the main-text equity trade-off remain open. Existing audit numbers establish internal production consistency, not empirical validity of the model chain.

## Additional unpublished audit evidence received after independent review

I subsequently read `docs/manuscript/figure3_ranking_input_audit.md` at the lead reviewer's request, without reproducing its checks. It reports that an existing controlled Green30 NoData-only change materially alters cooling on the same valid cells and can reverse the legacy Green30/Target30 ranking. Thus the manuscript must not explain changed rankings solely as better canopy allocation, geographic averaging or an InVEST upgrade. The audit separately documents revised matched eligible canopy area and source-class differences, supporting the interpretation of Green/Target as allocation packages. This is internal provenance absent from current Word, not a new independently verified scientific result. A focused numerical-handling check remains warranted; the separate methods review details it. This does not require replacing the manuscript's overall narrative structure.

## Requested revision sequence

1. Resolve the health estimand and conditional climate/UHI interpretation.
2. Integrate the already documented Figure 3 package and reconcile all scenario/units references.
3. Refocus the existing Introduction and Discussion on matched-canopy allocation and equity trade-offs, without restructuring the manuscript wholesale.
4. Conduct a separate full reference-validity/claim-support audit and rendered manuscript/figure cross-check before circulation or submission.
