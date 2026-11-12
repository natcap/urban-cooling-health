# Editorial synthesis: narrative and methods readiness

11 November 2026 · Simulated senior Nature-family journal editor perspective; no journal affiliation or actual editorial decision.

**Author status update:** further manuscript-review work and review-driven
actions are on hold until requested. Abstract revision is deferred until the
main-text sections are addressed. The synthesis below is retained as a record,
not an instruction to proceed; see the [current task status](../../../TODO.md).

## Verdict and evidence boundary

**Major revision before submission. Preserve the Results architecture; resolve headline health and temperature validity before polishing claims.** The strongest contribution is a matched-canopy allocation comparison with distributional trade-offs, not integration of three familiar models. This memo does not predict acceptance or certify novelty.

I adjudicated both reviews against the accepted-change manuscript and supplement, checking the principal claims, methods and scenario descriptions. This is not an image/code audit, simulation or full literature/reference audit. Internal audits and the coordinating reviewer's production-parameter check are attributed evidence, not independently reproduced validation.

The two review processes remain separate:

- Narrative: [Reviewer 1](reviewer_one_narrative_20261111.md) ([HTML](reviewer_one_narrative_20261111.html)) and [Reviewer 2](reviewer_two_narrative_20261111.md) ([HTML](reviewer_two_narrative_20261111.html)).
- Methods: [Reviewer 1](reviewer_one_methods_20261111.md) ([HTML](reviewer_one_methods_20261111.html)) and [Reviewer 2](reviewer_two_methods_20261111.md) ([HTML](reviewer_two_methods_20261111.html)).

No Word edits, model changes, reruns or pushes are authorized by this memo. Discussion revisions are **on hold**; comments about it below are a parked agenda. Figure 6 uncertainty refinement remains **Priority 3**, not a reason to halt the current reporting work.

## Process A — narrative audit and editorial adjudication

### A1. Consensus: sharpen the question, not the manuscript structure

Both reviewers identify the same story: **equal canopy increments can benefit different populations, and targeting one vulnerability objective need not improve every dimension.** The final Introduction instead leads with “integrate CMIP6-based climate projections” and a “transferable template.” Lead with the Green–Target question; limit transferability to the workflow. Establish novelty against the closest studies before claiming a gap.

Retain the sequence already in place: heat/canopy context → conditional cooling and co-benefits → **General greening yields uneven neighborhood health benefits** → **Targeting redistributes health benefits across vulnerability dimensions**. Do not merge the last two sections. Add Target10/20/30 to Table 1 and the main scenario overview so readers encounter the central comparison before the supplement.

### A2. Consensus: separate climate context from the controlled experiment

The Abstract's “integrated ... climate projections” and Results' “Across all future periods” imply an unestablished forecast chain. Separate Figure 1 climate context from the common-condition experiment; use conditional warm-condition language in Figures 3–7 unless verified coupling supports year/SSP claims.

The replacement Figure 3 exists. Integrate figure, caption, comparator and aggregation labels together. Its +1.11°C AllBuilt and −0.32°C Target30 are London-area-weighted contrasts, not interchangeable with borough maxima or equal-borough means. Integration does not validate exposure.

### A3. Consensus: the older-neighborhood trade-off belongs in the headline

Target Results reports older-neighborhood low-benefit shares increasing **28.7%→59.0%**, while high-SVI shares fall **45.5%→4.5%**. Add the trade-off to the Abstract without implying age-specific mortality. Retain fixed Green30 thresholds and paired neighborhoods.

SVI alignment is partly design-expected, not independent equity validation. Equal canopy controls quantity, not candidate land, cost or implementation. Describe allocation strategies, not an isolated vulnerability-weighting effect. Replace unsupported “conventional”/“typical” with “proximity-based,” and “feasible” with “spatially screened.”

### A4. Editorial emphasis: health leads; energy and work support it

Keep energy and heavy-work capacity as supporting consequences, with £/month and percentage-point units. Distinguish area-weighted capacity from actual worker output. Avoid unsupported “net,” “realized” or “conservative” benefits.

Do not finalize the annual deaths headline until Process B gates close. The present 135–496 range, 1,110 AllBuilt value, £22.4 million/month and 1.42-point gain are already incorporated in the text; the reviewers do not justify reopening them as missing copyedits. Internal consistency is different from scientific validity.

### A5. Where the reviews differ, and the decision

The narrative reviewers do not substantively disagree. Reviewer 1 emphasizes temporal mortality validity; Reviewer 2 emphasizes planning labels and literature benchmarking. Resolve health meaning before extensive prose revision.

Both recommend Discussion tightening. **Park it**: later emphasize allocation trade-offs, reduce general background, distinguish conjecture and avoid presenting Green's pattern as Target's. No Discussion rewrite now.

## Process B — methods audit and evidence gates

Reviewer 2 proposes more sensitivities; these are not a mandatory factorial analysis. Diagnose existing code, inputs and coefficient sources first. Missing documentation is not a proven model error.

| Priority / gate | Evidence and precise anchor | First action and closure criterion | Work class; rerun trigger |
|---|---|---|---|
| **1: Health estimand** | Health outcomes: “summer (July) temperature increases”; annual registered 2021 deaths; Figure 4: “annual ... deaths averted.” The exposure-to-duration bridge is not documented. | Trace the selected exposure–response estimates and implementation: temperature metric, risk window, threshold/lag, death denominator and fixed-rate assumption. Supply a worked calculation with units. Establish what the output actually estimates before retaining annual-prevention wording. | **Source/code check first.** If the existing calculation lacks a defensible temporal interpretation, author chooses a justified recalculation or a more limited standardized scenario index; the latter materially changes the health claim and is not an automatic cosmetic rescue. No arbitrary division by 12. |
| **1: Reference-risk algebra** | Health equations define scenario-minus-baseline ΔT, `RR=exp(beta×ΔT)`, `AF=(RR−1)/RR`, `ED=AF×D0`, with D0 called baseline deaths. | Name both risk states and derive the expression; inspect code signs and denominator, then hand-check warming and cooling cases. If D0 truly denotes baseline deaths and RR scenario/baseline, the direct contrast is D0(RR−1), not D0(1−1/RR). Determine intended semantics before changing anything. | **Code/estimand check**, not a diagnosed implementation bug. Recompute downstream health only if diagnosis confirms a changed formula/interpretation is needed. |
| **1: Missing-data sensitivity** | Ranking-input audit: identical valid Green30 LULC and mask, software and ancillary inputs; NoData 255→0 changes cooling on the same valid output cells. | Trace the affected input/reclassification/convolution stage and establish intended NoData semantics. Show why production handling is scientifically appropriate and add a focused invariance/edge-handling test or explicit justification of non-equivalent semantics. | **Focused numerical diagnosis.** The existing control need not be repeated. Rerun only affected outputs after a diagnosed correction; additional unchanged-input controls only if required to resolve remaining attribution. Do not extrapolate Green30 attribution to all tiers. |
| **1: Exposure meaning and UHI** | UCM Methods: Yale “average and maximum ... (2 and 5°C)”; Sup. Tables 1/3 prescribe 22/25/28°C and 2/5/5°C. Figure 1 mixes baseline product/year and future climatologies. | Separate climatic context from prescribed experiments; verify extraction, surface-versus-air metric compatibility and any existing local validation. Reconcile GCM inventory and temperature variables. A multi-model mean is not propagated uncertainty. | **Wording plus provenance check** now. If calibrated forecast claims are retained, harmonization/validation is required. Otherwise justify conditional parameters and scope the few sensitivities with material leverage; a UHI scaling identity is not empirical validation. |
| **2: Coefficient applicability** | Table 2/ref. 61 title concerns elderly populations; Health Methods uses “all ages and sexes.” | Verify the actual selected estimate, population, exposure, endpoint and source table; distinguish title-based warning from confirmed mismatch. Do not sum cause-specific estimates into all-cause. | **Source/code check.** Alternative coefficients or age-compatible data only if applicability is not defensible; not necessarily a new cooling run. |
| **2: Target/SVI reproducibility** | Methods “matched on realized added-canopy area”; Sup. Methods `UTCI_norm × SVI`; Social vulnerability drops housing yet describes four themes. | Report actual retained indicators/formula, directions, weights, missingness, geography and shared targeting/evaluation score. Import verified realized-area/rasterization/tie-breaking details from existing audit; document UTCI source and coverage. | **Reporting plus code check.** Strategy-package wording avoids requiring a common-candidate experiment. Such an experiment becomes necessary only for claims specifically isolating the priority score. |
| **2: Energy/work and input tables** | Energy Methods 2022 percentages versus Sup. Table 4 source; RH Methods leaves “mean ... or ... Tx”; Sup. Table 2 woodland shade=0. | Recover coefficient derivation/units, sector-zero meanings, footprint/floor-area convention, price year, WBGT equation and actual RH. Reconcile supplement with the exact production parameter file. | **Provenance/code check.** Correct reporting when code is sound; recompute only affected outcomes if substantive parameter/implementation errors are established. |
| **2: Reporting and release** | Figure 3 old confidence-interval label; Table 1 six scenarios versus nine; body work −6.37% versus audited −6.36 points. | Integrate existing Figure 3 package, align units/scenario inventory and freeze figure-to-input/run provenance. Distinguish spatial spread, coefficient uncertainty, sensitivity and validation. | **Wording/integration only** where existing outputs suffice. Final image and reference/claim-support checks remain separate gates. |
| **3: Figure 6 uncertainty refinement — held** | Figure 6 already states pointwise model-conditional intervals; Results already says “not spatially adjusted or causal.” | Retain descriptive scope and document current fitting settings. Keep new spatial/GAM uncertainty work on the Priority 3 list. Figure 2 significance claims also need appropriate spatial caveats. | **No elective refitting now.** Dependence-aware analysis becomes necessary if stronger inferential claims are retained. A health-estimand correction, if diagnosed, may mechanically require updating health-derived figures; that is not permission to redesign GAMs. |

### B1. Adjudication of the new input evidence

The [ranking-input audit](../figure3_ranking_input_audit.md) establishes a material controlled result, not just a hypothetical reviewer concern: NoData-only harmonization reduces Green30 cooling by **0.049786°C across borough means** (**0.076973°C London-area-weighted**). The subsequent canopy redesign restores **0.012330°C** (**0.003454°C area-weighted**). It therefore cannot be explained solely by better canopy allocation or a different aggregation mask. The exact numerical mechanism remains undiagnosed; the evidence does not prove which encoding is scientifically correct.

Historical Green30 and its same-input 3.20.2 rerun give effectively identical paired cooling. A script header naming an older release is not verified historical runtime provenance. Do not label this a demonstrated old-version-to-new-version effect, import the separate intermediate Target diagnostic, or assign a version percentage to unchanged historical Target without its control. Revised matched eligible canopy is a real improvement in experimental comparability, but does not resolve the numerical-handling gate.

The coordinating reviewer also reports a live check of the actual production biophysical file: woodland classes 1/2 use **Shade=1, Kc=0.7**, and class 100 **Shade=1, Kc=0.8**. The supplied supplement shows woodland Shade=0. Thus the presently supported concern is **table-versus-production reporting mismatch**, not evidence that the production model assigned zero shade to woodland. Tie the correction to the exact run manifest before publication. I did not independently inspect that production file.

### B2. Minimum practical revision sequence

1. Record the unresolved health temporal/reference-risk definitions and diagnose NoData handling using existing evidence. Verify UHI/exposure provenance in parallel where practical. Do not launch blanket reruns.
2. Apply reporting-only repairs and integrate the already-produced Figure 3 package for author review, keeping the quantitative health claims provisional until their gates close. Preserve the consistent revised dataset; do not mix historical temperatures with revised health outputs.
3. Have the author approve any diagnosis-driven calculation change or narrowed estimand. Recompute only dependent outcomes if necessary, then recheck all paired Green–Target quantities and threshold-derived shares.
4. Complete Target/SVI and co-benefit provenance, then finalize Abstract/Introduction/Results claims. Leave Discussion and elective Figure 6 uncertainty work held until requested.
5. Before submission, separately check rendered figures/captions, reference validity and claim support, and a frozen reproducibility package. Neither reviewer nor this synthesis has completed those gates.

**Editorial bottom line:** the appropriate immediate response is a small, evidence-led validity audit plus targeted reporting corrections—not a wholesale rewrite and not automatic rerunning of every model. The health-equity comparison is worth preserving; its strongest version reports conditional strategy differences and trade-offs without overstating forecast precision, causal isolation or real-world feasibility.

## Source snapshot and review process

Reviewed on 11 November 2026 using accepted-change text exported from:

- `London urban cooling-ms-0806.docx` in `/Users/yingjiel/Library/CloudStorage/OneDrive-Stanford/_papers/docs_london_cooling/submit/`; SHA-256 `90f937f6063af076e0bbd77b8b83679c2dedd979d364c5fa3726f7e6ee6c5bca`.
- `London urban cooling-SM.docx` in the same folder; SHA-256 `b3d2111de1fce73e6fdecae086520bae1fc291ed69ddc9064033aa3a1d053d92`.

Each of two independent reviewer agents read the complete manuscript and Supplement, formed its own assessment without reading the other review, and produced separate narrative and Methods reports. Both then checked documented project corrections; the new NoData audit was explicitly supplied as additional evidence. A third agent read all four reports and relevant source sections to adjudicate them. These are simulated academic/editorial perspectives, not external human peer review or journal decisions.

Extraction command: `pandoc INPUT.docx --track-changes=accept -t markdown -o OUTPUT.md`. No changes were accepted in Word itself. Full source-document copies are not added to the repository. The individual reports are retained unchanged as reviewer records; editorial adjudication distinguishes unverified concerns from production evidence.

This review generates no figures. Figure-producing scripts and current versioned exports remain documented in the [Figure 3 review](../figure3_climate_uhi_audit.md) and [Figures 4–5 methods index](../figure4_methods.md), rather than duplicated here.
