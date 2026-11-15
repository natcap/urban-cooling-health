# Editor's Decision, Synthesis and Revision Plan

**Manuscript:** "Urban greening strategies reshape heat-health benefits and equity" (London)
**Journal:** Nature Health
**Date:** 11 November 2026
**Referees:** Panel A (narrative): Reviewer #1 (A1, heat-health epidemiology), Reviewer #2 (A2, urban NbS / environmental justice). Panel B (methods): Reviewer #3 (B1, urban climate / InVEST / energy and WBGT), Reviewer #4 (B2, epidemiology / biostatistics).
**All four recommendations:** Major revision. A2 and B2 each state that the paper is close to "reject, resubmission encouraged".

**Editor's decision:** **Reject in current form; resubmission invited.** The resubmission will go back to the same referees where possible.

**Comment IDs used below:** A1-M1…M9 and A1-m1…m18 (A1 §§4–5); A2-M1…M7 and A2-m1…m17; B1-M1…M10 and B1-m1…m18; B2-M1…M10 and B2-m1…m13. "M" is a major comment and "m" a minor one, numbered as in each report.

---

## 1. Decision letter to the authors

Dear Dr Li,

Thank you for submitting "Urban greening strategies reshape heat-health benefits and equity" to *Nature Health*. Four referees have assessed it. Two read the full narrative. Two concentrated on the physical-modelling and epidemiological methods. Their reports are enclosed. I have also checked the most consequential points against the manuscript and Supplementary Information myself (Section 2 of the enclosed synthesis).

All four referees and I agree that the question you ask matters and suits this journal. Cities increasingly adopt canopy targets expressed as aggregate percentages, and your paper argues that *where* trees are planted, not only how many, decides who receives the protection. Several features were singled out for praise:
- the paired comparison of proximity-based (Green) and vulnerability-targeted (Target) allocation at nominally matched canopy;
- the link to the London Environment Strategy;
- the separation of land-use effects from climate effects;
- the candour of several captions;
- the commitment to open code.

These are real strengths. A well-supported version of this paper could inform policy.

We cannot, however, consider the manuscript further in its present form. The concerns go beyond presentation. They affect the validity of the headline health quantities, and the corrections are likely to change both the absolute results and possibly some distributional conclusions. Because the work needed amounts to re-deriving the main outcome and re-running the scenario set, we are declining the manuscript in its current form. We would welcome a resubmission that addresses the points below. It would be treated as a revision and returned to the same referees where possible. We cannot guarantee acceptance, because that will depend on what the corrected analyses show.

For further consideration, the editors regard the following as essential.

1. **Re-derive the mortality estimates using accepted heat health-impact-assessment (HIA) practice.** Your Methods do not limit baseline deaths to July or the warm season. Supplementary Fig. 9 shows full-year 2021 borough death *counts* that sum to roughly London's annual total. The reported deaths averted per °C of modelled cooling (about 1,050–1,300 across scenarios) equal annual all-cause deaths × ln(1.02). So a July daytime temperature change appears to have been applied, through a threshold-free log-linear relative risk, to deaths in every month of a COVID-affected year. The resulting figures (135–496 deaths averted per year; 1,110 additional deaths under AllBuilt) are of the same order as London's entire heat-associated mortality in the record 2022 summer, and all four referees question them. Please:
   - use a season-appropriate, multi-year, non-pandemic baseline;
   - use a London-relevant temperature–mortality function that respects a minimum-mortality temperature or threshold;
   - benchmark the baseline heat burden against independent UKHSA/ONS and published London estimates;
   - report greening effects as a share of that burden.

2. **Correct and validate the temperature model, and make its key assumption visible.**
   - Supplementary Table 2 gives deciduous and coniferous woodland a shade value of 0, while "Tree Cover" has 1. The Green scenarios convert woodland, among other classes, to Tree Cover. Please correct the parameter table and re-run.
   - The UHI magnitude is 2 °C for 2021 and 5 °C for all future runs. This is a change from the "average" to the "maximum" value of what appears to be a satellite surface-UHI product. Because land-use ΔT in the Urban Cooling Model scales linearly with UHI_max, every land-use result depends on this choice. Please justify UHI_max with air-temperature evidence and apply it consistently.
   - Validate the baseline temperature field against London observations.
   - Run a sensitivity analysis on the main parameters, and show whether the Target > Green ranking and the sign of the equity gradients survive it.

3. **Base the equity analysis on susceptibility and keep it out of circular evaluation.**
   - At present, borough-level crude all-age mortality is allocated to pixels in proportion to population, so within a borough, differences between LSOAs reflect only ΔT. Please use age-specific baseline mortality and LSOA age structure.
   - Targeting ranks sites on SVI, and you then evaluate its success on SVI. Please evaluate equity on indicators not used in targeting, use scale-invariant and resident-weighted distributional metrics, and add control allocations (at least heat-only and random street-tree allocation at equal canopy area). These will show *why* Target outperforms Green.
   - The finding that targeting roughly doubles the low-benefit share among neighbourhoods with the most residents aged ≥75 (28.7% → 59.0%) belongs in the Abstract and Discussion once re-estimated. The current concluding paragraph implies the opposite.

4. **Show that Green and Target are matched.** Add the Target scenarios to Table 1. Report the realised added canopy (km²), tree or pixel counts and from-class → to-class transitions for every scenario.

5. **Make the manuscript internally consistent and purge legacy material.** The submission file contains duplicate and superseded embedded images, including uncaptioned panels after Fig. 7 in the Methods region, an older Green-only version of Fig. 3b, and an older Extended Data Fig. 1. Some text values appear to come from those versions; for example, the Abstract's "0.37 °C" matches a superseded panel, while the final Fig. 3b shows −0.34 °C (Green30) and −0.38 °C (Target30). Please regenerate every figure from a single final pipeline run, remove all legacy images, and check every number in the Abstract, text, tables and captions against the final figures. Please also deposit scenario rasters, run configurations and LSOA-level outputs with a DOI.

We also ask you to temper causal language ("deaths averted", "preventable"), to label intervals as covering exposure–response uncertainty only, and to give health more weight than the energy and labour co-benefits. These two modules need documentation and plausibility checks; it would be acceptable to move them to Extended Data. The enclosed revision plan sets out the points in priority order, says which referee comments each resolves, and includes a template for a point-by-point response. Please use it.

We recognise that this is a substantial request. The referees' views were consistent: the design and question are valuable, and a corrected paper that rests on the *relative* and *distributional* findings, with appropriately modest absolute estimates, could still make a strong contribution. We look forward to seeing a revised version should you choose to resubmit.

Yours sincerely,

[Handling Editor]
*Nature Health*

---

## 2. Editor's synthesis of referee consensus

**Verification key:**
- **Confirmed:** checked directly against ms.txt, sm.txt or the embedded figures.
- **Partly confirmed:** the text or figure supports the claim, but its magnitude or mechanism depends on information the authors have not reported.
- **Not confirmed:** I could not verify the claim from the materials provided.

### 2a. Editor spot-checks of the most consequential claims

| # | Referee claim | Raised by | What I checked | Status |
|---|---|---|---|---|
| V1 | Annual deaths are computed by applying a July ΔT to full-year all-cause deaths | A1-M1, A2-M5, B1-M6, B2-M1 | Methods: "excess mortality attributable to summer (July) temperature increases"; D₀ = m × pop; "deaths registered in 2021"; "calendar year 2021". The text never restricts D₀ to July or summer. Fig. 4c axis reads "Annual deaths". Sup. Fig. 9 legends are borough **counts** of 0–3,000 (all causes), so 33 boroughs sum to roughly London's annual total. Deaths per °C of borough-mean ΔT are consistent across scenarios: AllBuilt 1,110/1.05 ≈ 1,057; Green30 387/0.34 ≈ 1,138; Target30 496/0.38 ≈ 1,305. Each implies D₀ ≈ 53,000–66,000 with β = ln 1.02. July deaths alone (about 1/12 of that) would give roughly 90–110 per °C. | **Confirmed** by arithmetic and figure inspection, although the text never states it in so many words. Authors must state D₀ explicitly. The higher ratio for Target is consistent with population-weighted ΔT exceeding the unweighted borough mean (B1-M10c). |
| V2 | UHI_max is 2 °C for 2021 and 5 °C for future runs, a switch from "average" to "maximum" | A1-M5, A2-M6, B1-M2, B2-M3 | Methods: "average and maximum magnitude of the UHI effect … (2 and 5 °C, respectively)". Sup. Table 3: 2021 = 2, 2050 = 5, 2070 = 5. Fig. 3a caption: settings 22/2, 22/5, 25/5, 28/5. | **Confirmed.** |
| V2b | Land-use ΔT is invariant to T_ref and scales linearly with UHI_max | B1-M2, B2-M3 | Fig. 3a: AllBuilt minus Baseline is about +1.05 °C in the 2030, 2050 and 2070 settings alike (≈3.2−2.17, 6.2−5.17, 9.2−8.17). The baseline shift 2021→"2030" (≈+2.2 °C) arises at a constant 22 °C T_ref, i.e. purely from UHI 2→5. | **Confirmed** for T_ref invariance. The linear scaling follows from the UCM's documented formulation, which I did not re-run. |
| V2c | The Yale UHI product is a MODIS surface-UHI (LST) product, not an air-temperature one | B1-M2a | The manuscript cites only a URL (yceo…/uhimap). | **Not confirmed from the manuscript.** It is consistent with my understanding of the product. The authors must state the product, variable and statistic. |
| V3 | Woodland shade = 0 vs Tree Cover = 1, and Green converts woodland to Tree Cover (a relabelling artefact) | A1-M4, B1-M3 | Sup. Table 2: deciduous and coniferous woodland Shade = 0, Kc 0.7; Tree Cover (100) Shade = 1, Kc 0.8. Methods (Green10): "conversion of selected grassland, woodland, suburban and urban categories to tree cover". | **Confirmed** for parameters and text. **Partly confirmed** for impact: the share of Green pixels drawn from woodland is not reported, so the size of the artefact is unknown. *Editor's note:* the same error biases AllBuilt *downwards*, because woodland→Urban produces little modelled warming, and it makes London's densest canopy behave almost like grassland in the baseline. |
| V4 | GCM list mismatch | A1-M5, A2-M6, B1-M5b, B2-m6 | Methods: EC-Earth3-Veg, MRI-ESM2-0, IPSL-CM6A-LR. Sup. Fig. 12 caption: ACCESS-CM2, IPSL-CM6A-LR, MRI-ESM2-0. Sup. Fig. 2 also shows SSP245, which the Methods do not mention. | **Confirmed.** |
| V5 | Abstract "up to 0.37 °C" vs Fig. 3b −0.34 (Green30) / −0.38 (Target30) | A1-M7, A2-M6, B1-M10d | Abstract "by up to 0.37 °C"; Results "−0.37 °C on average across boroughs". Final Fig. 3b (image13): Green30 −0.34, Target30 −0.38. A superseded Green-only Fig. 3b (image12) shows Green30 −0.37 and Green10 −0.16 (final −0.12). | **Confirmed.** The 0.37 °C value comes from the superseded panel, and "up to" vs "on average" is also inconsistent. |
| V6 | Table 1 omits the Target scenarios | A1-M4/m1, A2-M6, B1-M4, B2-M6c | Table 1 lists six scenarios. Methods say "we developed six LULC scenarios". Fig. 4 caption says "Nine scenarios (Table 1)". Sup. Table 1 lists S0–S9. | **Confirmed.** |
| V7 | Over-75 low-benefit share 28.7% → 59.0%, omitted from the Abstract and contradicted in the conclusion | A1-M6, A2-M3, B2-M5 | Results and Fig. 7d (image31) show 28.7% → 59.0%. The Abstract does not mention it. The concluding paragraph says "stronger mortality reductions in areas with more older adults", which is true only under Green. | **Confirmed.** |
| V8 | The 45.5% → 4.5% headline is circular | A1-M4, A2-M1, B2-M6 | SI step 3: "Priority score = UTCI_norm × SVI". Sup. Fig. S7 caption: "SVI is related to the targeting design and is not an independent validation measure". Fig. 7 caption: Green30 thresholds held fixed for Target30. | **Confirmed** (design fact). The level-shift confound is confirmed in principle: Target30's total is 28% higher (496 vs 387). |
| V9 | Fig. 1 baseline mixes CHELSA single-year July 2021 with WorldClim 20-year climatologies, and the Intro says ≈2.7 °C | A1-M5, A2-M6, B1-M5a, B2-m6 | Fig. 1 caption: "CHELSA for July 2021 (baseline year) and WorldClim for future periods". Intro: "≈2.7 °C by the 2050s". Fig. 1a means: 21.9 → 25.7 / 26.1 (2050). Methods: 25.5 / 25.9 °C. | **Confirmed.** Two different values are given for the same 2050 quantity. |
| V10 | Fig. 1b insets are identical across 2050 and 2070 | A1-m6, A2-M6 | Final Fig. 1 (image5): both insets show counts 2, 4, 7, 10, 10 on different axes. | **Partly confirmed.** The counts are identical. The caption says bins are set separately per inset, so identical counts *could* result from the binning rule, but that rule is not given and the result does not obviously match "13 of 33 > 4 °C". Authors to check. |
| V11 | Sup. Fig. 9 is labelled "rates" but shows counts | A1-M1d, A2-m14, B2-M4 | Sup. Fig. 9 legends read "number". There is also an undescribed Y10–Y34 panel. | **Confirmed.** |
| V12 | Energy coefficients imply ≈180 GWh per °C per month (≈£50 M) | B1-M7b, A2-M5 | Sup. Table 4: 9.89×12.24 M + 9.89×0.81 M + 30.8×1.54 M + 19.19×0.18 M + 0.01×222 M ≈ 182 GWh per °C per month. × ~£0.27–0.28 ≈ £50 M. This matches AllBuilt's £49.6 M for 1.05 °C. Offices (51% of UK cooling, per the text) are not assigned a coefficient: "Industry and Business" and "Mixed Use" are 0. | **Confirmed** (arithmetic). I did not check plausibility against DESNZ data; the authors should. |
| V13 | WBGT formula undocumented; fixed-RH approach overstates dWBGT/dT | B1-M8 | No WBGT equation in Methods or SI. RH is a single scalar per period (Sup. Table 3). | **Confirmed** that it is undocumented. B1's quantitative claim (up to ~75% overstatement) is **not confirmed**, because it depends on the unreported formula. |
| V14 | Circulatory and respiratory RRs come from elderly-only studies but are applied to all ages; the mental-disorder RR comes from a review that pools morbidity | A1-M1, B2-M2 | Table 2 gives no source population, age range, metric or lag. | **Not confirmed** from the manuscript (I did not check the cited sources). The omission of this information from Table 2 is confirmed. |
| V15 | 4,835 is the 2011 LSOA count, but the indicators come from the 2021 Census | B2-1(g), M7 | n = 4,835 used throughout; Sup. Fig. 12 uses 4,831. | **Partly confirmed.** The count is consistent with 2011 boundaries; no harmonisation is described. |
| V16 | Legacy or duplicate figures embedded in the file | A1 §7, A2 §7, B1-M10d, B2-m1/m13 | image12 (old Fig. 3b, Green-only, −0.37, with error bars); image44 (old ED Fig. 1, Green-only: 161/377/−1,078) next to image45 (final 135/387/−1,110); image46 (ED Fig. 2: no Target scenarios, raw variable names, no units); image32–34 (uncaptioned panels in the Methods region after Fig. 7, using a different metric). | **Confirmed.** See the editor's caution in Section 3 (D7). |

### 2b. Consensus issue table

| # | Issue | A1 | A2 | B1 | B2 | Editor verification | Severity |
|---|---|:-:|:-:|:-:|:-:|---|---|
| 1 | July ΔT applied to annual, COVID-year, all-cause deaths with a threshold-free global RR; implausible magnitudes; no benchmark | M1 | M5 | M6 | M1, M2, M4 | Confirmed (V1, V11) | **Essential** |
| 2 | ERF metric, lag and threshold undocumented; no MMT; cause-specific RRs mismatched by age and outcome | M1, m14 | m10 | – | M2, m8 | Partly (V14) | **Essential** |
| 3 | UHI_max 2→5 °C switch; possibly a surface-UHI source; all land-use effects scale with it | M5 | M6 | M2 | M3 | Confirmed (V2, V2b); source partly (V2c) | **Essential** |
| 4 | Woodland shade = 0; woodland→Tree Cover relabelling; canopy over Urban pixels not reclassified | M4 | – | M3 | – | Confirmed (V3) | **Essential** |
| 5 | No UCM validation against London observations | M2 | §6 | M1 | M3b | Confirmed (absent) | **Essential** |
| 6 | Uncertainty covers β only; no sensitivity on UCM, UHI, ERF, baseline or GCM; rankings carry no uncertainty | M2 | §6 | M9 | M9 | Confirmed (Methods statement) | **Essential** |
| 7 | Borough crude-rate allocation means within-borough benefit = ΔT only; no age-specific susceptibility | M3 | M3 | M10a | M5 | Confirmed (Methods, Fig. 6 caption) | **Essential** |
| 8 | Circular SVI targeting evaluation; level-shift confound from fixed Green30 thresholds; losers hidden | M4 | M1 | – | M6 | Confirmed (V8) | **Essential** |
| 9 | Target vs Green differ in more than allocation (street pixels from Urban class, UTCI siting, typology); mechanism not decomposed; no heat-only or random controls | M4 | M1d, M4 | M4 | M6d | Confirmed (SI design); mechanism untested | **Essential** |
| 10 | Matching on added canopy asserted but not shown; Target missing from Table 1 | M4, m1 | M2a, M6 | M4 | M6c | Confirmed (V6) | **Essential** |
| 11 | ≥75 trade-off omitted from Abstract, contradicted in conclusion; SVI includes a negatively correlated age term | M6 | M3 | – | M5, M7 | Confirmed (V7; Sup. Fig. 10: ≥75 r = −0.45 with income, −0.40 with ethnic share) | **Essential** (reporting); re-estimation under item 7 |
| 12 | Abstract and text numbers inconsistent with final figures; legacy images | M7, §6 | M6 | M10d | m1, m13 | Confirmed (V5, V16) | **Essential** |
| 13 | Causal and certainty language ("averted", "preventable", "±1 °C", "conservative") | M2, M9 | m1, m5, m6 | M8f | M10 | Confirmed (text) | Important |
| 14 | Climate baseline mismatch (CHELSA vs WorldClim); 2.7 vs 3.9–4.2 °C; 25.5 vs 25.7 °C; GCM list; Fig. 1 not linked to the health chain | M5, m4, m5 | M6 | M5 | m6 | Confirmed (V4, V9) | Important |
| 15 | SVI construction under-specified (formula, weights, z vs percentile, "tract-level", US instrument, boundary vintage) | m8 | M3d | M4f, m13 | M7 | Confirmed (absent); vintage partly (V15) | Important |
| 16 | Feasibility of Green30/Target30 (≈1.8 M street sites), instant maturity, "feasible"/"conventional" wording | M4 | M2, m2 | – | – | Confirmed (SI counts; Abstract wording) | Important |
| 17 | Energy module: undocumented coefficients, implausible sensitivity, offices missing, footprint area not floor area, coverage gaps | m12 | M5 | M7 | – | Confirmed (V12) | Important |
| 18 | WBGT and work-capacity module: formula missing, fixed RH, Jensen bias, area- not worker-weighted, "conservative" wrong | M8, m13 | M5 | M8 | – | Confirmed (V13) | Important |
| 19 | GAMs: univariable, unweighted, no spatial dependence, no diagnostics; CIs on deterministic output | m10 | m3 | – | M8 | Confirmed (captions) | Important |
| 20 | Baseline inequality (Fig. 2): 0.17 °C at ~1 km; non-monotonic canopy; "compounding" overstated; pseudo-replication | m7 | M7 | M5e, m14 | m5 | Confirmed (Fig. 2 text) | Important |
| 21 | Night-time and indoor exposure not modelled | M8 | – | M6, m7 | m9 | Confirmed (Methods) | Important (limitation and, ideally, a sensitivity analysis) |
| 22 | Health framing thin; co-benefits bolted on; missing HIA-greening literature (e.g. Iungman et al. 2023) | M8 | §2, M5 | M1iv | – | Confirmed (absent) | Important |
| 23 | Borough maps show totals, not per-capita values | §7 | m4 | – | m10 | Confirmed (Fig. 5 caption) | Minor/presentation |
| 24 | Data and code: rasters "upon publication"; UTCI source unstated | m16 | m17 | §6 | 1(h) | Confirmed | Important (reproducibility) |
| 25 | Citation errors, duplicates (refs 45/53), typos, spelling, "Stanford InVEST" | m14, m17 | m11 | m15, m16 | m12 | Typos and "Stanford InVEST" confirmed. The reference list was not in the extracted text, so duplicates and citation accuracy were not confirmed. | Minor |
| 26 | Gentrification and displacement; London-specific planning guidance | – | M2e, m7 | – | – | Confirmed (absent) | Important (Discussion) |
| 27 | Competing-interest transparency (authors develop InVEST) | m18 | – | – | – | Confirmed (affiliations) | Minor |

---

## 3. Points of disagreement among referees and editor's adjudication

**D1. How severe is the problem, and what is the decision?** A1 and B1 recommend major revision. A2 and B2 say major revision but describe the paper as close to "reject, resubmission encouraged".

*Adjudication:* I side with the more cautious pair on the *decision category*. I share the view of all four that the paper can be rescued. The headline health quantity is very likely wrong by a large factor (V1). Correcting it, together with the UHI and woodland-parameter fixes and the new control scenarios, means re-running essentially the whole results chain, and the conclusions after that are uncertain. A resubmission invitation reflects this honestly without closing the door.

**D2. Is the ≥75 trade-off a real finding (A1, A2) or an artefact (B2)?**
- A1 and A2 treat the 28.7% → 59.0% shift as a first-order health finding that the authors buried.
- B2 regards it as "probably an artefact of crude, borough-level mortality allocation".

*Adjudication:* both are partly right. As a statement about *where modelled cooling goes*, the shift is real: Target places trees away from older, whiter, outer neighbourhoods. As a statement about *mortality benefit for older people*, it is undetermined, because baseline risk does not vary with age within a borough. Age-specific baseline rates could make the trade-off larger or smaller. High-≥75 LSOAs would gain expected deaths under every scenario, which raises their benefit per resident. Whether that closes the Target–Green gap depends on how ΔT is distributed spatially. The authors must (i) report the finding prominently now, as an exposure-allocation result, and (ii) re-estimate it with age-specific mortality before drawing policy conclusions. Neither suppressing it nor dismissing it is acceptable.

**D3. What does UHI_max = 5 °C mean (B1, B2)?**
- B1 and B2 both note that using the 2021 value (2 °C) would shrink all land-use effects by about 60%.

*Adjudication:* I partly down-weight the implication that 2 °C is the "right" value. In InVEST, UHI_max is defined as the *maximum* urban–rural difference, so a maximum statistic is conceptually appropriate. The defect is (a) the *inconsistency* between baseline and future, which manufactures the "2030" climate step in Fig. 3a, and (b) the likelihood that the source is a *surface* (LST) UHI rather than an air-temperature UHI. What I require: an air-temperature-based UHI_max for London summer afternoons, applied consistently, with results shown across a range of values. I do not require the authors to adopt 2 °C specifically.

**D4. Should the authors expand the climate ensemble (B1: ≥10 GCMs or UKCP18) or demote Fig. 1 (A1, A2)?**

*Adjudication:* the climate rasters do not enter the UCM or the health chain (V2, Sup. Table 3), so a larger ensemble would not change any health or equity result. I therefore down-weight B1-M5c to *Important*, conditional on the authors' choice:
- **(a)** if Fig. 1 remains context only, move it to Extended Data and fix the baseline mismatch with a matched historical climatology; or
- **(b)** if the authors link T_ref and UHI to the projections and show how health results scale, the ensemble and UKCP18 requests become essential.

Option (a) is acceptable.

**D5. Co-benefits: integrate (A2 option) or demote (A1, A2 option)? Fully re-model energy and WBGT (B1)?**

*Adjudication:* for *Nature Health*, health must be the centre of the paper. The authors may move energy and work capacity to Extended Data with full documentation, plausibility benchmarks and caveats. In that case, B1's requests for floor-area energy modelling, hourly WBGT and worker weighting become *Important* rather than essential. If the co-benefits stay in the Abstract, B1-M7 and B1-M8 must be addressed in full. In either case, the "conservative" claim for heavy work must be corrected.

**D6. Feasibility, cost, maturation and gentrification (A2-M2).**
- A2 asks for constraint layers, growth trajectories, cost per death averted, a domestic-AC scenario and a gentrification discussion.

*Adjudication:* I down-weight most of this to *Important/optional*. Required:
- absolute canopy and tree counts compared with London's current stock;
- reframing Green30/Target30 as stylised upper bounds, with Green10/Target10 as the policy-relevant comparison;
- a 50%-crown-attainment sensitivity analysis or an equivalent statement;
- a Discussion paragraph on green gentrification and displacement.

A full cost-effectiveness analysis and a domestic-AC scenario are welcome but not required.

**D7. "Outdated figure versions."** All four referees cite superseded values, for example Green30 −0.37 °C, deaths 377/161/−1,078, and work +2.19 pp.

*Editor's caution:* the submitted .docx contains duplicate and legacy embedded images, including uncaptioned panels after Fig. 7 in the Methods region (image32–34) and earlier versions of Fig. 3b and Extended Data Fig. 1 (V16). Some referee comments about figure numbers may therefore refer to stale panels rather than to the final figures. I treat discrepancies tied *only* to legacy images as a file-hygiene problem, not an analytical error. The exception is the Abstract's 0.37 °C, which demonstrably propagated into the text. In the same spirit, I could not tell from the files whether A1's remarks about "UCM error bars dropped from earlier versions" describe a deliberate change. The authors should say so.

**D8. Recasting the GAMs as descriptive (B2-M8a) or spatial inferential models (B2-M8b).**

*Adjudication:* the simpler route is acceptable. Present the GAMs as descriptive smoothers of deterministic model output, remove significance language, and add scale-invariant distributional statistics (concentration indices, benefit shares by quintile, resident-weighted). Spatial or multivariable GAMs are optional. If inferential language is retained, they become required.

**D9. Novelty (A2) vs significance (A1).**
- A2 argues that "allocation matters" is not new.
- A1 sees the Green-versus-Target design as original.

*Adjudication:* both are right. The novelty lies in the *matched-allocation experiment with a decomposition of why allocation changes efficiency and distribution*, positioned against Iungman et al. (2023), Tree Equity-type frameworks and the optimisation literature. That decomposition currently does not exist. This is why the control scenarios (Tier 1, item E6) are essential rather than optional.

**D10. Points the editor down-weights or does not endorse as stated.**
- **A1-M1 and B2-M1 expected effect sizes.** A1 suggests "up to an order of magnitude"; B2 suggests about 12-fold if restricted to July. I endorse the direction but not a specific factor. The corrected size depends on the ERF, the threshold and the season definition. The authors should not simply divide by 12.
- **B1-M3c (albedo near-uniform).** This is a valid point, but it is a sensitivity item, not an error.
- **A1-m18 (competing interests).** This is a transparency note, not a conflict. A one-line statement suffices.
- **B2-m11 (Monte Carlo convergence).** Already adequate; report the seed only.

---

## 4. Prioritised revision plan for the authors

### Tier 1: Essential (the paper cannot proceed without these)

**E1. Rebuild the heat-mortality HIA.**
- *Problem:* D₀ appears to be full-year 2021 all-cause deaths, which include COVID-19 deaths. It is multiplied by an AF derived from a July daytime ΔT using a threshold-free global log-linear RR (V1, V11).
- *Do:*
  - **(a)** State D₀ explicitly.
  - **(b)** Use warm-season (e.g. June–September) or, preferably, daily deaths averaged over several non-pandemic years (e.g. 2015–2019 and/or 2022–2024, excluding U07).
  - **(c)** Apply ΔT as a perturbation to an observed or climate-perturbed daily temperature series (e.g. HadUK-Grid or UKCP18). Use a London-specific non-linear temperature–mortality function (DLNM-type, cumulative lag) that respects the MMT, with the source verified.
  - **(d)** Report results under at least three ERF specifications: current global log-linear; London non-linear; London heat-only slope.
  - **(e)** Complete Table 2 with source population, age range, metric, lag, threshold, and mortality vs morbidity for each RR. Restrict cause-specific D₀ to the age groups the RRs were estimated for, or drop the cause-specific outcomes.
  - **(f)** Benchmark the modelled baseline heat-attributable burden against UKHSA/ONS London estimates and published London UHI-attribution work (ref. 29), and against Iungman et al. 2023 for cooling per unit canopy and deaths per city. Report greening effects as a percentage of that burden.
  - **(g)** Relabel the outputs, e.g. "modelled warm-season heat-attributable deaths", and state the counterfactual and the assumption of fully grown canopy.
  - **(h)** Correct Sup. Fig. 9 (counts vs rates) and describe the Y10–Y34 panel or remove it.
- *Expected effect:* absolute deaths averted, AllBuilt additional deaths, per-million equity metrics and the Fig. 7 thresholds all fall substantially, probably by a large factor ([X]-fold). Relative rankings may survive; that remains to be shown.
- *Resolves:* A1-M1, A1-M2 (partly), A1-m2, A1-m14 (RR citations); A2-M5a, A2-m10; B1-M6, B1-m8, B1 §1(h); B2-M1, B2-M2, B2-M4a/d, B2-M10, B2-m2, B2-m8.

**E2. Correct the UCM parameterisation and make UHI_max explicit.**
- *Problem:*
  - Woodland Shade = 0 (V3).
  - UHI_max switches from 2 to 5 °C and appears to come from a surface-UHI source (V2, V2c).
  - All land-use ΔT scales with UHI_max and is invariant to T_ref (V2b).
  - Canopy over Urban pixels is not reclassified in the baseline.
- *Do:*
  - Assign shade values that reflect canopy fraction per class.
  - Decide whether existing canopy over Urban pixels is reclassified, apply the same rule to baseline and scenarios, and re-run all ten LULC configurations.
  - Derive UHI_max from London summer-afternoon *air-temperature* observations (urban minus rural MIDAS or citizen stations), apply it consistently, and present land-use effects across UHI_max of e.g. 1, 2, 3 and 5 °C, or per °C of UHI_max.
  - Relabel or remove the "2030" setting in Fig. 3a.
  - Justify T_ref from genuinely rural stations.
  - State explicitly that land-use magnitudes are conditional on UHI_max.
  - Report a from-class × to-class transition matrix (pixels, km²) for every scenario.
- *Expected effect:* Green ΔT may fall if part of it came from woodland relabelling. AllBuilt ΔT may rise. All absolute ΔT values rescale with the chosen UHI_max. The Green–Target contrast may change.
- *Resolves:* A1-M4 (typology), A1-M5iii, A1 Fig. 3 comment; A2-M6 (Fig. 3a row), A2-m13; B1-M2, B1-M3, B1-M5d, B1-m1, B1-m6; B2-M3.

**E3. Validate the UCM and run a sensitivity and uncertainty analysis on the physical chain.**
- *Do:*
  - Validate the baseline T_air field against July 2021/2022 station and citizen-station daytime Tmax (RMSE, bias, spatial rank correlation), and secondarily against Landsat/ECOSTRESS LST.
  - Calibrate or bound UHI_max, the blending and cooling distances, and the CC weights.
  - Run Morris screening or one-at-a-time ranges over UHI_max, the distances, CC weights, Kc, shade, crown size and scenario-generator settings.
  - Combine with E1 in a joint analysis over ERF × UHI_max × baseline year, with a variance decomposition.
  - Report paired Target − Green intervals, and show whether (i) the Target > Green ranking and (ii) the sign of each equity gradient survive.
- *Expected effect:* honest uncertainty ranges replace the β-only intervals. The decision-relevant inferences are either supported or weakened.
- *Resolves:* A1-M2, A1-M9 ("strengthens inference"); A2 §6 (missing sensitivity); B1-M1, B1-M9; B2-M3b, B2-M9, B2-M8e.

**E4. Base the equity analysis on susceptibility.**
- *Do:*
  - Compute LSOA expected deaths by applying borough (or MSOA) age-specific rates to LSOA age-specific population (indirect standardisation), with age-specific RRs where available.
  - Decompose the per-million benefit into population-weighted ΔT (exposure reduction) and the baseline-rate component, and report both.
  - Separate within-borough from between-borough gradients (borough fixed or random effects).
  - Re-estimate the ≥75 result under these specifications.
  - Clarify the WorldPop vs Census denominator and the downscaling grid.
  - Optionally, add a sensitivity analysis using ONS 2050 population projections, kept separate from the land-use contrast.
- *Expected effect:* the gradients in Figs 6–7 may change in magnitude and possibly in sign, especially for age. The paper will be able to distinguish *exposure-reduction equity* from *health-benefit equity*.
- *Resolves:* A1-M3, A1-M6; A2-M3a; B1-M10a/b; B2-M4b/c, B2-M5, B2-m3, B2-m4.

**E5. Make the targeting evaluation non-circular and scale-invariant.**
- *Do:*
  - Evaluate distributional outcomes on indicators *not* in the priority score: IMD domains not used in the SVI, health deprivation, long-term illness or disability, overcrowding, social housing, age-standardised baseline mortality, and the GLA Climate Risk index if it was not an input.
  - Add resident-weighted concentration indices or curves and benefit shares by quintile or decile.
  - Use within-scenario tertiles, or compare at matched total benefit, alongside the fixed Green30 thresholds.
  - Report who loses benefit (low-SVI quintiles; outer boroughs).
  - Demote 45.5% → 4.5% from the Abstract, or pair it explicitly with non-circular metrics and label it as a design check.
- *Expected effect:* the reversal may be smaller on independent indicators, which would make it more credible.
- *Resolves:* A1-M4b/d/e, A1-m11; A2-M1; B2-M6a/b/e, B2-m7.

**E6. Decompose why Target outperforms Green, using control allocations at equal added canopy.**
- *Do:* add at least
  - **(i)** heat-only targeting (UTCI only, no SVI) on the same street-point pool;
  - **(ii)** random allocation on the same street-point pool;
  - **(iii)** optionally, population-density or canopy-deficit targeting and an efficiency-optimal allocation.
- Show marginal cooling per added hectare by source class and by distance to existing canopy, and place the strategies on an efficiency–equity plot. Document the UTCI source, date, model and extent, and confirm coverage of all 33 boroughs.
- *Expected effect:* this identifies whether Target's advantage comes from street-pixel source class (Urban, low CC), heat siting, population density or SVI weighting. It sets the paper's actual novelty claim.
- *Resolves:* A1-M4c, A1 §7 Fig. 4; A2-M1d, A2-M4; B1-M4d/e; B2-M6d.

**E7. Demonstrate canopy matching; complete Table 1.**
- *Do:* add Target10/20/30 to Table 1. For every scenario, report realised added canopy (km², relative %, absolute %), pixel and tree counts, crown-size assumption, source-class composition and allocation rule. Reconcile "six" vs "nine" scenarios. Explain Target10 −0.14 vs Green10 −0.12 °C. Compare tree counts with London's existing street-tree and total tree stock.
- *Resolves:* A1-M4a, A1-m1, A1-m15; A2-M2a, A2-M6; B1-M4a–c, B1 §1(e); B2-M6c.

**E8. Report the ≥75 trade-off honestly and fix contradictory conclusions.**
- *Do:* state the (re-estimated) ≥75 result in the Abstract and conclusion. Remove "stronger mortality reductions in areas with more older adults" as a general statement. Explain how the SVI combines negatively correlated components. Test a variant priority score that includes age-weighted heat-mortality risk. Discuss explicitly the conflict between deprivation-based and mortality-risk-based equity objectives.
- *Resolves:* A1-M6; A2-M3b/c; B2-M5d, B2-M7b.

**E9. Ensure internal consistency, purge legacy material, and deposit data.**
- *Do:*
  - Remove every legacy or duplicate embedded image (old Fig. 3b, old ED Fig. 1, ED Fig. 2 unless updated, the uncaptioned Methods-region panels).
  - Regenerate all figures from one tagged pipeline run, and state the commit hash.
  - Reconcile every number using the checklist in Tier 3.
  - Deposit scenario LULC rasters, UCM outputs, run configurations and LSOA-level results with a DOI at resubmission. GiGL-derived products may be provided as non-identifying derived rasters.
- *Resolves:* A1-M7, A1 §6–7, A1-m16; A2-M6, A2 §7; B1-M10d, B1 §6; B2-m1, B2-m13, B2 confidential note.

### Tier 2: Important

| ID | Item | What to do | Resolves |
|---|---|---|---|
| I1 | Language and causal framing | Use "modelled", "estimated", "potentially avertable". Label intervals "exposure–response-only" until E3 is done. Replace "±1 °C" with the asymmetric values. Remove "conservative" for heavy work. Remove "feasible" and "conventional" unless supported. Soften "transferable template", "practical framework for bundled strategies" and "strengthens inference". Rebalance "likely understate". | A1-M2, A1-M9; A2-m1, A2-m2, A2-m5, A2-m6, A2-m8; B1-M8f; B2-M10 |
| I2 | Climate characterisation | Compute deltas against the matched WorldClim historical baseline for both SSPs. Reconcile 2.7 / 3.8–4.2 / 25.5 vs 25.7 °C. Use one GCM list throughout and explain SSP245 in Sup. Fig. 2. Harmonise the metric name (July mean Tmax). Move Fig. 1 to Extended Data unless the climate is linked to the health chain (D4). State which SSP's ET₀ and TreeRisk feed Figs 3b–7. | A1-M5, A1-m4, A1-m5, A1-m6; A2-M6; B1-M5, B1-m5, B1-m12; B2-m6 |
| I3 | SVI specification | Give the exact formula, transform, weights and signs. Fix "tract-level". Explain z vs percentile. Run with and without age components, and with equal or PCA weights. Report correlation with IMD 2019 and the GLA Climate Risk index, or show results with a UK-native index. Document the 2011/2021 boundary harmonisation and 4,835 vs 4,831. | A1-m8, A1-m9; A2-M3d; B1-M4f, B1-m13; B2-M7 |
| I4 | GAM and statistical reporting | Present the GAMs as descriptive smoothers (D8), or fit spatial, population-weighted, multivariable models. Report k, edf, deviance explained and diagnostics. Give block-bootstrap CIs for 90th–10th contrasts and report them as relative differences. Note the U-shaped income curve. Overlay Green30 and Target30. | A1-m10, A1 Fig. 6; A2-m3, A2 Fig. 6; B2-M8 |
| I5 | Baseline inequality (Fig. 2) | Recompute with the validated UCM baseline field or the 1 m UTCI layer. Report resident-weighted effect sizes with CIs and Kruskal–Wallis statistics. Use spatially robust tests. Soften "compounding" and "opposing gradients". | A1-m7; A2-M7; B1-M5e, B1-m14; B2-m5 |
| I6 | Energy module | Document how each coefficient was derived. Benchmark against DESNZ/National Grid London demand. Include offices. Use floor area if possible. Report building-use coverage by borough. Propagate uncertainty. Label results as July only. Or move the module to Extended Data with caveats (D5). | A1-m12; A2-M5b/c; B1-M7, B1-m10, B1-m11 |
| I7 | WBGT and work capacity | Give the WBGT equation. Hold vapour pressure fixed. Apply to an hourly or daily distribution. Weight by workplace-zone employment or built land. Describe the 10→30 m resampling. Or move to Extended Data (D5). | A1-m13; A2-M5d; B1-M8 |
| I8 | Feasibility and realism | Frame Green30/Target30 as stylised upper bounds and elevate Green10/Target10. Add a 50%-crown-attainment sensitivity analysis. Reconcile with TreeOpp (+8%). Note that Suburban pixels are largely private gardens. Indicative costs are optional. | A1-M4 (feasibility); A2-M2a–d, A2-M2f |
| I9 | Night-time and indoor exposure | Add at least a night-time UCM run as a sensitivity analysis, or a dedicated limitation paragraph on daytime outdoor air temperature vs indoor and nocturnal exposure. | A1-M8; B1-M6, B1-m7; B2-m9 |
| I10 | Health framing and literature | Restructure the Introduction around London's heat-health burden (who dies, where). Position against Iungman et al. 2023, MCC/DLNM work and London UHI-mortality studies (authors to verify). Pose the allocation question explicitly in the Introduction. Shorten the restated Discussion paragraphs and repeated contextual material. | A1-M8; A2-M5, A2 §6 |
| I11 | Gentrification and London-specific guidance | Add a Discussion paragraph on green gentrification and anti-displacement measures. Make guidance London-specific (Urban Forest Plan, London Plan G5 / Urban Greening Factor, GLA Climate Risk Map). Optionally add a priority map of high-vulnerability LSOAs where even Target30 gives low benefit. | A2-M2e, A2-m6, A2-m7 |
| I12 | Scenario documentation | Report Scenario Generator settings (focal class, convertible classes, priorities, seed). Report TreeRisk details (fraction flagged, crown size, SSP/period, GiGL coverage of private trees). Report TreeOpp classes and density. | A1-m15; A2-m9; B1-m2, B1-m3, B1-m4 |
| I13 | Population-weighted reporting | Report population-weighted ΔT alongside unweighted borough means in Fig. 3. Show per-capita (per 100,000) values in Fig. 5 and Target − Green difference maps. | A1 Fig. 5; A2-m4, A2 Fig. 5; B1-M10c; B2-m10 |

### Tier 3: Minor, presentation and consistency fixes

**Number, table and figure checklist.** Verify each item against the *final* figures only.

- [ ] Abstract "up to 0.37 °C": replace with final values (Green30 −0.34, Target30 −0.38, after E2) and use "on average across boroughs" (V5).
- [ ] Abstract "up to 1.05 °C" and "up to 1,110": these are borough-mean and point values. Label AllBuilt as an extreme bounding counterfactual. Report TreeRisk (+0.10 °C, −105 deaths at present) as the realistic loss case.
- [ ] Abstract "135–496": name the scenarios (Green10 to Target30). Values will change after E1.
- [ ] Define "high-SVI" (top tertile, n = 1,612) in the Abstract; Fig. 2 uses quintiles.
- [ ] Results: "vegetation loss, including climate-related tree mortality, … up to 1.05 °C" conflates AllBuilt and TreeRisk.
- [ ] Results "3.9–4.2 °C / 4.8–5.5 °C" vs Fig. 1a-implied 3.8–4.2 / 4.7–5.4 °C.
- [ ] Introduction "≈2.7 °C by the 2050s" vs Fig. 1 (I2).
- [ ] Methods 25.5 / 25.9 °C vs Fig. 1a 25.7 / 26.1 °C.
- [ ] Fig. 1b insets: identical counts (2, 4, 7, 10, 10) in 2050 and 2070; check against "13 of 33 > 4 °C" and "11 > 5 °C"; remove unused legend classes (5.6–5.8 °C).
- [ ] GCM list: Methods vs Sup. Fig. 12; SSP245 in Sup. Fig. 2.
- [ ] Table 1: add Target10/20/30 and canopy km²; TreeRisk absolute 18% vs 19.56 × 0.94 ≈ 18.4%; report the model-LULC baseline TCC.
- [ ] "Six LULC scenarios" (Methods) vs "Nine scenarios (Table 1)" (Fig. 4) vs Sup. Table 1 (ten including baseline).
- [ ] "Five health outcomes": state that cause-specific outcomes are subsets of all-cause and not additive (ED Fig. 1).
- [ ] Remove promised but unmodelled outcomes ("physical/mental health outcomes", "physical activity") from the UCM paragraph.
- [ ] Sup. Fig. 9: "rates" → counts, or re-plot as rates; describe or remove the Y10–Y34 panel.
- [ ] ED Fig. 1: remove the stale Green-only version (161/377/−1,078); keep only the final version.
- [ ] ED Fig. 2: update with Target scenarios, units, labels and a sign convention matching ED Fig. 1, or delete it.
- [ ] Remove the uncaptioned Methods-region images after Fig. 7 (image32–34) and the superseded Fig. 3b (image12) from the file.
- [ ] Fig. 3a: relabel the axis "UCM setting (T_ref/UHI_max)" or drop the panel. Fig. 3b: add a spread measure and canopy km² (cooling per unit canopy).
- [ ] Fig. 4: put "exposure–response-only 95% UI" on the panel; visually separate AllBuilt; remove ".." at the end of the caption.
- [ ] Fig. 5: per-capita scale; supply borough tables (Target30 range 0.29–29.75; 22/33 and 21/33 counts) in the SI.
- [ ] Fig. 6: mark the 10th/90th percentiles; overlay Target30.
- [ ] Fig. 7: give SVI tertile cut-points; add low-vulnerability groups and resident-weighted percentages; note that SVI is a targeting input.
- [ ] Fig. 2: add n per quintile, effect sizes and the axis unit.
- [ ] LSOA count 4,835 vs 4,831 (Sup. Fig. 12); explain.
- [ ] State the SE derivation and distribution for β draws, and the seed; state the AF sign convention for cooling.
- [ ] SI step 4: remove the duplicated fragment (". , using average crown-size assumptions…"); "tract-level" → "LSOA-level".
- [ ] SI captions referring to "the same as Sup. Fig. 2" for LULC colours: correct to Sup. Fig. 1. Make the Sup. Fig. 1 legend readable. Harmonise the S6-1 and S7 numbering with Sup. Figs 1–12.
- [ ] Sup. Table 2: "ECOTRESS" → ECOSTRESS; state the albedo product year and statistic; check the inconsistent "green area" flags (Acid grassland, Arable, Saltmarsh = 0).
- [ ] Sup. Table 3: state which SSP's ET₀ is used ("SSP370 or 585"); fix "Urban Cooling mode".
- [ ] Sup. Table 4: units (kWh m⁻² °C⁻¹ month⁻¹), tariff sources and year.
- [ ] RH SI derivation: clarify "(Tmax+Tmin)/2, or … T = Tx"; remove the unsupported closing sentence or use it.
- [ ] UHI source: cite the peer-reviewed reference, not a URL; add a citation for the "2016 London climate report".
- [ ] Clarify the resolution chain (10 m LULC, 30 m work grid, 100 m population, health grid) and "the same high spatial resolution".

**References and language.**
- [ ] Ref. 5 (Bowler 2010) cited for heat vulnerability of older adults; ref. 42 cited for variation in heat ERFs; ref. 58 (under-five LMIC mortality) cited as the HIA basis; ref. 11 used for "citywide" cooling. Check and replace as needed.
- [ ] Ref. 19 incomplete; refs 45 and 53 duplicate; "61,000 deaths in Europe" → specify summer 2022.
- [ ] "Stanford InVEST suite" → Natural Capital Project / Natural Capital Alliance InVEST; state the InVEST version actually used and the UCM cooling-capacity option.
- [ ] Typos: ". ." (Results), "vulnerability ." (Results), ".." (Fig. 4 caption), "(a)" vs "b," (Fig. 1 caption), "results are consistent" → "is"; "bias-corrected" (WorldClim); "business-as-usual" for SSP370.
- [ ] Use UK or US spelling consistently, per journal style.
- [ ] Competing interests: note the authors' role in developing InVEST.

---

## 5. Suggested reframing of the narrative

**Revised one-paragraph pitch.** Many cities set tree-canopy targets as aggregate percentages, but a target says nothing about who is protected. Using London, we hold climate fixed and compare canopy expansions of identical size that differ only in where trees go: extending existing canopy versus siting street trees by heat exposure and social vulnerability. We trace modelled cooling through a validated urban-cooling model and a London-specific temperature–mortality function to warm-season heat-attributable deaths, and evaluate their distribution with indicators independent of the targeting rule. At London's policy target (+10% relative canopy), allocation alone changes modelled heat-attributable deaths by [X]% (range across specifications [X–X]). It also shifts the share of benefit reaching the most deprived quintile from [X]% to [X]%. Control allocations show that [mechanism: street-pixel siting / heat siting / vulnerability weighting] accounts for most of this difference. Allocation also creates trade-offs: neighbourhoods with older populations receive [more/less] benefit under targeting, which shows that the definition of "vulnerability" decides who gains. Greening is a modest lever against London's heat burden (about [X]% of warm-season heat-attributable deaths at the policy target). Where trees go determines whether that lever narrows or widens heat-health inequities.

**Title options.**
1. "Where trees are planted shapes who is protected from urban heat in London"
2. "Allocation, not only extent, determines the heat-health equity of urban tree planting"
3. "Matched canopy targets, unequal protection: spatial allocation of urban greening and heat-related mortality in London"
4. "Equity trade-offs in targeted urban greening for heat-health protection"

(Avoid "reshape heat-health benefits" unless the corrected health effects are clearly non-trivial.)

**Outline of the revised Abstract once the mortality approach is corrected.** Use placeholders; do not carry over current numbers.

1. *Problem (1 sentence):* canopy targets are aggregate, while heat mortality and its drivers are unequally distributed. It is unknown whether allocation changes who benefits.
2. *Design (1–2 sentences):* London; climate held fixed at [setting]; matched-canopy Green, Target and control allocations at +10/20/30%; validated UCM (RMSE [X] °C); London-specific non-linear ERF; warm-season, non-pandemic, age-specific baseline mortality.
3. *Context for magnitude (1 sentence):* modelled baseline warm-season heat-attributable deaths of [X] (benchmark [X]). The policy-target canopy reduces these by [X] ([X–X] across specifications), or [X]% of the burden.
4. *Allocation effect (1 sentence):* at equal canopy, targeted allocation changes citywide benefit by [X]%, driven mainly by [mechanism] (from control allocations).
5. *Distribution (1–2 sentences):* using indicators not in the targeting rule, the concentration index changes from [X] to [X], and the share of benefit reaching the most deprived quintile rises from [X]% to [X]%.
6. *Trade-off (1 sentence):* neighbourhoods with the largest ≥75 populations [gain/lose] ([X]), with age-specific susceptibility included.
7. *Robustness (1 clause):* Target > Green ranking holds in [X]% of sensitivity runs. Results are conditional on daytime outdoor air temperature and fully grown canopy.
8. *Implication (1 sentence):* canopy targets should be paired with explicit, deliberately defined allocation objectives.
9. *Co-benefits:* omit from the Abstract, or give one clause, if they have been moved to Extended Data.

**Suggested main-text structure.**
1. London's heat-health burden and its spatial and social structure, including fine-scale baseline exposure.
2. The matched-allocation experiment; magnitude of greening against the burden.
3. Why allocation matters (decomposition with controls).
4. Distribution on independent indicators, and the age trade-off.
5. Robustness.
6. London-specific implications.

Climate projections, AllBuilt, energy and work capacity go to Extended Data.

---

## 6. Point-by-point response template

Instructions: for each comment, state (i) the action taken, (ii) where it appears in the revision (page, line, figure or table), and (iii) the effect on results. Cross-reference Tier items (E#, I#) where one change addresses several comments.

### Reviewer #1 (A1)

| ID | Comment (short) | Tier | Response | Change location | Effect on results |
|---|---|---|---|---|---|
| A1-M1 | July ΔT applied to annual deaths; benchmark; Sup. Fig. 9 | E1 | | | |
| A1-M2 | Causal language; β-only intervals; sensitivity; validation | E3, I1 | | | |
| A1-M3 | Equity lacks susceptibility; age-specific rates; decomposition | E4 | | | |
| A1-M4 | Circularity; typology; matching; feasibility; "conventional"; controls | E5–E7, I8 | | | |
| A1-M5 | Climate baseline mismatch; UHI step; GCM list | E2, I2 | | | |
| A1-M6 | ≥75 trade-off omitted or contradicted; SVI construction | E8 | | | |
| A1-M7 | Abstract numbers vs figures | E9, Tier 3 | | | |
| A1-M8 | Thin health framing; co-benefits; literature; "conservative" | I1, I6, I7, I9, I10 | | | |
| A1-M9 | Discussion overclaims | I1 | | | |
| A1-m1 … m18 | (list each separately) | Tier 3 / I# | | | |
| A1 §6 rows 1–30 | Consistency audit | Tier 3 | | | |
| A1 §7 | Figure comments (Figs 1–7, ED 1–3, Tables 1–2, SI) | Tier 3 | | | |

### Reviewer #2 (A2)

| ID | Comment (short) | Tier | Response | Change location | Effect on results |
|---|---|---|---|---|---|
| A2-M1 | Circular equity metric; non-circular indicators; EJ statistics; allocation benchmarks | E5, E6 | | | |
| A2-M2 | Feasibility, maturity, gentrification, cost | E7, I8, I11 | | | |
| A2-M3 | ≥75 trade-off; age-specific mortality; UK-native index | E4, E8, I3 | | | |
| A2-M4 | Decompose drivers of Target's advantage | E6 | | | |
| A2-M5 | Magnitudes; benchmarks; co-benefit integration | E1, I6, I7 | | | |
| A2-M6 | Internal inconsistencies table | E7, E9, I2 | | | |
| A2-M7 | Baseline inequality evidence (Fig. 2) | I5 | | | |
| A2-m1 … m17 | (list each separately) | Tier 3 / I# | | | |
| A2 §6–7 | Structure and figure comments | I10, Tier 3 | | | |

### Reviewer #3 (B1)

| ID | Comment (short) | Tier | Response | Change location | Effect on results |
|---|---|---|---|---|---|
| B1 §1(a)–(h) | Items not reconstructable from text | E1, E2, E6, E7, I3, I7 | | | |
| B1-M1 | No UCM validation | E3 | | | |
| B1-M2 | UHI_max source, switch, scaling | E2 | | | |
| B1-M3 | Biophysical table errors (woodland shade, Urban canopy) | E2 | | | |
| B1-M4 | Matching; source-class confound; UTCI source | E6, E7 | | | |
| B1-M5 | Climate forcing baseline, GCMs, T_ref | I2 (see D4) | | | |
| B1-M6 | Exposure time base for mortality; night-time | E1, I9 | | | |
| B1-M7 | Energy model | I6 | | | |
| B1-M8 | WBGT and work capacity | I7 | | | |
| B1-M9 | Sensitivity and uncertainty | E3 | | | |
| B1-M10 | Scale handling; consistency | E4, E9, I13 | | | |
| B1-m1 … m18 | (list each separately) | Tier 3 / I12 | | | |
| B1 §6 | Reproducibility checklist | E9 | | | |

### Reviewer #4 (B2)

| ID | Comment (short) | Tier | Response | Change location | Effect on results |
|---|---|---|---|---|---|
| B2 §1(a)–(h) | Unreported items | E1, I3, I4 | | | |
| B2-M1 | Annual totals from July ΔT; back-calculation | E1 | | | |
| B2-M2 | ERF metric, MMT, cause-specific RRs | E1 | | | |
| B2-M3 | Climate invariance; UHI_max sensitivity | E2, E3 | | | |
| B2-M4 | COVID baseline; crude rates; fixed population | E1, E4 | | | |
| B2-M5 | Uniform allocation drives equity patterns | E4 | | | |
| B2-M6 | Circularity; level confound; losers; controls | E5, E6, E7 | | | |
| B2-M7 | SVI construction | I3 | | | |
| B2-M8 | GAM inference | I4 | | | |
| B2-M9 | Uninformative intervals; joint uncertainty | E3 | | | |
| B2-M10 | "Deaths averted" language | I1 | | | |
| B2-m1 … m13 | (list each separately) | Tier 3 | | | |
| B2 §6 | Statistical reporting checklist | E9, I4 | | | |

### Editor's points (C)

| ID | Point | Response |
|---|---|---|
| C-1 | Confirm D₀ definition and provide the deaths-per-°C reconciliation (V1) | |
| C-2 | State the UHI product, variable and statistic, and justify an air-temperature UHI_max (V2c, D3) | |
| C-3 | Confirm that all legacy embedded images are removed, and give the pipeline commit used for final figures (V16, D7) | |
| C-4 | State whether earlier UCM error bars (old Fig. 3b) were removed deliberately, and what they represented | |
| C-5 | Choose option (a) or (b) for Fig. 1 and the climate chain (D4), and for co-benefit placement (D5) | |

---

## 7. Confidential note on fit for Nature Health and alternative venues

*(For the editorial team; not for transmission to the authors.)*

**Fit.** The question (equity of canopy allocation for heat-health protection), the policy hook (city canopy targets) and the matched-allocation design all suit *Nature Health*. The current version, however, is an urban-climate and ecosystem-services modelling study with a health module whose central quantity is very likely mis-specified. I reproduced the arithmetic of V1 from the paper's own figures: deaths averted per °C ≈ annual London deaths × ln(1.02). I do not regard this as misconduct. It is a seasonal-scope error compounded by hasty assembly; the legacy figures show the text was not fully updated after regeneration. All referees offered to re-review.

**What resubmission should demonstrate before it is sent out.** Triage the resubmission before sending it to referees. It should go out only if:
- (i) the mortality baseline is season-appropriate and benchmarked against an independent London heat burden;
- (ii) the woodland and UHI parameterisation is corrected, with some observational validation;
- (iii) control allocations and non-circular equity metrics are present;
- (iv) the figures and text are consistent.

If the corrected health effect is very small (e.g. a few deaths per summer at the policy target), the paper can still have a home here, but only if the distributional and allocation findings are robust and framed as such. If they are not robust, the paper loses its *Nature Health* rationale.

**Panel composition for re-review.** Retain B2 (HIA and biostatistics) as the key methods referee, as A1 suggested. Retain B1 for the physical chain. If energy and WBGT stay in the main text, consider a building-energy specialist.

**Alternative journals if core concerns cannot be resolved.** These should be mentioned only if the authors ask or after a second decline.
- If the health chain is corrected but the equity and allocation findings prove fragile, or the health effect is small: *Nature Cities* or *Nature Communications*, as a methods and scenario contribution; *Lancet Planetary Health*, if HIA quality is high and the London policy angle is strengthened.
- If the paper stays primarily an ecosystem-service modelling study with co-benefits: *Environmental Research Letters*, *Landscape and Urban Planning*, *Sustainable Cities and Society*, *Urban Forestry & Urban Greening*, or *One Earth*, if reframed around natural-capital decision tools.
- If the equity and allocation contribution is strong but the health quantification is modest: *Environmental Health Perspectives* or *Environment International*, for an HIA-centred version.

**Bottom line.** This is a promising and policy-relevant paper with a correctable but foundational flaw in its headline health numbers, a circular headline equity metric, and substantial file hygiene problems. Decline in current form, invite resubmission, and set a firm expectation of the essential analyses above.
