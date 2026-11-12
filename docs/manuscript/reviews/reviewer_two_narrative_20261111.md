# Independent reviewer 2 — narrative audit

## Scope and recommendation

Simulated academic peer review for a Nature Health submission; I am not acting for the journal. I read the full supplied manuscript and supplement text (`london_manuscript_review_20261111.md` and `london_supplement_review_20261111.md`), formed an independent assessment, and then consulted only the requested Figure 3 and Figures 4–5 audit documents to distinguish current manuscript problems from completed but unincorporated work. I did not read another reviewer's report, inspect the embedded figure images, rerun analyses, or conduct a full reference-validity/claim-support audit. Numerical statements below are manuscript statements or explicitly attributed repository-audit findings, not independent reproduction.

**Recommendation: major revision before submission.** The manuscript has a potentially strong health-equity contribution: comparable amounts of added canopy can deliver different aggregate benefits and redistribute benefits across vulnerability dimensions. The current narrative, however, alternates between a calibrated future-health forecast and a conditional spatial-allocation experiment. The former is not established by the supplied methods. The latter is a defensible, interesting contribution, especially when its older-population trade-off is treated as a central result rather than an exception.

## Strengths

1. The matched-climate comparisons are a clear organizing principle, and the Methods explicitly states that Green and Target are matched on realized added-canopy area. This provides an interpretable planning question beyond whether vegetation cools cities.
2. The Results now separates general and targeted greening and reports the same-neighborhood comparison using fixed Green30 thresholds. The 45.5%→4.5% contrast is intelligible without relying solely on GAM curves.
3. Figure 4's legend distinguishes monthly energy, area-weighted work capacity, annual-labelled health outcomes, fixed 2021 demographics, and exposure–response-only uncertainty. These qualifications are unusually useful and should govern the prose throughout.
4. The explicit statement that Target30 does not improve every borough and the older-population trade-off make the policy interpretation more credible.

## Major comments

### N1. Make the study's estimand—not the general heat crisis—the narrative spine

**Anchors:** Introduction, “We integrate CMIP6-based climate projections”; Results, “Across all future periods”; Methods opening, “under identical mid-century climate conditions”; Supplementary Table 3, reference temperatures 22/25/28°C and UHI 2/5/5°C.

The reader is led to interpret the co-benefits as SSP-conditioned 2050 predictions. The text does not establish that the prescribed urban-cooling parameters constitute such a projection. Separately, Figure 1 compares future WorldClim climatologies with a single CHELSA year. These are distinct analytical objects, not a continuous prediction chain merely because they both use temperature.

**Revision:** State early that the principal question is how spatial allocation changes *conditional modelled benefits under common prescribed warm conditions*. Explain the role of Figure 1 as climate context separately from the controlled land-use comparison. Preserve a forecast framing only if the authors provide the needed climatic harmonization and coupling evidence. This is not solved by adding “modelled” to otherwise unqualified mid-century claims.

**Status distinction:** The Figure 3 audit documents an already-generated replacement that uses this conditional framing; it has not replaced the legacy Figure 3 text in the supplied manuscript. That resolves an integration task, not the underlying calibration/validation requirement.

### N2. Put the most novel comparison and its trade-off in the Abstract and concluding synthesis

**Anchors:** Abstract, “Vulnerability-targeted planting redistributed benefits”; Results, “the low-benefit share increased from 28.7% to 59.0%” for older-population neighborhoods; Discussion ending, “muted benefits in more socioeconomically vulnerable areas and stronger mortality reductions in areas with more older adults.”

The Abstract presents targeting as an unqualified equity improvement. The final paragraph then describes the general-greening pattern without making its scenario restriction clear, although the Target pattern is different. This can leave the reader with the wrong overall finding. The trade-off is not incidental: the composite vulnerability objective places competing demographic priorities in tension.

**Revision:** Include one short Abstract clause on the increased low-benefit share in neighborhoods with older populations. Organize the concluding synthesis around (i) canopy allocation changes aggregate outcomes, (ii) proximity-based greening favors some groups, and (iii) composite targeting shifts—not universally resolves—distributional disadvantage. Do not imply age-specific mortality was estimated; retain the existing neighborhood-composition qualification.

### N3. Define the novelty against the closest relevant literature, not the broad existence of greening benefits

**Anchors:** Introduction, “fewer translate cooling into multiple human outcomes within a unified framework”; Discussion, “A key contribution ... three linked outcomes”; Introduction last paragraph.

Cooling, heat mortality, energy demand and work capacity are established pathways. Combining model modules is useful but does not by itself establish scientific novelty or a new health insight. The sharper contribution appears to be a controlled spatial-allocation contrast with distributional evaluation and explicit demographic trade-offs.

**Revision:** Add a compact comparison with the closest integrated urban-greening/health-targeting studies: what they compare, whether canopy quantity is controlled, whether they report distributional effects, and what this study adds. This requires checking those papers, not assuming absence. Replace broad “remain poorly quantified” statements with the specific unresolved question. In the final Introduction paragraph, explicitly introduce Green versus Target, the common canopy budget, and the health-equity hypothesis or evaluative question.

### N4. Avoid turning a scenario-generator comparison into a verdict on conventional planning

**Anchors:** Abstract, “conventional proximity-based greening”; Methods Green10, “reflecting typical planting patterns”; Discussion, “amplifying spatial inequalities”; Supplementary Methods, street-based Target candidates.

Green is a nearest-edge algorithm; Target uses a different street-candidate construction plus joint heat/SVI prioritization. Neither automatically represents ordinary London planning, and matched added-canopy area does not make the two interventions identical except for equity prioritization. The current language risks attributing the observed contrast to municipal practice or the targeting criterion alone.

**Revision:** Call Green “proximity-based” or “nearest-edge” consistently. Support “typical” with evidence or remove it. Describe the result as comparing two allocation strategies unless a same-candidate-set control isolates the priority score. Explicitly distinguish equal area from equal cost, feasibility, survival or implementation effort. “Feasible planting strategies” in the Abstract and “maximum plausible” in Table 1 should become “spatially constrained candidate scenarios” until real-world constraints are demonstrated.

### N5. Distinguish co-benefit indicators from measured economic and clinical outcomes

**Anchors:** Results, “lowering heavy-work productivity by 6.37%”; Discussion, “livelihoods,” “measurable work productivity benefits,” and “largest combined benefits”; Limitations, “conservative and policy-relevant estimate.”

The work endpoint is an area-weighted physiological capacity index, not worker-weighted output, earnings or hours. Energy figures are modelled monthly avoided costs with restricted building assumptions, not net economic gains. Health figures are conditional expectations rather than observed prevention. Heavy-work focus is not necessarily conservative: it selects a particularly heat-sensitive response and does not enumerate the exposed workforce. Comparing directions across endpoints does not aggregate them into a combined welfare metric.

**Revision:** Use “heavy-work capacity” and percentage points throughout; identify July/warm-month energy costs wherever numbers first appear. Remove “conservative” unless established by a bounding analysis. Replace “largest combined benefits” with “largest modelled citywide benefit for each of the three indicators” if that is what is shown. Keep livelihood and affordability implications as possible relevance, not measured findings. Resolve the annual mortality interpretation in the separate methods review before emphasizing annual lives saved.

### N6. Tighten the Discussion around what the study actually learned

**Anchors:** “Land-use remains a modifiable lever...” and “Co-benefits link cooling...” repeat general heat statistics and foundational evidence; stewardship and SDG paragraphs; “results likely understate the full suite of benefits or trade-offs.”

The Discussion devotes considerable space to re-establishing that heat is harmful and trees provide benefits, while the distinctive allocation comparison and its uncertainties receive less synthesis. Some proposed mechanisms—implementation capacity, baseline health burden, limited plantable land—are plausible explanations, not findings from the presented analysis. Omitted benefits do not establish the direction of net bias when costs and adverse pathways are also omitted.

**Revision:** Shorten repeated global statistics; lead with the matched-area contrast, spatial heterogeneity and demographic trade-off. Clearly label untested mechanisms. Condense stewardship and SDG material to a short implications paragraph. Replace “likely understate” with “do not estimate the net or total effects of urban greening.” Add displacement/green-gentrification and procedural equity as boundaries of the study's specific distributional definition of equity, without requiring a new qualitative study.

## Minor comments

1. **Table 1/Methods scenario inventory:** the text says six scenarios and Table 1 lists only six, although Figures 4–5 and the central argument use nine. Add Target10/20/30 and concise common-budget/candidate-rule descriptions.
2. **Temperature language:** “mean temperature of the warmest month,” “average maximum,” “peak summer temperatures,” and “July mean daily maximum” are not interchangeable. Define one metric and distinguish it from heatwave maxima and UCM air temperature.
3. **Numerical integration:** the current +1.05/−0.37°C extrema and Figure 3 uncertainty wording remain legacy. The Figure 3 audit proposes +1.11/−0.32°C London-area-weighted means. Do not simply swap numbers without changing aggregation/comparator wording. Figures 4–5 audit records AllBuilt capacity −6.36 percentage points, whereas the manuscript says −6.37%; reconcile source and rounding.
4. **Evidence language:** Figure 2 statistical significance is not evidence that vulnerability causes heat or reduced canopy. Report effect sizes and keep “compounding risk” an interpretation rather than a demonstrated mechanism.
5. **Scope of “mental health”:** the model includes mortality categories, not a general mental-health outcome. Use precise terminology in the UCM overview and Discussion.
6. **Presentation:** remove the accidental heading formatting on the entire Target Results paragraph, double punctuation after “Green30” and the Figure 4 legend, the repeated crown-size clause in Supplementary Methods, and the dangling “health disorders” phrase in the Introduction. Check supplement cross-references: land-cover scenario captions repeatedly point to Sup. Fig. 2 although that figure is climate.

## Additional technical evidence received after the independent draft

The subsequent repository `figure3_ranking_input_audit.md` reports a controlled Green30 missing-data-encoding sensitivity large enough to change its cooling ordering against historical Target30. This was not evident from the supplied manuscript and is addressed separately in methods comment M11. Until resolved, avoid a narrative that explains historical-to-revised ranking changes solely through improved planting design or model upgrades. This additional evidence reinforces the need to ground the story in a validated, consistently versioned analysis rather than reconcile old and new numbers rhetorically.

## Editorial readiness

The health-equity story is potentially competitive, but not submission-ready in its present framing. Prioritize the climate/health estimand and methodological blockers first; then integrate the already-generated Figure 3 replacement, align every scenario/number/qualifier, and shorten the Discussion. A full reference-validity and claim-support audit and image-level figure review remain separate pre-submission tasks.
