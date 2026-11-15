# Referee Report — Reviewer #2

**Manuscript:** "Urban greening strategies reshape heat-health benefits and equity" (London)
**Journal:** Nature Health
**Reviewer expertise:** urban nature-based solutions, green infrastructure and tree-canopy equity, social vulnerability, environmental justice, policy translation
**Remit this round:** the full narrative: novelty, storyline, how realistic and policy-relevant the scenarios are, whether the equity inferences hold, translation to practice, generalisability, limitations, and clarity of the figures and writing. I comment on methods only where they weaken the narrative. A separate panel covers methods in depth.

---

## 1. Summary of the submission

The authors combine CMIP6 (WorldClim v2.1) climate projections for London with the InVEST Urban Cooling Model (UCM). They run ten land-use/land-cover configurations: a baseline; an extreme vegetation-to-impervious counterfactual (AllBuilt); climate-driven canopy loss (TreeRisk); opportunity-land planting (TreeOpp); three proximity-based canopy expansions (Green10/20/30, i.e. +10/20/30% relative canopy, with Green10 matching the 2018 London Environment Strategy target); and three vulnerability-targeted street-tree expansions matched on added canopy area (Target10/20/30, Supplementary Methods). Modelled temperature changes feed exposure–response functions for building cooling-energy cost, heavy-work capacity, and all-cause plus four cause-specific heat-related deaths. Distributional analyses at the LSOA level (n = 4,835) use a four-component, CDC-inspired Social Vulnerability Index (SVI) and its components.

The headline claims are:
- Under constant climate, land use shifts borough temperatures by +1.05 °C (AllBuilt) to about −0.37 °C (greening).
- Greening averts 135–496 deaths per year, and AllBuilt adds 1,110.
- Greening avoids up to £22.4 m per month in energy costs and raises heavy-work capacity by up to 1.42 percentage points.
- Proximity-based greening gives smaller per-resident benefits to more income-deprived and higher ethnic-minority-share neighbourhoods.
- Vulnerability-targeted planting reverses these gradients: among high-SVI LSOAs, the share receiving "low" benefit falls from 45.5% (Green30) to 4.5% (Target30).
- The take-home message is that "its contribution to equitable heat adaptation depends critically on where new trees are allocated."

## 2. Overall assessment: novelty, significance, fit for Nature Health

The question matters and suits a health-and-policy readership. Cities, London included, have canopy targets expressed as aggregate percentages, and the paper rightly argues that "target attainment alone does not reveal who benefits." The integrated design (one scenario set, several outcomes, explicit spatial allocation) is a sensible framework. The paired Green-versus-Target comparison matched on canopy area is the most interesting design element in the paper.

In its current form, however, I do not think the manuscript reaches the bar for novelty or inferential strength that Nature Health requires. There are four main reasons.

1. **The central message is not new as stated.** That tree-canopy benefits are unequal, that existing canopy follows income and race/ethnicity, and that the distribution of benefit depends on where trees go are established findings. Relevant work includes the authors' own global analysis (ref. 33), McDonald et al. (ref. 39), the American Forests "Tree Equity Score" framework, European health-impact assessments of tree-cover targets (e.g., Iungman et al., *Lancet* 2023, *verify*), and a growing literature on optimising equitable tree-planting allocation. The paper needs to state precisely what it adds beyond "targeting works."
2. **The equity result is largely circular by construction.** Target scenarios are ranked on UTCI × SVI percentile, and success is then judged on the same SVI. The 45.5% → 4.5% headline is therefore close to a design check, not a finding (Major 1).
3. **The scenarios that carry the headline results are not shown to be feasible.** Target30 requires about 1.8 million new street-tree sites at 5 m spacing. Green30 exceeds the authors' own feasibility-based TreeOpp scenario (+8%) by nearly four times. Neither scenario is costed, and no time to canopy maturity is modelled (Major 2).
4. **Several internal inconsistencies weaken trust in the numbers.** Abstract, text and figures disagree on temperature effects, death counts, GCM lists, scenario tables and warming magnitudes (Major 6).

There is a publishable and useful paper here. To become one for Nature Health, it should be reframed around what the matched comparison can genuinely show: how much allocation alone changes efficiency and distribution, against non-circular benchmarks and under feasible, costed planting constraints. The claims should also be tightened to what a deterministic, borough-mortality-allocated model supports.

## 3. Strengths

1. **Policy anchoring.** Scenarios are tied to an actual municipal target (London Environment Strategy +10% relative canopy by 2050) and to existing London strategies. That makes the work legible to practitioners.
2. **Matched-allocation design.** "Green and Target scenarios were matched on realized added-canopy area to isolate the influence of spatial allocation" (Methods). This is the right experiment for the question asked, and it is underused in the literature.
3. **Climate–land-use separation.** The counterfactual logic (land-use effects evaluated under identical climate) is clearly explained and helps readers who conflate the two.
4. **Candid on trade-offs and limits.** The authors acknowledge that targeting does not improve every borough ("it delivered greater health ... benefits in 22 of 33 boroughs"). They report a trade-off for ≥75 neighbourhoods. They state in Supplementary Fig. S7 that "SVI is related to the targeting design and is not an independent validation measure." The maintenance/aftercare discussion (Discussion, "In practice, this also requires attention to implementation and long-term stewardship") is welcome and practice-informed.
5. **Multiple outcomes in one framework.** In principle this permits synergy and trade-off analysis across health, energy and labour.
6. **Open code** (GitHub) and clear data-availability statements.

## 4. Major concerns

### Major 1. The headline equity metric is circular. The equity conclusion needs non-circular evidence.

**Location:** Abstract ("Target30 reduced the share of high-SVI neighborhoods receiving relatively low health benefits from 45.5% to 4.5%"); Results, "Targeting redistributes health benefits..."; Fig. 7; Supplementary Methods step 3 ("Priority score = UTCI_norm × SVI"); Supplementary Fig. S7 caption.

**Why it matters:**
- Target scenarios place trees preferentially where SVI is high. It is therefore close to inevitable that high-SVI LSOAs receive more benefit, and that "all five 90th–10th percentile contrasts reversed direction." The same applies to SVI components such as income deprivation and ethnic-minority share, which are strongly collinear with SVI (Sup. Fig. 10).
- The reported redistribution is therefore largely a validation that the allocation algorithm did what it was designed to do. It is not evidence about the effectiveness of equity-targeted greening.
- The fixed-threshold metric adds a second issue. Green30 cut-points (33.21, 51.95 deaths averted per million) are applied to Target30, which also has higher total benefit (496 vs 387). Some of the shift out of the "low" class is level shift, not redistribution.
- The authors acknowledge the circularity only in a Supplementary caption. Yet the metric is the abstract's concluding quantitative claim.

**Requested actions:**
- (a) Remove or demote the 45.5% → 4.5% statistic from the abstract, or pair it with explicitly non-circular metrics.
- (b) Evaluate distributional outcomes on dimensions **not** used in targeting. Candidates include baseline heat-mortality risk, the share of residents with long-term illness or disability (Census 2021), overcrowding or top-floor flats, lack of private outdoor space, existing canopy deficit, and the GLA Climate Risk Mapping index (ref. 63) if it was not an input.
- (c) Add standard environmental-justice summary statistics computed on residents, not neighbourhoods: concentration indices or Gini/Theil of benefit by deprivation decile, and population-weighted benefit shares by IMD decile.
- (d) Add alternative allocation benchmarks matched on canopy area: heat-only targeting (UTCI only), canopy-deficit targeting, population-density targeting, random allocation, and an efficiency-optimal allocation (maximise deaths averted). Show where Target sits on an efficiency–equity frontier. Without these benchmarks, readers cannot tell whether Target's advantage comes from vulnerability weighting, from heat weighting, or simply from placing trees where people live (see Major 4).
- (e) Use resident-weighted statistics throughout. The authors already note that "Percentages describe neighborhoods, not residents."

### Major 2. Feasibility of Green30 and Target30 is not demonstrated. Realism and green gentrification are not addressed.

**Location:** Table 1 (Green30 "maximum plausible co-benefits"); Methods (6) ("approaches structural reconfiguration of urban land cover within existing spatial constraints"); Supplementary Methods step 4 ("For Target20 and Target30, approximately 1,129,000 and 1,819,000 tree sites were selected"); Discussion paragraph on stewardship.

**Why it matters:**
- *Scale.* About 1.8 million new street and footway trees at 5 m spacing, set 3.5 m off residential street centrelines and along footpaths and cycleways, would be a multiple of London's existing street-tree stock (on the order of 1 million; *verify* against GLA/London Tree Officers figures). The rules ignore the main real-world constraints: underground utilities, footway width and accessibility standards, sightlines, driveways and crossovers, root–building conflicts on shrink-swell London Clay, and highway-authority approvals.
- *Inconsistency with the authors' own feasibility scenario.* TreeOpp, which the paper calls "feasible canopy expansion based on available planting space", yields only +8% relative canopy and 56 deaths averted. The text never reconciles this with the claim that Green20/30 and Target20/30 remain "within existing spatial constraints." Green scenarios also convert "suburban and urban" pixels to tree cover. In London, the suburban class is largely private gardens and housing, which planners cannot directly allocate.
- *Instantaneous maturity.* Canopy is represented with "average crown-size assumptions," apparently at full size, under 2050 climate. Street trees planted in the 2020s–2030s would be far from mature by 2050, and urban street-tree mortality in establishment years is substantial. This bears directly on the policy translation ("by 2050").
- *Green gentrification and displacement.* Targeting greening at high-SVI, high-ethnic-minority neighbourhoods is exactly the setting in which the environmental-justice literature documents green gentrification and displacement risk (e.g., the work of Anguelovski and colleagues; *verify*). The Discussion does not engage with this. For a paper whose thesis is equity, the omission is significant: the benefit may accrue to future, not current, residents.
- *Cost.* No capital or maintenance costs are given, so readers cannot compare Target10 against Green30. Target20 averts nearly as many deaths as Green30 (345 vs 387) with two-thirds of the canopy, and that is potentially the most policy-relevant result in the paper.

**Requested actions:**
- (a) Report the absolute added canopy area (km²) and number of trees for every scenario in Table 1. Compare these with London's current street-tree and total tree stock.
- (b) Apply a realistic constraint layer to Target scenarios (footway width, utilities proxies, existing street furniture, or the GLA/Forest Research planting-opportunity data). Report the share of Target30 sites that survive these constraints.
- (c) Add a growth trajectory (canopy at 2030/2040/2050 under realistic establishment and mortality), or at least a sensitivity analysis with 50% crown attainment.
- (d) Add an indicative cost-per-death-averted (or per DALY) comparison across scenarios, drawing on published UK street-tree planting and aftercare costs.
- (e) Add a Discussion paragraph on green gentrification and anti-displacement measures (community land trusts, tenure protection, co-design). Note that the model cannot capture residential sorting.
- (f) Reframe Green30/Target30 as stylised upper bounds unless feasibility is shown. Elevate Green10/Target10, which match actual policy, to the main comparison.

### Major 3. The ≥75 trade-off undercuts the health-equity framing and needs to be confronted, not footnoted.

**Location:** Results ("among neighborhoods with the highest proportions of residents aged ≥75 years, the low-benefit share increased from 28.7% to 59.0%"); Extended Data Fig. 3 (Target benefits fall almost to zero at high ≥75 z-scores); Discussion, closing paragraph ("stronger mortality reductions in areas with more older adults").

**Why it matters:**
- Older adults are the population most physiologically susceptible to heat mortality, and the Introduction itself emphasises "biologically vulnerable populations." A strategy that roughly doubles the low-benefit share in older neighbourhoods is a first-order finding for a health journal.
- The closing paragraph of the Discussion states that Green scenarios yield "stronger mortality reductions in areas with more older adults" without noting that Target reverses this. That is selective.
- Mechanistically, mortality is "allocated to pixels in proportion to population" from borough totals, and a single all-age RR is used. Within a borough, the per-resident benefit therefore varies only with ΔT, never with age structure. The ≥75 associations thus reflect where trees were placed relative to suburban, older, whiter LSOAs. They do not reflect any age-specific health response; the authors acknowledge this ("not age-specific mortality responses").
- The same logic means the SVI–benefit associations in Fig. 6 mainly describe correlations between canopy allocation and demographic geography. They are not health-equity effects in the epidemiological sense.
- The CDC-SVI construction compounds the problem. It places "percent over age 75" in the SVI, yet that component is negatively correlated with the others (Sup. Fig. 10), so the composite index partly cancels a key heat-susceptibility dimension.

**Requested actions:**
- (a) Use age-specific baseline mortality (ONS publishes deaths by age at local-authority level, and LSOA population age structure is available) together with age-specific heat RRs. At minimum, run a sensitivity analysis in which ≥75 residents carry a higher RR.
- (b) Report how the Target trade-off changes when heat susceptibility (age, chronic illness) is included in the priority score.
- (c) Rewrite the closing Discussion so that it states the trade-off explicitly and its implication: "vulnerability" must be defined deliberately, and different definitions yield different winners.
- (d) Justify the CDC-SVI adaptation over UK-native indices (IMD 2019/2025 domains, Wolf & McGregor's London heat-vulnerability index, ref. 8, the GLA Climate Risk Map, ref. 63). Report results with at least one UK-native index. Describe in Methods how the components are combined (weights, percentile ranking); this is currently not stated. Note too that the Supplementary Methods refers to "tract-level SVI percentile", a US term, while Methods uses LSOA. Clarify which spatial unit was used for targeting.

### Major 4. What drives Target's higher aggregate benefit? The "where" story needs decomposition.

**Location:** Results ("larger citywide benefits under Target at each tree canopy tier across all three outcomes"); Fig. 3b (Target10 −0.14 vs Green10 −0.12 °C); Fig. 4.

**Why it matters:**
- The paper's most interesting and least circular result is that, *at equal canopy area*, Target allocation also raises aggregate deaths averted: 188 vs 135 at the 10% tier (+39%) and 496 vs 387 at 30%. It also raises energy and work-capacity gains.
- Several mechanisms could produce this, and they imply different policy messages: (i) placing trees in denser, more populous areas, so that more people are exposed per unit of cooling; (ii) placing trees away from existing canopy, which increases marginal cooling under the UCM (diminishing returns near existing green space); (iii) placing trees in hotter locations (the UTCI term); (iv) the vulnerability term itself.
- If (i)–(iii) dominate, the message is "plant where people and heat are," which is an efficiency result, not an equity one, and the equity reversal is incidental.

**Requested actions:**
- (a) Run the benchmark allocations in Major 1d and decompose the Green→Target difference into these components.
- (b) Show marginal cooling per added hectare for both allocation rules, binned by distance to existing canopy.
- (c) Rewrite the Abstract and Discussion so that they state which mechanism carries the result.

### Major 5. Plausibility and positioning of the absolute health and economic magnitudes, and the integration of co-benefits.

**Location:** Abstract ("averting 135-496 deaths and vegetation removal producing up to 1,110 additional deaths per year"); Methods (Health outcomes, Energy); Fig. 4; Discussion, "Co-benefits link cooling...".

**Why it matters:**
- *Mortality.* The temperature input is a modelled July daytime field. The mortality denominators are "deaths registered in 2021" for whole calendar years, and the outcome is labelled "annual." If a July ΔT is applied to annual all-cause deaths, the annual totals will be substantially overstated, since heat-attributable mortality is concentrated on hot days in summer months. For context, UKHSA estimated about 2,985 heat-associated deaths across *all of England* in the extreme summer of 2022 (ref. 28). Simpson et al. (ref. 29) attribute 38% of London's July 2022 heatwave heat deaths to the UHI. Against these benchmarks, 496 deaths per year averted in London by a mean 0.38 °C cooling, or 1,110 added by AllBuilt, look high. I leave the technical audit to the methods panel. For the narrative, the authors must benchmark their numbers against UKHSA/ONS London heat-mortality estimates and explain any discrepancy before these figures appear in an abstract that a health audience will quote.
- *Energy.* Sup. Table 4 implies very large per-°C cooling coefficients for "Community Services" (9.89 kWh m⁻² °C⁻¹ month⁻¹ applied to 12.2 million m², a category that includes many non-air-conditioned schools and community buildings) and for Retail (30.8). These coefficients are derived from a heatwave-spike report (Drax). Readers need a sanity check against London's actual summer commercial cooling electricity use. The largest energy benefit falls in Hillingdon (Heathrow), which suggests a single land-use class drives the result.
- *Work capacity.* "Heavy-work capacity" is area-weighted, not worker-weighted, and London's heavy outdoor workforce is small. The WBGT derivation is not described in the main text. The Discussion's support (Indian manufacturing, ref. 32; global Lancet Countdown hours, ref. 4) is not transferable to London.
- *Integration.* As written, the energy and work-capacity strands feel bolted on. Each gets one sentence of results and a paragraph of literature-based plausibility. No analysis examines whether co-benefits align or conflict spatially (e.g., energy benefits accrue to commercial landowners in Hillingdon while health benefits accrue to residents in Newham, per Fig. 5), and no equity analysis is done for energy or work capacity. Residential cooling is treated as "negligible", yet domestic AC uptake by 2050 is a major equity issue (fuel poverty, overheating in flats).

**Requested actions:**
- (a) Benchmark mortality against independent London estimates. Clarify the temporal window (July only, or summer) and restate as "deaths averted per summer" if appropriate.
- (b) Either integrate the co-benefits properly or move them to the Extended Data/SI and keep the paper focused on health equity, which suits Nature Health better. Integration would include a spatial synergy/trade-off analysis (e.g., a bivariate map of health and energy benefits by scenario, and who captures each benefit: residents, businesses, workers) and an equity analysis for each outcome.
- (c) If energy is retained, add a domestic-cooling scenario for 2050 and discuss who pays and who saves.
- (d) Express work capacity per worker using workplace-zone employment data (ONS Workplace Zones), or drop it.

### Major 6. Internal inconsistencies between Abstract, Results, Methods, figures and tables.

These may look like proofing errors, but several affect headline numbers and, taken together, they undermine confidence.

| Item | Location A | Location B |
|---|---|---|
| Maximum greening cooling | Abstract/Results: "up to 0.37 °C", "reaching −0.37 °C" | Fig. 3b: Green30 −0.34, **Target30 −0.38** |
| Mid-century warming | Intro: "≈2.7 °C by the 2050s" | Results: "3.9–4.2 °C at mid-century" |
| July Tmax 2050 SSP370 | Methods: "25.5 °C" | Fig. 1a: 25.7 °C |
| Temperature metric | Results: "mean temperature of the warmest month" | Fig. 1 caption: "Average maximum temperature of the warmest month"; Methods: monthly Tmax |
| GCMs | Methods: EC-Earth3-Veg, MRI-ESM2-0, IPSL-CM6A-LR | Sup. Fig. 12: ACCESS-CM2, IPSL-CM6A-LR, MRI-ESM2-0 |
| Number of scenarios | Methods: "we developed six LULC scenarios"; Table 1 lists 6 | Fig. 4: "Nine scenarios (Table 1)"; Sup. Table 1 lists 10 incl. baseline. **Target10/20/30 are missing from Table 1** |
| Borough counts exceeding thresholds | Results: "13 of 33 boroughs ... exceed 4 °C ... 11 (33%) exceed 5 °C" | Fig. 1b insets (composite panel) show identical histograms (2, 4, 7, 10, 10) for 2050 and 2070, which looks like a plotting error, and do not obviously match the text |
| Fig. 1b legend | Range 3.3–5.8 °C with top classes apparently unused | |
| Extended Data Fig. 2 | Shows only the six original scenarios with unlabelled colour bars, and duplicates ED Fig. 1 | Should include Target scenarios or be removed |
| SVI–temperature (future) | Results: "26.00 °C versus 25.65 °C" | Sup. Fig. 12 / panel image values appear to differ (please verify) |
| Fig. 3a year labels | 2021 uses UHI = 2 °C (stated as the *average* UHI) | 2030/2050/2070 use UHI = 5 °C (the *maximum*). The "climate" signal in Fig. 3a (~5 °C by 2050) therefore mixes a UHI-assumption change with warming and is not comparable with Fig. 1 |
| "five health outcomes" | Methods/Intro | Abstract and main text report almost only all-cause; cause-specific results appear only in ED Fig. 1 |

**Requested action:** Reconcile every number. Add the Target scenarios to Table 1 with absolute canopy area and tree counts. Explain the 2021 versus future UHI parameterisation in the main text, because it affects how readers interpret the "climate versus land use" comparison that frames the paper.

### Major 7. The baseline-inequality section rests on weak evidence and oversells "compounding" exposure.

**Location:** Results, "Heat exposure and baseline inequalities" ("socially vulnerable neighborhoods face both higher ambient heat and reduced access to local biophysical cooling"); Fig. 2.

**Why it matters:**
- The SVI Q5–Q1 difference in July 2021 temperature is 0.17 °C. It comes from the ~1 km CHELSA product, which is coarser than most LSOAs and cannot resolve intra-urban heat. Statistical significance with n ≈ 4,800 does not make the difference substantively meaningful.
- Canopy is not monotonic either: Q4 has the lowest median and Q5 is "bc", not lower than Q3 (Fig. 2b). "Opposing gradients" is too strong.
- This section is the empirical premise for the equity argument, and readers in environmental justice will expect fine-scale exposure (LST, UCM output, or UTCI, which the authors already have at 1 m for targeting).

**Requested action:** Recompute baseline exposure inequality with the UCM baseline temperature field or the 1 m UTCI layer, and report resident-weighted differences. Soften "compounding" unless the finer-scale data support it.

## 5. Minor concerns

1. **Abstract, "socially and biologically vulnerable populations".** Biological vulnerability (age, comorbidity) is not modelled in the health function. Remove the phrase or model it.
2. **Abstract, "conventional proximity-based greening".** The authors explain that the proximity rule is an artefact of the InVEST Scenario Generator. Calling it "conventional" implies that it represents actual London planting practice, which is not demonstrated. Either show evidence that London planting clusters near existing canopy, or describe it as a modelling convention. This matters because the paper's causal story about inequity under "general greening" depends on it (Discussion: "One contributing factor ... may be the spatial logic of the proximity-based land-use scenario generator"). If the inequity is a property of the generator, the generalisation "without explicit targeting, greening may improve aggregate outcomes without proportionately benefiting..." needs qualification.
3. **Results, Fig. 6 text.** The "8.8 and 16.5 lower" contrasts sit on a range of roughly 30–50 deaths averted per million, which is modest. The income-deprivation curve is U-shaped (Fig. 6), so "Benefits were lower at high than at low-income deprivation" oversimplifies. Describe effect sizes in absolute terms (deaths per LSOA or per borough) as well.
4. **Results, "Citywide gains masked substantial borough-level variation".** Fig. 5 maps borough totals, which mostly reflect borough size and population. Show per-capita or per-hectare values so variation is interpretable.
5. **Discussion, "baseline health burdens that alter exposure–response translation".** The model uses a single RR and borough-level rates, so this mechanism is not represented and cannot explain the results. Rephrase.
6. **Discussion, "Our spatially explicit equity analysis (Fig. 7) provides a practical framework for identifying neighborhoods where targeted, bundled adaptation strategies are most needed."** This claim is not demonstrated. If retained, show a priority map that a borough officer could use (e.g., LSOAs with high vulnerability where even Target30 gives low benefit, i.e., where non-green interventions are needed). That would be a genuinely actionable output.
7. **Discussion, "Implications for urban planning".** The guidance is generic ("cities could pair canopy expansion goals with metrics of vulnerability-weighted cooling"). Make it London-specific: how the findings bear on the London Urban Forest Plan, the GLA Climate Risk Map, Urban Greening Factor (London Plan G5), borough tree strategies, the Mayor's Green and Resilient Spaces Fund, and the 2050 canopy target. A one-paragraph or tabular "what a borough should do" summary would help a health audience.
8. **Generalisability.** "the framework provides a transferable template" (Intro and Discussion) needs specifics: which inputs are London-specific (GiGL tree inventory, Colouring Britain footprints, ONS mortality), what data-poor cities could substitute, and how the conclusions might differ in cities where deprived areas are peripheral and greener (e.g., many European cities with outer-ring social housing) or in hot, AC-saturated climates.
9. **TreeRisk.** It is described as "−6%" canopy loss, but trees are converted to grassland "without immediate replacement." Clarify the fraction of GiGL trees flagged, which species, and whether the GiGL inventory (mainly street and park trees) covers private-garden trees, a large share of London's canopy.
10. **Self-harm and mental-disorder mortality.** Results (0.3–2.4 deaths) are tiny and highly uncertain. Presenting them alongside all-cause in ED Fig. 1 invites over-interpretation. Mental-health benefits of greening operate largely through non-heat pathways; the authors note this, but they also cite their own mental-health work (refs 9, 42) as support. Consider removing these outcomes or framing them as illustrative.
11. **Language.** Mixed US/UK spelling ("neighborhood" vs "behavioural"; "labor" vs "labour"). Pick one per journal style. There is a double period after "Green30. ." (Results, Fig. 4 paragraph) and ".." at the end of the Fig. 4 caption. "Stanford InVEST suite" should read "Natural Capital Alliance InVEST" for consistency with ref. 36. Refs 45 and 53 are duplicates. Ref. 5 (Bowler 2010 BMC Public Health) is cited for heat vulnerability of older adults, which it does not address; please check.
12. **Methods, LULC.** "this dataset has the same high spatial resolution" is ambiguous (same as what?). UKCEH LCM is 10 m, while the UCM run is described at 10 m, the work model at 30 m and population at 100 m. Clarify the resolution chain.
13. **Methods, UHI.** The UHI values come from a "global study conducted by Yale" (URL only). Cite the peer-reviewed source and justify 5 °C as the *mean* July daytime UHI for 2050.
14. **Sup. Fig. 9** is captioned as "mortality rates" but maps counts ("number").
15. **Discussion, "Our earlier studies"** (refs 9, 42). This is fine, but avoid implying that the mental-health co-benefit is quantified here.
16. **Abstract, "Holding climate constant, vegetation loss increased borough temperatures by up to 1.05 °C".** This is a *mean across boroughs* for an extreme counterfactual, not an "up to" per-borough maximum. Clarify, and say up front that AllBuilt is a non-realistic bounding case so that health readers do not read "1,110 additional deaths" as a plausible risk.
17. **Data availability.** The 1 m UTCI raster used for targeting has no stated source in either Methods or Data Availability. It should be described.

## 6. Narrative and structure audit

The Introduction poses, implicitly, four questions:
- **(Q1)** How much will London warm, and is exposure unequal?
- **(Q2)** How much can land use modulate heat under fixed climate?
- **(Q3)** What does that mean for health, energy and labour?
- **(Q4)** Are benefits equitably distributed?

| Results section | Intro question answered | Assessment |
|---|---|---|
| Heat exposure and baseline inequalities | Q1 | Answered, but the climate-projection material (Fig. 1) is standard and takes prime real estate; most could go to the SI. The inequality evidence is weak (Major 7). |
| Land-use choices modify cooling and co-benefits | Q2, Q3 | Answered. The Target scenarios appear here (Figs 3–4) **before** they are introduced or motivated; the reader meets "Target10" in Fig. 3 with no definition in Table 1 or the text. Introduce targeting in the Intro and Table 1. |
| General greening yields uneven benefits | Q4 | Answered, with the caveats in Major 3 and Minor 2. |
| Targeting redistributes health benefits | Not posed in the Intro | The paper's key result is never set up as a question. The Intro asks "whether urban greening systematically reduces heat-related inequities", not "does *allocation* matter, and at what efficiency cost?" Rewrite the final Intro paragraphs to pose the allocation question explicitly, with the relevant literature (optimisation, Tree Equity, McDonald et al.), so the Target analysis reads as the climax and not an add-on. |

**Missing:**
- Any analysis of costs, feasibility or time to maturity.
- A non-circular equity validation.
- Engagement with green gentrification and procedural justice (community co-design is mentioned only for aftercare).
- Equity analysis of the energy and work outcomes.
- A sensitivity/uncertainty section beyond exposure–response coefficients. The UCM parameters, UHI assumptions and crown-size assumptions drive everything, yet the "uncertainty intervals" reflect β only.

**Redundant or misplaced:**
- Extended Data Fig. 2 duplicates ED Fig. 1 for an outdated scenario set.
- The Discussion's first three paragraphs restate the Results ("Our results yield three overarching insights").
- The WHO/IPCC/Met Office contextual material is repeated almost verbatim between the Introduction and Discussion (40.3 °C record; IPCC AR6 WGII).
- The Discussion's "Co-benefits" section relies on literature-based plausibility rather than interpretation of the authors' own results.

**Storyline recommendation:** Organise the paper around the matched-allocation experiment:
1. Current canopy and fine-scale heat are unequally distributed.
2. Under policy-relevant targets (10%, plus stylised 20/30%), allocation rules change both efficiency and distribution.
3. Decompose why (density, distance to canopy, heat, vulnerability).
4. Show the trade-offs (older-adult neighbourhoods, feasibility, cost, gentrification).
5. Give London-specific guidance.

Move most of the climate projection, AllBuilt and energy/work material to Extended Data.

**Abstract–Results–Discussion consistency:** The Abstract's "0.37 °C" is the Green30 value from an earlier analysis, while the maximum is Target30 at −0.38 °C. The Abstract's emphasis on energy and work co-benefits ("demonstrating benefits beyond mortality reduction") is not matched by equivalent depth in the Results. The Discussion's closing paragraph omits the ≥75 trade-off found under targeting. The Intro promises "preventable heat-related mortality across multiple causes", yet the main text reports all-cause only.

## 7. Figures and presentation

- **Fig. 1.** Panel a is standard and could move to Extended Data. Panel b has a legend range (to 5.8 °C) that is mostly unused, and its inset histograms appear duplicated across 2050 and 2070. The SSP585 maps live in the SI, which is fine. For a health audience, show population-weighted rather than unweighted borough means.
- **Fig. 2.** Add a y-axis unit to panel a (°C is present in one version and missing in another). Effect sizes (Q5–Q1 difference with 95% CI) would be more informative than letter groupings. Consider replacing the CHELSA field (Major 7).
- **Fig. 3.** Panel a's caption explains that year labels are "assumed reference-temperature/maximum UHI settings ... not projections derived directly from Fig. 1." That caveat is crucial but buried; the baseline bars (~2, ~5, ~8 °C) will be misread as climate projections. Consider dropping panel a. Panel b lacks uncertainty and lacks the absolute canopy added for each scenario; adding canopy km² on a secondary axis would let readers see cooling per unit canopy, and therefore the allocation effect, at a glance.
- **Fig. 4.** Clear and well designed. Add a cooling-per-km²-canopy or per-1,000-trees panel. Label AllBuilt visually as a bounding counterfactual (e.g., hatched or separated) so health readers do not read −1,110 as a plausible risk.
- **Fig. 5.** Borough totals hide per-capita variation. Use per-100,000 residents for health and per-hectare for energy. The shared pseudo-log scale makes Green10 and Target10 indistinguishable; consider difference maps (Target − Green) at each tier, which would directly show where allocation matters.
- **Fig. 6.** Show resident-weighted fits and an indication of LSOA counts at the tails. Univariable GAMs on collinear predictors invite readers to treat each panel as independent evidence; say explicitly that they are not. Overlaying Green30 and Target30 in the same panel would be more informative than splitting them across Fig. 6 and ED Fig. 3.
- **Fig. 7.** Visually effective, but panels c and d foreground the circular metric. Panel b shows that Target30 leaves large parts of outer London (Bromley, Havering, Richmond, Kingston) with essentially no benefit; this should be discussed in terms of political feasibility and proportionate universalism. Consider adding a panel on an independent vulnerability measure.
- **Extended Data Fig. 1.** The current version (with Target scenarios and CIs) is good. An older version in the source files shows different numbers (e.g., Green30 377 vs 387; Green10 161 vs 135; AllBuilt −1,078 vs −1,110). Please confirm that the final numbers propagate everywhere.
- **Extended Data Fig. 2.** Outdated: no Target scenarios, unlabelled units, and a rotated-label layout. Remove it or update it.
- **Table 1.** Add the Target scenarios, absolute canopy area (km²), number of trees, allocation rule and feasibility status.
- **Captions.** These are generally thorough, sometimes excessively so (Fig. 6 and ED Fig. 3 run to about 200 words of model detail). Move statistical detail to Methods and keep captions interpretive for a broad audience.
- **Readability for a health audience.** Jargon such as "T_WM," "UCM," "scaled-t GAMs fitted by REML," "pseudo-logarithmic," "LSOA" and "SSP370" appears without lay framing. A short schematic of the modelling chain (scenario → UCM temperature → exposure–response → outcomes → equity analysis) as Fig. 1 would help greatly, and the current Methods equations could be replaced by a box.

## 8. Recommendation

**Major revision.** On balance this sits at the border with "Reject, resubmission encouraged." The research question is important and the matched-allocation design is promising. However, the headline equity result is largely circular. The ambitious scenarios are not shown to be feasible and carry no costing or maturation timeline. The ≥75 trade-off and the drivers of Target's efficiency gain are not resolved. The absolute health numbers need benchmarking. Internal inconsistencies must be fixed. Addressing Majors 1–4 would substantially change the paper's framing and probably its conclusions about what "equitable" allocation means.

## 9. Confidential comments to the editor

The topic fits Nature Health well and the team is credible, with an established InVEST/natural-capital track record. My central reservation is that the paper's headline equity claim (45.5% → 4.5%) is close to tautological, because the targeting algorithm optimises on the same SVI used to evaluate it. The authors acknowledge this only in a Supplementary caption. The most novel, non-circular finding is that equal-area allocation away from existing canopy raises aggregate benefit, but it is not decomposed or foregrounded. Feasibility (about 1.8 million new street trees for Target30) and gentrification are not addressed, and the absolute mortality magnitudes look high relative to UKHSA/ONS benchmarks. I defer to the methods panel on the last point, but it should be resolved before any public-facing numbers are accepted. Numerous inconsistencies between abstract, text and figures suggest the manuscript was revised in haste. I would be willing to review a revised version.
