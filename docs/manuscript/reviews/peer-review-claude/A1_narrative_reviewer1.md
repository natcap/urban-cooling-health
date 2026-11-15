# Referee Report: Reviewer #1 (narrative, framing, claims and equity)

**Manuscript:** "Urban greening strategies reshape heat-health benefits and equity" (London)
**Journal:** Nature Health
**Reviewer expertise:** environmental epidemiology; heat-related mortality; climate adaptation for health; health equity
**Remit this round:** the full narrative. That covers significance, how the health question is framed, logic and flow, whether the evidence supports the claims, internal consistency, figures and captions, limitations, and the equity argument. I comment on methods only where they weaken the narrative. A separate panel is reviewing methods in depth.

---

## 1. Summary of the submission

The authors combine land-use/land-cover (LULC) scenarios for Greater London with the InVEST Urban Cooling Model (UCM). They then apply exposure–response functions to estimate three things: July building energy cost, heavy-work capacity (WBGT-based) and heat-related mortality (all-cause plus four cause groups). The nine counterfactual scenarios are:
- AllBuilt (all vegetation converted to impervious surface)
- TreeRisk (loss of climate-risked trees)
- TreeOpp (planting on opportunity land)
- three proximity-based canopy expansions, Green10/20/30, generated with the InVEST Scenario Generator
- three "vulnerability-targeted" street-tree scenarios, Target10/20/30, which rank OSM street-side planting points by normalised UTCI × SVI

Each scenario is compared against baseline land cover under a single fixed "mid-century" UCM setting: reference temperature 25 °C and maximum UHI 5 °C. Separately, the authors present WorldClim/CHELSA projections of July temperature for London's boroughs (Fig. 1) and LSOA-level gradients in baseline temperature and canopy by a four-indicator SVI (Fig. 2).

Headline claims (Abstract):
- Vegetation loss warms boroughs by "up to 1.05 °C" and canopy expansion cools them by "up to 0.37 °C".
- Greening averts "135–496 deaths" per year, while vegetation removal produces "up to 1,110 additional deaths per year".
- Greening avoids energy costs of up to £22.4 M per month and raises heavy-work capacity by up to 1.42 percentage points.
- Under proximity-based ("conventional") greening, per-resident benefits are lower in neighbourhoods with more income deprivation and larger ethnic-minority shares.
- Targeted planting reverses these gradients: compared with Green30, Target30 cuts the share of high-SVI LSOAs receiving "low" benefits from 45.5% to 4.5%.

The Discussion recommends equity-informed canopy targets, vulnerability-weighted metrics and bundled interventions.

## 2. Overall assessment: novelty, significance, fit for Nature Health

The question matters and suits Nature Health well. The central idea is that *where* trees are planted, and not only how many, decides who gets the health protection. That message is useful to the many cities, London included, that have adopted canopy-cover targets. The paired Green-versus-Target design is the most original element of the paper. The multi-outcome framing is attractive, and so is the commitment to open code.

In its current form, however, the manuscript reads mainly as an urban-climate and ecosystem-services modelling study with a health-impact module attached. It does not yet read as a health paper. Four problems stand in the way of publication in Nature Health:

1. **The headline mortality numbers do not look epidemiologically valid.** Everything in Methods, figures and SI points the same way. A July daytime temperature change appears to have been multiplied by *annual all-cause deaths from 2021* through a threshold-free, log-linear RR applied all year. If so, the absolute "deaths averted" figures are likely inflated by up to an order of magnitude. They also look implausible next to published estimates of London's total heat-attributable mortality (Major 1).
2. **The equity analysis cannot speak to differential health vulnerability.** Borough-level all-age mortality is spread to pixels in proportion to population, so differences in per-resident benefit between LSOAs within a borough reflect only ΔT. Age structure, deprivation-related susceptibility and effect modification are absent (Major 3). This is a serious gap for a health-equity claim.
3. **The Target-versus-Green comparison is partly circular, and it is not equal on everything except allocation.** Target ranks sites on SVI, which is also the outcome stratifier. It plants street trees in dense, populous, hot areas. It also produces larger total benefits (Majors 4–5).
4. **The paper's own finding most relevant to heat mortality is missing from the Abstract and contradicted in the conclusion.** Targeting *doubles* the share of older-adult neighbourhoods receiving low benefits (28.7% → 59.0%) (Major 6).

I also found several internal inconsistencies between the Abstract, text, figures and Methods (Section 6). Some suggest that the text was not updated after figures were regenerated.

With substantial reanalysis and reframing this could become a strong, policy-relevant Nature Health paper. Without those changes, the absolute health claims are not supported.

## 3. Strengths

1. **A clear, policy-relevant counterfactual design.** Holding climate fixed and varying land use is a sensible way to separate a lever that planners can actually pull (Methods: "Green and Target scenarios were matched on realized added-canopy area to isolate the influence of spatial allocation").
2. **The scenarios are tied to London policy.** The +10% relative canopy target in the London Environment Strategy anchors Green10. TreeRisk usefully reminds readers that existing canopy is itself vulnerable to climate.
3. **The equity question is the right one.** Testing whether proximity-based expansion reproduces existing canopy inequities, and whether explicit targeting changes the result, addresses a real gap.
4. **Captions have been tightened in places.** They state "Percentages describe neighborhoods, not residents", "Age indicators describe neighborhood composition, not age-specific mortality", "These intervals represent exposure–response uncertainty only", and "SVI is related to the targeting design and is not an independent validation measure" (Supplementary Fig. S7). This candour is welcome. It now needs to reach the Abstract and Discussion.
5. **Borough-level heterogeneity is reported honestly.** The text acknowledges that Target30 beats Green30 in only 22 of 33 boroughs ("Targeting therefore increased aggregate benefits without improving every borough's outcome").
6. **Code is public** (GitHub), and figures are generally clean and modern (Figs 4, 5, 7; Extended Data Fig. 1).

## 4. Major concerns

### Major 1. Annual "deaths averted" appear to come from applying a July temperature change to all annual deaths, which probably inflates the headline numbers greatly

**Location:** Abstract ("greening scenarios averting 135–496 deaths and vegetation removal producing up to 1,110 additional deaths per year"); Results ("Target30 averted 496 deaths (95% … 347–640), compared with 387 (272–500) under Green30"); Methods, Health outcomes ("We quantified the projected excess mortality attributable to summer (July) temperature increases … D₀(x,y) = m(x,y) × pop(x,y) … deaths registered in 2021"); Fig. 4c axis "Annual deaths"; Sup. Fig. 9 (borough *counts* of 2021 deaths, mislabelled "rates").

**Why it matters:** The numbers back-calculate almost exactly to the following:

> annual 2021 all-cause deaths in London (≈55,000–60,000; consistent with Sup. Fig. 9) × [1 − exp(−ln(1.02) × ΔT)], with ΔT = the July daytime UCM difference (Fig. 3b: −0.34 °C for Green30 gives ≈380; −0.38 °C for Target30 gives ≈430–500; +1.05 °C for AllBuilt gives ≈1,100).

In other words, the model treats one July daytime cooling as if it acted on every death in the calendar year. The RR is also applied log-linearly across the whole temperature range, with no minimum-mortality temperature and no restriction to warm-season or hot days. Heat-attributable mortality in temperate cities is concentrated on a limited number of hot days. For context, UKHSA's estimate of heat-associated deaths for *all of England* across the extreme 2022 summer was about 2,985 (cited by the authors as ref. 28). London-specific analyses (ref. 29 and the wider multi-country literature; verify) put London's typical annual heat-attributable mortality at a few hundred deaths, with higher figures in extreme years.

A Target30 scenario that averts 496 deaths a year would therefore prevent more deaths than London plausibly suffers from heat in a typical year, and it would do so through 0.38 °C of July daytime cooling. Three further problems compound this:
- 2021 was a COVID-affected year, so registered deaths were unusually high.
- The cause-specific RRs for circulatory and respiratory disease (ref. 61, Bunker et al.) come from studies in older adults but are applied to all ages.
- The "mental and behavioural disorders" RR is drawn from a review largely of mental-health morbidity outcomes (ref. 44; verify).

These choices are central to the narrative, not technical footnotes. Abstract, Fig. 4, Fig. 5, Extended Data Fig. 1 and all the equity metrics (deaths averted per million) inherit them.

**Requested action:**
(a) Rebuild the health-impact assessment along accepted heat-HIA practice:
- baseline daily mortality for the warm season, averaged over several years and excluding COVID-distorted 2020–21;
- a London- or UK-specific temperature–mortality function with a threshold or minimum-mortality temperature (for example, a published distributed-lag non-linear model for London; verify the specific source);
- apply the land-use ΔT on the relevant days, ideally to daily mean temperature, or justify using daytime Tmax.

(b) Benchmark the new baseline heat-attributable burden against published London and England estimates (UKHSA monitoring; Simpson et al. 2024), and report greening benefits as a *fraction* of that burden.

(c) If the authors keep the current approach, they should relabel the outputs as an index ("modelled mortality-equivalent units"), drop "annual deaths", and remove absolute death counts from the Abstract.

(d) Correct Sup. Fig. 9, which shows counts, not rates.

### Major 2. Causal and certainty language ("deaths averted", "preventable") goes beyond a model chain with uncertainty from exposure–response only

**Location:** Abstract ("averting", "producing up to 1,110 additional deaths per year"); Introduction ("preventable heat-related mortality"); Discussion ("greening-oriented scenarios … avert heat-related mortality, consistent with empirical evidence"; "resulting in larger reductions in heat-related mortality among populations with higher social vulnerability"); Methods ("95% exposure–response uncertainty interval").

**Why it matters:** The estimates pass through a chain of unvalidated steps: a rule-based LULC scenario, then the UCM with literature parameters and no London validation shown, then a fixed 25/5 °C setting, then an ERF from a global meta-analysis, then population allocation. The 95% intervals (e.g., 347–640) cover only the ERF coefficient. The authors admit this in Methods, but the interval is presented in Results as if it described uncertainty in the estimate as a whole. UCM structural and parameter uncertainty is not propagated. Neither is uncertainty from the scenario generator, the reference temperature, the UHI magnitude (which Fig. 3a shows is itself a dominant driver), population or baseline mortality. This matters especially because the paper ranks scenarios that differ by 10–30% (Green30 387 versus Target30 496). Several earlier figure versions embedded in the submission (see Section 7) had UCM-based error bars that were later dropped.

**Requested action:**
- Use "modelled", "estimated" or "potentially avertable" consistently in the Abstract, Results and Discussion.
- Present intervals as "exposure–response-only" wherever they appear.
- Add at least a one-at-a-time or global sensitivity analysis on the UCM parameters (cooling distance, Kc, shade, UHI magnitude, reference temperature) and on the ERF choice.
- Show whether the **ranking** Target > Green and the **sign** of the equity gradients survive that sensitivity analysis. That is the decision-relevant inference.
- Report UCM validation against London observations (e.g., air-temperature networks or LST; Zawadzka et al. 2021 is cited but its relevance is not shown).

### Major 3. The equity analysis cannot capture differential susceptibility, which is the core of heat-health equity

**Location:** Fig. 6 caption ("Borough mortality is allocated to pixels in proportion to population"); Methods ("population-weighted allocation of deaths registered in 2021"); Results ("Benefits were lower at high than at low-income deprivation and ethnic-minority population share"); Discussion ("neighborhoods with higher socioeconomic vulnerability tend to show weaker marginal benefits").

**Why it matters:** Each LSOA receives its borough's all-age crude mortality rate in proportion to its population. Within a borough, deaths averted per million is therefore a monotone function of local ΔT. Across boroughs, it also reflects differences in crude mortality, which are driven mainly by age structure. Two consequences follow:
- (i) The "equity" gradients in Figs 6–7 are essentially maps of where the UCM produces cooling, multiplied by the borough's crude death rate. They are not estimates of health benefit for vulnerable people.
- (ii) The finding that older-population neighbourhoods benefit more under Green probably reflects outer boroughs with high crude mortality (older populations) and more existing canopy for proximity-based expansion to extend. It is not evidence of protection for older people.

The heat-health equity literature is concerned precisely with effect modification by age, deprivation, housing, pre-existing disease and social isolation. The authors' Introduction makes the same point ("Older adults, young children, and people with pre-existing … conditions … are consistently more vulnerable"). None of that enters the model. The current analysis addresses equity of *exposure reduction*, not equity of *health benefit*.

**Requested action:**
- At minimum, use age-specific (≥65/≥75) and, if possible, deprivation-stratified baseline mortality at MSOA/LSOA level (ONS small-area mortality, or modelled age-specific rates applied to LSOA age structure).
- Consider age- and deprivation-specific RRs from UK studies (verify sources).
- Alternatively, reframe Figs 6–7 explicitly as "population-weighted cooling" or "exposure-reduction equity" and remove health-equity language from the Abstract and title.
- Report person-weighted (resident-weighted) metrics alongside neighbourhood counts.
- Add the obvious decomposition: ΔT per LSOA by SVI tertile, separately from the mortality translation.

### Major 4. The Target-versus-Green comparison is partly circular and is not "allocation only"

**Location:** Abstract ("Vulnerability-targeted planting redistributed benefits toward neighborhoods experiencing jointly elevated heat exposure and social vulnerability … Target30 reduced the share of high-SVI neighborhoods receiving relatively low health benefits from 45.5% to 4.5%"); SI step 3 ("Priority score = UTCI_norm × SVI"); Supplementary Fig. S7 caption ("SVI is related to the targeting design and is not an independent validation measure").

**Why it matters:** Target30 plants trees *in proportion to SVI*. It then follows almost by construction that high-SVI LSOAs receive more cooling and therefore more modelled "benefit". The 45.5% → 4.5% headline is thus mostly a check that the algorithm did what it was designed to do. It is not an empirical finding about equity. Several other features confound the comparison:

- **Different typology of planting.** Target uses OSM street-side points at 5 m spacing within 2 m of buildings and away from existing canopy. Green converts "selected grassland, woodland, suburban and urban" pixels next to existing canopy. Sup. Table 2 gives deciduous and coniferous woodland a shade value of 0 but "Tree Cover" a value of 1. If Green converts *woodland* pixels to "Tree Cover", part of Green's modelled cooling could be a reclassification artefact rather than new trees. If Green does not, the authors should state this. Either way the two strategies differ in more than location.
- **Different exposure metric used for siting.** Target sites by 1 m UTCI at 14:00, from an unstated source, while the outcome model is the UCM. Planting where people and buildings are dense mechanically raises population-weighted benefit.
- **Matching on canopy area is asserted but not shown.** Table 1 omits Target scenarios entirely. Target tree counts are approximate ("~565,000 … 1,129,000 … 1,819,000 tree sites"). Fig. 3b shows Target10 at −0.14 °C versus Green10 at −0.12 °C, which needs explaining if canopy is matched.
- **Feasibility is not discussed.** Adding about 1.8 million street-side trees is a very large number by any comparison with London's current street-tree stock (verify the current count with GLA/borough data). Sidewalk width, underground utilities, sightlines and maintenance would bind long before this. Calling Target "feasible" (Abstract: "feasible planting strategies") needs evidence.
- **"Conventional" is unsupported.** The Abstract calls proximity-based greening "conventional", but no evidence shows that London's actual planting follows a nearest-edge rule.

**Requested action:**
- (a) Add Target scenarios to Table 1 with realised added canopy (km², % relative) for every scenario.
- (b) Evaluate equity using indicators *not* used in targeting. Examples: IMD domains not in the SVI, health deprivation, prevalence of chronic disease, social housing, overheating-prone housing.
- (c) Add at least one *heat-only* targeted scenario (UTCI or population-weighted heat, no SVI) and one *random street-tree* scenario. This would separate (i) the street-tree typology effect, (ii) the heat-targeting effect and (iii) the SVI-targeting effect.
- (d) Compare distributions at *matched total benefit* as well as matched canopy. Target30's benefit is about 28% larger in total, so with fixed Green30 thresholds most LSOAs move up a class regardless.
- (e) Report what happens in low-SVI LSOAs. Extended Data Fig. 3 shows Target30 benefits near zero at low SVI (≈10–15 per million versus ≈45–50 under Green30), and Fig. 7b/S7 show large grey areas. Readers need to see who loses.
- (f) Drop or support "feasible" and "conventional".

### Major 5. The narrative mixes two climate "signals" that are not linked, and the warming comparison overstates both climate change and the land-use contrast

**Location:** Introduction ("projected to experience substantial increases in summer heat (≈2.7 °C by the 2050s)"); Results ("differences of approximately 3.9–4.2 °C at mid-century"); Fig. 1 caption ("CHELSA for July 2021 (baseline year) and WorldClim for future periods"); Fig. 3 caption ("Year labels … denote assumed reference-temperature/maximum UHI settings of 22/2, 22/5, 25/5 and 28/5 °C … not projections derived directly from Fig. 1"); Sup. Fig. 2 (WorldClim SSP370 2021–2040 ≈ 24.6 °C).

**Why it matters:**
- (i) Fig. 1 compares a single observed year (CHELSA July 2021, 21.9 °C) with WorldClim future climatologies. Sup. Fig. 2 shows WorldClim already at about 24.6 °C for 2021–2040. That implies roughly 2.7–2.9 °C of "warming" between 2021 and the 2030s, which is implausible and points to a product or baseline offset. The 3.8–4.2 °C mid-century figure therefore conflates real climate change with differences between data products. It also conflicts with the ≈2.7 °C quoted in the Introduction.
- (ii) The SSP585 maps use a different (WorldClim) baseline from the SSP370 maps (CHELSA), per the caption, so the two pathways are not comparable.
- (iii) Fig. 3a's "2030" bar (+2.2 °C) comes entirely from the *assumed* tripling of UHI from 2 °C to 5 °C at the same 22 °C reference temperature. The headline sentence "Although smaller in magnitude than the projected climate signal" therefore depends on an untested assumption about UHI amplification, which Methods concedes is weakly evidenced for London.
- (iv) None of the SSP or GCM information propagates into the health or equity results, which all use one 25/5 °C setting. Fig. 1 is therefore narrative context, not an input. Sup. Fig. 12 also names a different GCM set (ACCESS-CM2, IPSL, MRI) from Methods (EC-Earth3-Veg, MRI, IPSL).

**Requested action:**
- Use a consistent delta approach: future minus *matched* historical climatology from the same product and GCMs (or UKCP18, the UK's national projections).
- Reconcile the warming figures with the Introduction.
- Either link the UCM reference temperature to the SSP projections and show how health results scale between 2050 and 2070 and between SSP370 and SSP585, or state clearly that land-use effects are evaluated at one stylised setting.
- Remove or qualify "smaller than the projected climate signal".
- Fix the GCM inconsistency.

### Major 6. The trade-off for older adults is central to heat-mortality policy, yet it is left out of the Abstract and contradicted in the concluding paragraph

**Location:** Results ("among neighborhoods with the highest proportions of residents aged ≥75 years, the low-benefit share increased from 28.7% to 59.0%"); Extended Data Fig. 3 (≥75 panel: Target30 benefits fall from ≈70 to ≈15 per million across the distribution); concluding paragraph ("with muted benefits in more socioeconomically vulnerable areas and stronger mortality reductions in areas with more older adults"); Abstract (silent).

**Why it matters:** Older adults carry most heat-related deaths in the UK and Europe. A Nature Health reader would see "targeted greening halves protection where older people live" as at least as important as the SVI result. Yet the Abstract reports only the favourable SVI result. The concluding paragraph describes the *Green* pattern ("stronger mortality reductions in areas with more older adults") as if it were the general finding, without saying that Target reverses it. This is a fairness issue in how the two strategies are presented. It also exposes a design problem: the four-indicator SVI includes % aged ≥75, but that indicator correlates negatively with deprivation and ethnic-minority share (Sup. Fig. 10), so an SVI-weighted score down-weights older-adult neighbourhoods.

**Requested action:**
- State the ≥75 trade-off in the Abstract and in the conclusion.
- Explain how the SVI combines indicators that are negatively correlated with one another.
- Test a targeting variant that includes age-weighted heat-mortality risk (for example, priority = heat × expected heat deaths).
- Discuss explicitly whether an equity objective based on social deprivation conflicts with one based on mortality-risk reduction.

### Major 7. The Abstract's quantitative claims do not match the figures in several places

**Location:** Abstract versus Fig. 3b and Results.
- "canopy expansion reduced them by **up to 0.37 °C**". Fig. 3b (final version) shows Green30 at **−0.34** and Target30 at **−0.38**. The 0.37 value matches an earlier Green-only figure (Green30 −0.37) embedded in the file. Results says "−0.37 °C **on average** across boroughs", so the Abstract ("up to") and Results ("on average") also disagree in meaning.
- "vegetation loss increased borough temperatures **by up to 1.05 °C**". 1.05 is the *borough-mean* AllBuilt effect, not a maximum. "Vegetation loss" also blurs the extreme AllBuilt counterfactual (which Methods calls "unrealistic") with the realistic TreeRisk (+0.10 °C, −105 deaths).
- "135–496 deaths" combines Green10 (lowest) and Target30 (highest). That is fine, but the Abstract should say which scenarios are meant.
- "**up to 1,110** additional deaths". This is the AllBuilt point estimate, not an upper bound.
- "high-SVI neighborhoods" is undefined in the Abstract. It means the top SVI *tertile* (n = 1,612), whereas Fig. 2 uses quintiles.

**Requested action:** Correct the values. Replace "up to" with "on average across boroughs" where that is meant. Identify AllBuilt as an extreme bounding counterfactual. Add the TreeRisk result, which is the policy-relevant loss scenario. Define the high-SVI group.

### Major 8. The health framing is thin for Nature Health

**Location:** Introduction paragraphs 3–4; Discussion "Co-benefits link cooling to energy saving, livelihoods, and health protection".

**Why it matters:** Energy and labour co-benefits receive as much space as health. Both are far from health endpoints. The energy model applies per-°C coefficients from a Drax report to building *footprint* area, not floor area. The work-capacity index is area-weighted, not worker-weighted, which the authors acknowledge. The Discussion calls the focus on heavy outdoor work "a conservative … estimate". It is the opposite: heavy work has the steepest response, so it yields the *largest* modelled change, and in London it applies to a small share of workers.

At the same time, several core heat-health issues are missing:
- night-time and indoor exposure (London's overheating burden is substantially indoor and nocturnal, in top-floor flats and care homes);
- the time–activity locations of vulnerable people;
- the established health-impact-assessment literature on urban trees and heat mortality.

Two literatures in particular are absent. One is the European health impact assessment estimating how much UHI-attributable mortality 30% tree cover could prevent across European cities (Iungman et al., Lancet 2023; verify). The other is the multi-country temperature–mortality and UHI-mortality literature (e.g., MCC Collaborative Research Network studies; Heaviside and colleagues' London/West Midlands UHI-mortality work; verify). Engaging with these would both provide benchmarks (Major 1) and sharpen the claim to novelty. Iungman et al. is very close in design, so the authors must explain what this paper adds: the targeting comparison and the equity lens.

**Requested action:**
- Restructure the Introduction around the heat-health problem in London: burden, who dies, where, and the evidence on trees and heat mortality.
- Position the paper against existing heat-HIA work on greening.
- Move energy and labour to a shorter co-benefits section or Extended Data.
- Correct the "conservative" claim.
- Add a limitations paragraph on daytime-only outdoor air temperature as the exposure, versus indoor/night-time exposure and personal exposure.

### Major 9. Discussion claims that go beyond the evidence

**Location and requested changes:**
- "alternative land-use pathways shift local heat exposure by **up to ±1 °C**". This is asymmetric: +1.05 for an unrealistic total loss of vegetation versus −0.38 for the best greening. Rephrase.
- "**resulting in larger reductions in heat-related mortality among populations with higher social vulnerability**" (Discussion). These are modelled per-resident ΔT-driven quantities with uniform borough rates (Major 3). Rephrase as modelled neighbourhood-level benefit.
- "**Our spatially explicit equity analysis (Fig. 7) provides a practical framework for identifying neighborhoods where targeted, bundled adaptation strategies are most needed**". No bundled strategies are modelled. Soften, or demonstrate.
- "**the approach is transferable**" and "**provides a transferable template**". This rests on London-specific data (GiGL, Colouring Britain, GLA canopy). Specify which components transfer.
- "**This structure strengthens inference regarding the relative performance of alternative land-use pathways**". Holding climate constant does not remove UCM or ERF structural error. Relative rankings can still be affected when scenarios differ in spatial pattern, as Green and Target do. Show this with sensitivity analysis (Major 2), or soften.
- "**our results likely understate the full suite of benefits or trade-offs**". Given Major 1, the mortality component is more likely *overstated*. Balance this sentence.
- The long paragraphs on stewardship and funding models (Discussion, "In practice…") are reasonable policy commentary, but nothing in the results supports them. Shorten them and cite the evidence they rely on.

## 5. Minor concerns

1. **Scenario count.** Methods says "we developed six LULC scenarios", Fig. 4 says "Nine scenarios (Table 1)", but Table 1 lists six. Add Target10/20/30 to Table 1.
2. **Outcome count.** Methods says "five health outcomes", but the Abstract and main figures show all-cause only. Cause-specific results (Extended Data Fig. 1) overlap with all-cause and must not be summed. State this.
3. **Promised outcomes not delivered.** UCM section: "co-benefits, such as energy saving, work productivity, and physical / mental health outcomes". Physical activity and mental-health (non-mortality) outcomes are not modelled. Remove.
4. **Temperature metric is described inconsistently.** "mean temperature of the warmest month" (Results), "Average maximum temperature of the warmest month" (Fig. 1 caption), "July 2021 temperature from the CHELSA maximum-temperature product" (Fig. 2), "July mean Tmax" (Sup. Fig. 2). Harmonise.
5. **Arithmetic in Results.** Results gives "differences of approximately 3.9–4.2 °C … 4.8–5.5 °C". From the Fig. 1 means: 25.7 − 21.9 = 3.8, 27.3 − 21.9 = 5.4 and 26.6 − 21.9 = 4.7. Check rounding or explain.
6. **Fig. 1b insets.** In the submitted composite Fig. 1 (image5), the 2050 and 2070 insets show identical counts (2, 4, 7, 10, 10). An embedded alternative version shows different counts (2, 5, 7, 13, 6 and 1, 2, 7, 9, 14), and the latter matches the text ("13 of 33 … exceed 4 °C"; "11 … exceed 5 °C" is still not obviously consistent). This looks like a plotting error. The 5.6–5.8 °C legend class appears unused.
7. **Fig. 2 and practical significance.** Median July temperatures differ by 0.17 °C between Q5 and Q1 (22.00 versus 21.83) at about 1 km product resolution. Statistical significance with n ≈ 970 per group is not the same as practical significance. The canopy gradient is not monotonic either: Q4 is lowest and Q5 is higher than Q4, with letters "c" and "bc". Temper "significantly warmer" and "opposing gradients" ("generally lower" is accurate, "compounding" is overstated).
8. **SVI construction.**
   - "at least one representative indicator from each of the four SVI themes", but the housing indicator (social housing) and LEP were then dropped. The index is still called "four-dimension". State exactly which indicators enter and with what weights and signs.
   - The SI says "tract-level SVI percentile", but tracts are a US unit. Clarify that LSOA is meant.
   - The CDC SVI is a US instrument. Justify it over the English Indices of Multiple Deprivation or the GLA Climate Risk Mapping indices, or show results are robust to them.
9. **LSOA counts.** 4,835 in most places but 4,831 in Sup. Fig. 12. Explain the exclusions.
10. **GAMs.** Univariable fits with pointwise CIs that ignore spatial autocorrelation (the authors note "not spatially adjusted"). The 90th–10th contrasts (e.g., −8.8 and −16.5 per million) are modest relative to a mean of about 45 per million. Report them as relative differences as well. Note that the income-deprivation curve is U-shaped (Fig. 6), with benefits recovering at the highest deprivation. "Benefits were lower at high … income deprivation" is therefore only partly true.
11. **Fig. 7 thresholds.** Tertile cut-points are fixed at Green30 values. Show a sensitivity analysis with scenario-specific tertiles or with absolute ΔT classes.
12. **Energy outcome.** "£22.4 million per month" is a July figure. Do not imply annual savings in the Discussion. Avoid "energy resilience" claims without peak-demand modelling. Footprint area times a per-m² coefficient ignores floor area and AC penetration in each building. Clarify the units (kWh/m²/°C per month) and where the 12–16% per °C comes from.
13. **Work capacity.** The WBGT derivation is not described (UCM air temperature plus RH? which RH?). The SI RH method assumes constant vapour pressure, which the authors note may overestimate drying. Its effect on WBGT and hence on the productivity results should be quantified.
14. **Citation accuracy (check these).**
    - Ref. 5 (Bowler et al. 2010, on health benefits of natural environments) is cited for "older adults … more vulnerable to heat-related illness".
    - Ref. 42 (Li et al., Nature Cities 2025, acute mental-health benefits of nature) is cited for variation in heat ERFs.
    - Ref. 58 (Zhu et al., under-five mortality in LMICs) is cited as the basis of the London adult HIA.
    - Ref. 11 (Bowler et al.) is used both for "green spaces can measurably cool urban areas by ~0.94 °C" (park versus non-park daytime difference, not citywide cooling) and in the Discussion.
    - Ref. 19 is incomplete ("The Lancet 0, (2025)").
    - Refs 45 and 53 are duplicates.
    - The "61,000 deaths in Europe" figure should specify summer 2022.
    - The "2016 London climate report" sentence has no citation.
15. **Tree-risk scenario.** Table 1 gives an absolute TCC of 18% for TreeRisk after a 6% loss, from a baseline of 19.56% (2022), which implies 18.4%. Report the baseline TCC of the model LULC itself; the GLA reclassification may change it. TreeRisk also mixes 2050 and 2070 and SSP370/585 risk, so specify which version is used in Figs 3–5.
16. **Data availability.** "will be made available upon publication" does not meet Nature Portfolio expectations. Deposit the processed scenario rasters and LSOA-level outputs in a repository (Zenodo, Figshare) at resubmission, and state the UTCI source used for targeting.
17. **Language and typos.** Results has a double full stop (". . These patterns"). There is a stray space before a full stop ("effects of vulnerability ."). The Fig. 4 caption ends with "..". The Fig. 1 caption mixes "(a)" and "b,". "The plausibility of our avoided energy-cost results are consistent" should read "is". "Stanford InVEST suite" should be "Natural Capital Project / Natural Capital Alliance InVEST". "Five-health outcomes" versus "multiple causes" is inconsistent. Mixed US/UK spelling (neighborhood/neighbourhood).
18. **Competing interests.** Several authors belong to the group that develops InVEST. That is not a financial conflict, but transparency would be served by noting the authors' role in developing the tool.

## 6. Internal-consistency audit table

| # | Claim | Where stated | Where supported | Consistent? | Note |
|---|---|---|---|---|---|
| 1 | Vegetation loss warms boroughs "by up to 1.05 °C" | Abstract | Fig. 3b (AllBuilt +1.05, borough mean) | Partly | Value matches, but it is a borough mean, not "up to". It is the extreme AllBuilt case; TreeRisk is +0.10. |
| 2 | Canopy expansion cools "by up to 0.37 °C" | Abstract; Results ("−0.37 °C on average") | Fig. 3b final: Green30 −0.34, Target30 −0.38 | **No** | 0.37 matches a superseded figure. "Up to" and "on average" also conflict. |
| 3 | Greening averts 135–496 deaths per year | Abstract; Results | Fig. 4c / ED Fig. 1a (Green10 135, Target30 496) | Yes (numerically) | Validity questioned (Major 1). The range spans Green and Target. |
| 4 | Vegetation removal: up to 1,110 additional deaths per year | Abstract | Fig. 4c (−1,110; interval ≈ −1,430 to −790) | Partly | Point estimate, not "up to". Extreme counterfactual. |
| 5 | Target30 496 (347–640) versus Green30 387 (272–500) | Results | Fig. 4c / ED Fig. 1a | Yes | Interval covers ERF only. |
| 6 | Avoided energy cost up to £22.4 M/month | Abstract; Results | Fig. 4a (Target30 +22.4) | Yes | July only. Green30 is +16.3. Superseded figure version showed +0.53 for Green30. |
| 7 | AllBuilt energy +£49.6 M/month | Results | Fig. 4a (−49.6) | Yes | — |
| 8 | Heavy-work capacity gain up to 1.42 pp; AllBuilt −6.36 pp | Abstract; Results | Fig. 4b | Yes | Superseded version showed Green30 +2.19 and AllBuilt −7.40. |
| 9 | Mean July temperature 21.9 → 25.7 (SSP370) / 26.1 (SSP585) by 2041–2060; 26.6 / 27.3 by 2061–2080 | Results | Fig. 1a | Yes | — |
| 10 | Differences of 3.9–4.2 °C (mid-century), 4.8–5.5 °C (late century) | Results | Fig. 1a means imply 3.8–4.2 and 4.7–5.4 | Minor mismatch | Probably rounding. |
| 11 | ≈2.7 °C summer warming by the 2050s | Introduction | Fig. 1 implies 3.8–4.2 °C | **No** | Different baselines and products (Major 5). |
| 12 | Mid-century July Tmax 25.5 °C (SSP370) / 25.9 °C (SSP585) | Methods | Sup. Fig. 2 (25.5 / 25.9); Fig. 1a (25.7 / 26.1) | **No** | Two different values for the same quantity. |
| 13 | 13 of 33 boroughs > 4 °C (2050); 11 > 5 °C (2070), SSP370 | Results | Fig. 1b insets (identical across periods in the final version) | **Unclear / likely error** | Insets appear duplicated. |
| 14 | GCMs: EC-Earth3-Veg, MRI-ESM2-0, IPSL-CM6A-LR | Methods | Sup. Fig. 12: ACCESS-CM2, IPSL, MRI | **No** | — |
| 15 | Q5 vs Q1 median July temperature 22.00 vs 21.83 °C; canopy 14.3% vs 17.9% | Results | Fig. 2a/b | Yes | Small effect. Canopy is non-monotonic (Q4 lowest). |
| 16 | SSP370 mid-century Q5 vs Q1 26.00 vs 25.65 °C | Results | Sup. Fig. 12 (image) | Approximately yes | n = 4,831 versus 4,835. |
| 17 | Target30 per-borough deaths averted 0.29–29.75; Newham largest | Results | Fig. 5c (no values shown) | Not verifiable | Fig. 5 shows counts with a pseudo-log scale; supply a table. |
| 18 | Target30 > Green30 in 22/33 boroughs (health, energy) and 21 (work) | Results | Not shown | Not verifiable | Add a supplementary table. |
| 19 | Green30 90th–10th: −8.8 (income), −16.5 (ethnic), +5.7 (≥75) per million | Results | Fig. 6 (visually plausible) | Plausible | Income curve is U-shaped. |
| 20 | SVI contrast +34.0 → +58.0 (Target10 → 30); −4.2 → −9.8 (Green) | Results | ED Fig. 3; Sup. Fig. S6-1 | Plausible | No table of contrasts. Provide one with CIs. |
| 21 | High-SVI low-benefit share 45.5% → 4.5%; high-benefit 25.1% → 84.6% | Abstract; Results | Fig. 7c | Yes | Circular by design (Major 4). |
| 22 | ≥75 low-benefit share 28.7% → 59.0% | Results | Fig. 7d | Yes | **Omitted from the Abstract.** The conclusion implies the opposite (Major 6). |
| 23 | "Stronger mortality reductions in areas with more older adults" | Conclusion | True only for Green (Fig. 6). Reversed for Target (ED Fig. 3, Fig. 7d) | **No** | — |
| 24 | Green and Target matched on realised added canopy | Methods | Table 1 lacks Target. Fig. 3b: Target10 −0.14 vs Green10 −0.12 | Not demonstrated | — |
| 25 | Nine scenarios (Table 1) | Fig. 4 caption | Table 1 lists six; Methods says "six LULC scenarios" | **No** | — |
| 26 | Five health outcomes | Methods | ED Fig. 1 (five panels) | Yes | Main text uses all-cause only. |
| 27 | Land-use shifts exposure by "±1 °C" | Discussion | Fig. 3b (+1.05 / −0.38) | **No** (asymmetric) | — |
| 28 | Heavy-work focus is "conservative" | Discussion | Logic of ERF | **No** | Heavy work has the largest response. |
| 29 | Baseline "mortality rates" | Methods; Sup. Fig. 9 | Sup. Fig. 9 shows counts ("number") | **No** | — |
| 30 | Fig. 3a "2030" warming of ~2.2 °C | Fig. 3a | Assumed UHI 2 → 5 °C at the same 22 °C reference | Internally consistent but misleading | Not a climate projection. |

## 7. Figures and presentation comments

- **Fig. 1.** Mixing CHELSA (single year) and WorldClim (climatology) inflates the warming signal (Major 5). The panel b insets appear duplicated across periods. The legend includes an unused class. The y-axis label "Temperature (°C)" should name the variable. Consider whether Fig. 1 earns a main-figure slot, since it does not feed the health analysis. It might be better as Extended Data, with the main figure space used for a health-burden benchmark.
- **Fig. 2.** Clear. Add n per quintile and effect sizes (median differences with CIs), not only Dunn letters. Note that the 0.17 °C difference is at about 1 km resolution.
- **Fig. 3.** Panel a's "Year" axis implies projections, but the caption says they are assumed UCM settings. Relabel the axis as "UCM setting (T_ref/UHI_max)" or remove the panel. Panel b has lost the uncertainty bars present in an earlier version. Restore a spread measure (borough IQR or range), because the text says "up to".
- **Fig. 4.** Clean. Interleaving Green and Target bars is good for comparison. Consider ordering by canopy level with Green and Target side by side and labelled more clearly. Panel c should state "exposure–response-only 95% UI" on the figure itself. Add TreeRisk to the narrative.
- **Fig. 5.** Absolute borough counts with a pseudo-log scale mainly reflect population and area. For an equity narrative, show **per-capita** rates (per 100,000) and ideally Target30 − Green30 difference maps. The colour bar labels (5.4, 1.1, …) are not intuitive. Supply a borough table in the SI.
- **Fig. 6.** Improved over the earlier version (full predictions rather than centred partial effects). Add the 10th/90th percentile markers used for the contrasts. Overlay Target30 on the same axes, or make a combined Green30/Target30 figure. The current split across Fig. 6, Sup. Fig. S6-1 and ED Fig. 3 forces readers to jump between documents for the central comparison.
- **Fig. 7.** The strongest visual in the paper. The bivariate legend should define tertile cut-points for SVI. Panel d should add the *low-vulnerability* groups so that losses are visible, and resident-weighted percentages. Note on the figure that SVI is a targeting input.
- **Extended Data Fig. 1.** Good. State clearly that panels b–e are subsets of panel a and not additive.
- **Extended Data Fig. 2.** Appears superseded. It omits Target scenarios, uses a reversed sign convention (red = additional deaths) relative to ED Fig. 1, has no unit labels, uses raw variable names ("Mental_disorder", "All_cause") and duplicates ED Fig. 1. Remove or update it.
- **Extended Data Fig. 3.** Informative. It shows that Target benefits collapse to near zero in low-SVI and high-≥75 neighbourhoods, a result the text should discuss (Majors 4, 6).
- **Superseded embedded figures.** The submission package (ms_figs) contains several earlier panels with values that differ from the final figures: Green30 −0.37 °C; energy +0.53 £M; work +2.19 pp; Green-only Fig. 5 and Fig. 6 versions. The Abstract's 0.37 °C appears to derive from one of these. Please make sure only final versions are embedded, and recheck every number in the text against the final figures.
- **Table 1.** Add Target scenarios, realised added canopy (km²) and the number of pixels or trees for every scenario.
- **Table 2.** State the population and age group each RR was derived for, and the temperature metric and lag.
- **SI.** Several Supplementary Figures are mislabelled or reference the wrong figure (e.g., "The LULC categories and color scheme are the same as in Sup. Fig. 2", where Sup. Fig. 2 is a temperature boxplot). "Supplementary Fig. S6-1" and "S7" use a numbering scheme that differs from Sup. Figs 1–12. The SI step 4 contains a duplicated sentence fragment (". , using average crown-size assumptions…").

## 8. Recommendation

**Major revision.** The core question and the Green-versus-Target design have real value for Nature Health. However, the headline health quantities need to be re-derived using accepted heat-HIA practice, and the equity claims need to rest on non-circular, susceptibility-aware analysis. I would be glad to see a revised version. If, after re-estimation, the absolute mortality effects shrink substantially, the paper's main contribution could still stand on the *relative* and *distributional* findings, provided these survive the sensitivity analyses requested in Majors 2–4.

## 9. Confidential comments to the editor

This is a topical and potentially influential paper with a clear policy hook: canopy targets do not guarantee equitable protection. My main worry is that the absolute mortality numbers in the Abstract are almost certainly overstated. They appear to apply a July daytime temperature change to all annual (COVID-year) deaths with a threshold-free RR, which yields more "averted" deaths than London's plausible annual heat burden. If published as is, these numbers would likely be quoted in policy settings and later challenged. The headline equity result (45.5% → 4.5%) is also largely guaranteed by the targeting algorithm, which uses SVI. The authors acknowledge this in an SI caption but not in the Abstract. The trade-off for older adults is under-reported. I also found several inconsistencies between text and figures that suggest the text was not fully updated after figures were regenerated. None of this suggests misconduct, but it calls for careful re-checking. I believe a substantially revised version could be suitable, but it needs a stronger heat-epidemiology component. The editor may wish to make sure that one of the methods referees is a temperature–mortality/HIA specialist.
