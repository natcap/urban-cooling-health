# Referee Report — Reviewer #4 (Methods: environmental epidemiology and biostatistics)

**Manuscript:** "Urban greening strategies reshape heat-health benefits and equity" (submitted to *Nature Health*)
**Remit:** The health and equity analysis chain: exposure–response functions (ERFs), exposure metric, attributable/averted-death calculation, baseline mortality and population, spatial allocation, uncertainty, Social Vulnerability Index (SVI), GAMs and other distributional statistics, and wording. I comment on the urban cooling, energy and labour components only where they feed into the health estimates.

---

## 1. Summary of the methodological approach

The health chain, as I reconstruct it from the Methods (lines 245–288), the SI and the figures, works as follows.

1. **Exposure.** The InVEST Urban Cooling Model (UCM) produces gridded daytime air-temperature surfaces for a "representative July". The 2050 run uses reference temperature 25 °C and UHI_max 5 °C; the 2021 run uses 22 °C and 2 °C (Sup. Table 3). For each land-use scenario, ΔT(x,y) = T_scenario − T_baseline land use, both under the same "mid-century" setting.
2. **ERF.** The model assumes a log-linear, no-threshold relative risk per 1 °C. For all-cause mortality RR = 1.020 (1.014–1.026), taken from a global urban meta-analysis (ref. 59). Four cause-specific RRs come from refs. 44 and 61 (Table 2). RR(x,y) = exp(β·ΔT) and AF = (RR − 1)/RR.
3. **Baseline deaths.** These are borough-level deaths registered in 2021 (ONS, all ages, both sexes; ICD-10 A00–R99, U00–Y89, which includes COVID-19). They are allocated to grid cells in proportion to WorldPop 2021 population. ED(x,y) = AF·D₀(x,y), summed to borough and city totals. These totals are presented as **"annual heat-related deaths averted"**.
4. **Uncertainty.** 2,000 Monte Carlo draws of β, with the same draws used across scenarios. The 2.5th and 97.5th percentiles form the "95% exposure–response uncertainty interval".
5. **Equity.**
   - An SVI is built for 4,835 LSOAs from income deprivation, % aged <5, % aged ≥75 and % ethnic minority (limited English and social housing were dropped for collinearity).
   - SVI quintiles are compared using Dunn's tests with Bonferroni correction (temperature and canopy, Fig. 2).
   - Separate univariable scaled-t GAMs fitted by REML regress LSOA deaths averted per million on each standardised indicator (Fig. 6, Sup. Fig. S6-1, ED Fig. 3).
   - Fig. 7 builds bivariate tertile maps and compares Green30 with Target30 using fixed Green30 benefit tertile thresholds (33.21 and 51.95 per million).
6. **Targeting.** Target10/20/30 rank candidate street-tree points by UTCI_norm × SVI percentile (SI, step 3). They are then compared with the proximity-based Green10/20/30, "matched on realized added-canopy area".

**Points that are unclear or not reported:**

- **(a) ERF exposure metric.** The temperature metric behind each ERF (daily mean, daily max, heatwave-day indicator, above-threshold increment) and its lag structure are not stated.
- **(b) Seasonal scope of D₀.** It is not stated whether D₀ is annual or July deaths. The labels "annual deaths" (Fig. 4c, ED Fig. 1) imply that annual deaths were multiplied by an AF derived from a July temperature change.
- **(c) Sign convention.** The convention for "averted" is not given (ΔT < 0 ⇒ AF < 0 ⇒ ED < 0).
- **(d) SVI construction.** How the four indicators are combined is not described: standardisation, direction, weighting, or a percentile transform. The SVI is described both as z-scores (Fig. 6) and as a 0–1 "tract-level SVI percentile" (SI step 3; Sup. Fig. 11 legend 0–1).
- **(e) Resolutions and denominators.** The model grid resolution used for the health calculation is not given (10 m LULC vs 30 m productivity grid). Nor is it stated which population denominator is used for "per million residents": WorldPop summed to LSOA, or Census 2021.
- **(f) GAM specification.** The basis dimension k, smoothing basis, effective degrees of freedom, deviance explained and any weighting are not reported.
- **(g) Boundary vintage.** 4,835 is the number of London LSOAs on **2011** boundaries, whereas the 2021 Census indicators are released on 2021 boundaries (about 4,994 London LSOAs; please verify). The boundary harmonisation is not described.
- **(h) UTCI source.** The source and model of the 1 m UTCI surface used for targeting are not given.

---

## 2. Overall methodological assessment

The question matters, the scenario design is policy-relevant, and the authors are commendably candid in several captions ("not spatially adjusted or causal", "SVI is related to the targeting design and is not an independent validation measure"). However, the health impact assessment has structural problems that, in my view, invalidate the headline mortality numbers as currently computed. They also make the equity conclusions largely a product of modelling choices rather than findings.

**(i) Implausibly large totals.** A July-only temperature change is applied to annual all-cause deaths, with a no-threshold global meta-analytic slope. My back-calculation (below) shows that the reported 387–496 deaths averted per year from roughly 0.3 °C of daytime cooling is of the same order as London's entire heat-attributable burden in the record 2022 summer.

**(ii) Climate-invariant and uninformative uncertainty.** Because the ERF has no threshold and ΔT in the UCM scales with UHI_max but not with T_ref, the "mid-century climate" plays essentially no role in the health results. The only uncertainty propagated (β) is a near-proportional scale factor, so it carries no information about scenario rankings.

**(iii) Distributional patterns driven by modelling choices.** Borough-level crude (not age-standardised) mortality is allocated uniformly to pixels. As a result, LSOA "benefit per million" equals a borough constant multiplied by local ΔT. The equity GAMs therefore largely recover the between-borough geography of crude death rates (older, whiter outer boroughs) and of the scenario generator's allocation rules.

**(iv) Circular targeting evaluation.** Targeting is defined on SVI and then evaluated on SVI using thresholds that do not account for Target30's 28% larger total.

Each of these can be fixed with standard methods. The fixes are substantial, however, and will change the numbers.

---

## 3. Strengths

1. **Clean attribution of land use versus climate.** The design holds climate fixed and isolates land use, and the Green and Target scenarios are matched on added-canopy area. This is a sound comparative logic.
2. **Coherent uncertainty draws.** The same Monte Carlo draws are used across scenarios, and the authors state honestly what the intervals exclude (line 271).
3. **Neighbourhood-scale analysis.** Analysis at LSOA level (n = 4,835) with explicit acknowledgement of ecological interpretation (lines 85, 95; Fig. 7 caption).
4. **Flexible, robust distributional fits.** Non-linear GAMs with a robust likelihood are used rather than linear trends, and predictor densities are displayed.
5. **Candid captions.** Captions qualify that energy values are monthly, that work capacity is area-weighted, and that age indicators describe composition, not age-specific mortality.
6. **Code availability.** A public code repository is stated, which will allow verification.

---

## 4. Major methodological concerns

### M1. Annual mortality totals derived from a single summer-month temperature change; seasonal scope of the ERF

**Location:** Methods lines 247–269; Fig. 4c ("Annual deaths"); ED Fig. 1 ("annual deaths"); Abstract ("averting 135–496 deaths … per year").

**Quote:** "We quantified the projected excess mortality attributable to summer (July) temperature increases … ED(x,y) = AF(x,y) × D₀(x,y)"; "Gridded excess deaths, estimated using population-weighted allocation of deaths registered in 2021".

**Why it matters.**

- *Back-calculation.* London registered roughly 58,000–62,000 deaths in 2021 (verify against ONS). With β = ln(1.020) ≈ 0.0198 and 387 deaths averted (Green30), the implied population-weighted ΔT is 387 / (0.0198 × ~60,000) ≈ 0.33 °C. This matches the scale of cooling in Fig. 3. The same check gives ≈ 0.93 °C for AllBuilt (1,110 deaths) against the 1.05 °C quoted. **The totals are therefore consistent with D₀ = full-year deaths.**
- *Overstatement.* A July daytime temperature change cannot plausibly modify deaths in November–March. Restricting D₀ to July deaths (about 1/12 of the annual total, or less, since summer mortality in the UK is below the winter peak) would reduce the estimates roughly 12-fold: about 30–40 for Green30/Target30. Restricting to a June–September warm season would reduce them about 3–4-fold, and only if the July ΔT were assumed to apply all summer.
- *External benchmark.* The manuscript cites UKHSA's estimate of 2,985 (2,258–3,712) heat-associated deaths across all of England in the record summer of 2022 (line 113). London holds about 16% of England's population, which implies on the order of 400–600 heat-associated deaths in London in an exceptional year. The paper claims that about 0.3 °C of tree cooling prevents 387–496 deaths every year. That is comparable to removing London's entire heat burden in the hottest summer on record, which is implausible.

**Requested analyses.**

- **(a)** State explicitly whether D₀ is annual, and justify that choice or correct it.
- **(b)** Recompute using warm-season, or ideally daily, deaths. The preferred approach is a daily HIA over a set of representative summers (observed and climate-perturbed), using a daily temperature series that the UCM-derived ΔT perturbs (for example, scaled by daily UHI intensity).
- **(c)** Relabel the outputs ("warm-season heat-attributable deaths").
- **(d)** Add an external plausibility check against UKHSA London-region heat mortality estimates and published London UHI attribution studies (e.g., ref. 29).

### M2. Transferability and exposure-metric compatibility of the ERF; no threshold or minimum-mortality temperature

**Location:** lines 247, 275–284 (Table 2); Discussion lines 143.

**Quote:** "The exposure–response function assumes log-linear associations between temperature change and relative risk … 2.0% increase in mortality per 1 °C"; "we selected exposure–response functions that best reflect climatic and geographical similarities".

**Why it matters.**

1. **Source and metric of the all-cause RR.**
   - The RR comes from a global urban heat meta-analysis (ref. 59). These pooled estimates typically refer to daily temperature above a location-specific threshold, or to heatwave versus non-heatwave days. They do not describe a linear slope across the whole temperature range. The authors should state which metric (daily mean, daily maximum, apparent temperature), lag and threshold ref. 59 refers to (please verify).
   - The exposure here is a modelled daytime maximum air temperature for an average July. Applying a daily, above-threshold slope to a monthly-mean daytime ΔT mixes metrics (Tmax vs Tmean) and timescales (daily vs monthly). It also ignores that average July conditions in London sit close to the minimum-mortality temperature (MMT), with heat deaths concentrated on a few hot days.
2. **Background climate drops out.** With a no-threshold log-linear ERF, AF depends only on ΔT, so background climate is irrelevant (see M3). Temperature–mortality curves for London are strongly non-linear: flat around the MMT and steep in the upper percentiles. A cooling of 0.3 °C on a 25 °C day and on a 33 °C day have very different consequences.
3. **Cause-specific RRs are poorly matched.**
   - The circulatory and respiratory RRs (ref. 61, Bunker et al.) come from a meta-analysis restricted to the **elderly**, yet they are applied to all-age deaths.
   - The mental-disorder and self-harm RRs (ref. 44) come from a review that pools morbidity and mortality outcomes. Please confirm that the extracted RRs are mortality-specific (verify).
   - Ref. 58 (under-five mortality in LMICs) is cited as the methodological basis for the HIA. It is a poor methodological anchor for London all-age mortality.
4. **London-specific curves exist.** Multi-country DLNM studies (e.g., the MCC Network), UK regional analyses, and UKHSA's heat mortality monitoring methodology all provide London-specific curves with MMTs and percentile-based RRs. Using a global meta-analytic slope when London-specific DLNM curves exist is hard to justify given the claim of "climatic and geographical similarity".

**Requested analyses.**

- **(a)** Replace or complement the global slope with a London-specific, non-linear ERF (DLNM-derived, cumulative over lags 0–3 or similar), applied to daily temperature with the MMT respected.
- **(b)** Report results under at least three ERF specifications: global log-linear (current), London DLNM, and London heat-only (above-MMT) slope.
- **(c)** Tabulate, for each RR in Table 2: source population, age range, temperature metric, lag, threshold, outcome type (mortality vs morbidity), and the conversion applied.
- **(d)** For cause-specific estimates, use age-appropriate RRs, or restrict D₀ to the age group the RR was estimated in.

### M3. Health results are essentially invariant to the climate scenario and highly sensitive to the unvalidated UHI_max parameter

**Location:** Methods lines 163–165, 249; Sup. Tables 1 and 3; Fig. 3 caption; ED Fig. 2.

**Quote:** "ΔT … computed as the difference between a counterfactual land-use scenario and the baseline land-use configuration under the same mid-century climate forcing"; Sup. Table 3: 2021 UHI = 2 °C, 2050 and 2070 UHI = 5 °C.

**Why it matters.**

- In the UCM, air temperature is approximately T_ref + (1 − HM)·UHI_max, followed by spatial blending. The land-use ΔT is therefore about −ΔHM·UHI_max. It scales linearly with UHI_max and is nearly independent of T_ref.
- Combined with the threshold-free ERF (M2), the deaths averted are approximately D₀(2021) × β × ΔHM × UHI_max. SSP370 versus SSP585, and 2050 versus 2070, then differ only through the second-order effects of ET₀ and RH.
- The jump of UHI_max from 2 °C (2021) to 5 °C (2050) mixes the "average" and "maximum" UHI from the Yale product (line 165). By itself, this jump multiplies every mortality estimate by about 2.5×.
- The framing that the results are "under mid-century climate conditions" is therefore misleading. The key driver is an assumed UHI magnitude that has not been validated for London against observations such as urban station networks or MIDAS stations.

**Requested analyses.**

- **(a)** Show deaths averted for UHI_max = 2, 3, 4 and 5 °C (and for the 2070 settings) to make the scaling explicit.
- **(b)** Validate UCM air temperatures, or at least the spatial ΔHM pattern, against observed London data before using them as the ERF exposure.
- **(c)** Report SSP370 and SSP585 health results separately. If they are identical, say so and explain why.

### M4. Baseline mortality and population: 2021 COVID-affected deaths, crude rates, and a fixed 2021 population applied to 2050

**Location:** lines 247, 269, 273; Sup. Fig. 9.

**Quote:** "Baseline registered cause-specific mortality rates … in calendar year 2021 … all ages and sexes"; ICD range "A00–R99, U00–Y89".

**Why it matters.**

1. **COVID-inflated baseline.** 2021 included a large COVID-19 wave in London (January–February 2021). The all-cause baseline includes U07 deaths, so D₀ is inflated. Heat-ERF RRs are estimated on non-pandemic baselines.
2. **Crude rates.** The rates are crude all-age rates, so between-borough differences reflect age structure more than vulnerability (see M5).
3. **Fixed 2021 population.** A fixed 2021 population and age structure is combined with 2050 canopy and climate. London's ≥75 population is projected to grow substantially, and heat vulnerability is concentrated in this group.
4. **Figure mislabelled.** Sup. Fig. 9 is captioned "mortality rates", but the legends show counts ("number", 0–3,000). It also contains an extra panel (Y10–Y34) that is not described.

**Requested analyses.**

- **(a)** Use a pre-pandemic multi-year average (e.g., 2015–2019) or a post-pandemic average (2022–2024) of warm-season deaths, excluding U07.
- **(b)** Provide age-specific baseline rates. Compute LSOA expected deaths by applying borough age-specific rates to LSOA age-specific populations (indirect standardisation).
- **(c)** Add a sensitivity analysis with ONS 2050 population projections (size and ageing), clearly separated from the land-use contrast.
- **(d)** Correct the Sup. Fig. 9 labels.

### M5. Uniform within-borough mortality allocation drives the equity patterns, including the age ≥75 "trade-off"

**Location:** line 269; Fig. 6 caption ("Borough mortality is allocated to pixels in proportion to population"); Results lines 85, 95.

**Quote:** "Neighborhoods with higher proportions of residents aged ≥75 received greater modelled per-resident benefits"; "among neighborhoods with the highest proportions of residents aged ≥75 years, the low-benefit share increased from 28.7% to 59.0%".

**Why it matters.**

- If D₀(x,y) = m_borough × pop(x,y), then deaths averted per million in LSOA i is approximately m_borough(i) × β × ΔT̄_i (population-weighted). **The only within-borough source of variation is ΔT. All between-borough variation in the outcome partly reflects crude death rates.**
- Crude death rates are highest in older outer boroughs (Sup. Fig. 9), which are also less deprived and less ethnically diverse (Sup. Fig. 10: ≥75 correlates −0.45 with income deprivation and −0.40 with ethnic minority share).
- The positive ≥75 gradient under Green30 and the negative SVI and ethnicity gradients are therefore expected even if ΔT were spatially uniform.
- The reverse inference is also flawed. An LSOA with many ≥75 residents in a young borough is assigned the borough's low crude rate, so its benefits are under-estimated. Heat-related deaths fall mostly at ages ≥75. Not modelling age-specific baseline mortality (and age-specific RRs) removes the most important vulnerability axis. This is exactly the axis on which the authors report a "trade-off".

**Requested analyses.**

- **(a)** Decompose the per-million benefit into (i) population-weighted ΔT (the pure exposure benefit) and (ii) the baseline-rate component. Re-run all equity analyses on ΔT alone.
- **(b)** Re-run with age-standardised LSOA expected deaths (M4b) and age-specific RRs (e.g., <65, 65–74, ≥75).
- **(c)** Re-fit the GAMs with borough fixed or random effects to separate within-borough from between-borough gradients.
- **(d)** Report the ≥75 trade-off under (a)–(c) before drawing policy conclusions about older residents.

### M6. Circularity and a level-versus-distribution confound in the targeting evaluation

**Location:** SI Methods step 3 ("Priority score = UTCI_norm × SVI"); Results lines 93–95; Fig. 7; Abstract ("reduced the share of high-SVI neighborhoods receiving relatively low health benefits from 45.5% to 4.5%"); Sup. Fig. S7 caption.

**Why it matters.**

1. **Circular evaluation.** Trees are placed in proportion to SVI, so a higher per-capita ΔT in high-SVI LSOAs follows by construction. The authors acknowledge this in Sup. Fig. S7 ("not an independent validation measure"), but the Abstract and Fig. 7 present it as the headline equity finding. The SVI also contains the ≥75 share with a positive sign. Yet the targeting reduces benefits to high-≥75 LSOAs, which shows the composite is internally conflicted (M7).
2. **Level confound.** Target30 averts 28% more deaths citywide than Green30 (496 vs 387; mean per-million ≈ 56 vs 44 using 8.8 M). Holding the Green30 tertile cut-points fixed (33.21 and 51.95) therefore moves LSOAs into the "high" class even under a uniform upward shift. The 45.5% → 4.5% and 25.1% → 84.6% changes mix more total benefit with redistribution.
3. **Losers are hidden.** ED Fig. 3 shows Target30 benefits close to zero at low SVI (z ≈ −2). The redistribution has clear losers, which a share-of-high-SVI metric does not reveal.
4. **Unclear matching.** "Matched on realized added-canopy area" is asserted but not shown. Street-point targeting (1.82 M sites) disperses canopy into residential streets where both people and UCM cooling kernels overlap. This may explain the higher total benefit independently of equity.

**Requested analyses.**

- **(a)** Evaluate equity with metrics that are not the targeting variable: income deprivation alone (not in the priority score if SVI is replaced), IMD domains not used in the SVI, and age-standardised baseline mortality.
- **(b)** Use scale-invariant distributional measures: concentration index or curve of per-capita benefit ranked by deprivation (population-weighted), the share of total benefit accruing to each SVI quintile, and within-scenario tertiles.
- **(c)** Report realized canopy area by scenario and borough (Green vs Target).
- **(d)** Run a "Target-uniform" control (street-tree allocation without SVI weighting, UTCI-only) to separate the street-placement effect from the vulnerability-weighting effect.
- **(e)** Report who loses benefit, by SVI quintile and by borough.

### M7. SVI construction is under-specified and conceptually inconsistent

**Location:** lines 286–288; Sup. Figs. 10–11; SI step 3.

**Why it matters.**

- No aggregation formula is given (sum of z-scores? percentile ranks as in the CDC SVI? equal weights?). Directionality is not discussed.
- The "four-dimension" SVI includes %≥75 and %<5 under "Household characteristics". The ≥75 share is negatively correlated with every other indicator (Sup. Fig. 10), so it partially cancels the socioeconomic and ethnic signal.
- The SVI appears as z-scores in the GAMs and as 0–1 percentiles (labelled "tract-level") in targeting.
- Dropping limited English and social housing because of collinearity (r = 0.76 and 0.85) is defensible for a regression, but it is not a reason to drop them from a composite index.
- No sensitivity analysis to weighting, or validation against established UK indices (IMD 2019, the Climate Just neighbourhood flood/heat vulnerability indicators, the GLA Climate Risk Mapping index; verify names and versions), is presented.
- The boundary vintage issue (2011 vs 2021 LSOAs; item 1(g)) affects every SVI value.

**Requested analyses.**

- **(a)** Give the exact formula, the transform used (z vs percentile) and the weights.
- **(b)** Construct the SVI with and without the age components (keeping a socioeconomic SVI separate from biological sensitivity), with equal and PCA-derived weights.
- **(c)** Report the correlation of the SVI with IMD 2019 and the GLA Climate Risk index.
- **(d)** Document the boundary harmonisation (2011 vs 2021 LSOAs, lookup used, and how the 4,835 vs 4,831 counts arise; Sup. Fig. 12).

### M8. GAM inference on deterministic model output, with no spatial dependence, weighting or multivariable adjustment

**Location:** Figs. 6, S6-1, ED Fig. 3 captions; Results lines 83–93.

**Quote:** "separate univariable robust scaled-t GAMs fitted by restricted maximum likelihood … pointwise 95% model-conditional confidence intervals".

**Why it matters.**

- The outcome is a deterministic function of model inputs, so there is no sampling variability in the usual sense. The CIs reflect only residual scatter around a smooth fit and treat 4,835 strongly autocorrelated LSOAs as independent. They are far too narrow and should not be read as evidence of "significant" gradients.
- The fits are univariable even though the predictors are correlated (|r| up to 0.55), so the separate curves are not separable effects.
- LSOAs are unweighted, so results describe neighbourhoods rather than people.
- No k, edf, basis, deviance explained or residual diagnostics are given, and no concurvity check is reported.
- The 90th–10th contrasts quoted in the text (−8.8, −16.5, +5.7, +34.0, +58.0) have no uncertainty attached.
- Fig. 6 itself is inconsistent across versions in the submission (see m1): one version is labelled "relative to mean / partial effect centred at 0", while the caption says "full predictions including intercept".

**Requested analyses.**

- **(a)** Frame the GAMs explicitly as descriptive smoothers of model output, or switch to descriptive distributional statistics (M6b).
- **(b)** If inferential language is retained, fit spatial GAMs (adding s(easting, northing) or a Markov random field over LSOA adjacency, plus a borough random effect). Report Moran's I of the residuals, population weights, and a multivariable model.
- **(c)** Report k, edf, deviance explained and the outcome of `gam.check`.
- **(d)** Provide bootstrap intervals, block- or borough-clustered, for the 90th–10th contrasts.
- **(e)** Propagate β-draws and UCM/UHI sensitivity into these contrasts.

### M9. The uncertainty intervals are almost uninformative and the reporting overstates precision

**Location:** line 271; Fig. 4c; ED Fig. 1; Abstract.

**Why it matters.**

- With a single log-linear β and fixed ΔT, D₀ and population, every scenario's estimate is approximately proportional to β. The intervals are therefore a uniform ±~30% scaling, and the ranking and the Target/Green ratios carry zero propagated uncertainty.
- The dominant uncertainties are excluded: ERF form and transferability (M2), UHI_max and UCM error (M3), baseline mortality (M4), and GCM spread. Yet the headline is presented as "496 (347–640)".
- The way the SE was derived from the published confidence limits is not stated (log scale is implied; please confirm). The distribution assumed for β draws is also not stated.

**Requested analyses.**

- **(a)** Run a joint Monte Carlo or a structured sensitivity analysis over ERF specification × UHI_max × baseline-mortality year × GCM. Report a variance decomposition (for example, the share of total variance attributable to each source).
- **(b)** Report paired intervals for the Target30 − Green30 difference under that joint uncertainty.
- **(c)** In the Abstract, present ranges across specifications rather than a single β-only interval.

### M10. "Deaths averted" language and the causal framing of modelled steady-state contrasts

**Location:** Abstract, Results lines 69–71, Fig. 4 and 5 titles, Discussion lines 123 and 143.

**Why it matters.** The numbers are counterfactual, model-derived differences in heat-attributable deaths under one representative July. They rest on 2021 population and mortality, an assumed UHI, and fully grown canopy in 2050. They are not deaths prevented. Mortality displacement (harvesting) is not considered, nor is the time needed for canopy to mature.

**Requested change.** Use "modelled reduction in heat-attributable deaths (warm-season equivalent)". State the counterfactual explicitly in the Abstract, and remove "per year" unless the calculation is genuinely annual and season-appropriate.

---

## 5. Minor methodological concerns

- **m1. Inconsistent figure versions.**
  - Two versions of ED Fig. 1 appear in the figure set. image44 shows Green-only results with different values (Green10 = 161, Green30 = 377, AllBuilt = −1,078) and no error bars. image45 matches Fig. 4c (135, 387, −1,110). Remove the stale version and confirm which numbers are final.
  - Two versions of Fig. 6 also appear (image29 with "relative to mean, partial effect centred at 0" and y-values of 0.04–0.08; image30 with per-million predictions).
  - ED Fig. 2 includes no Target scenarios.
- **m2. AF formula.** AF = (RR − 1)/RR with RR = exp(βΔT) and ED = AF × D₀ is the attributable-fraction form for the exposed. Here D₀ is observed (2021-climate) deaths, not deaths under the baseline-land-use 2050 climate. For a scenario contrast, D_cf − D_base = D_base(exp(βΔT) − 1) is the more natural form. State the formula and sign convention exactly, and say which baseline D₀ represents.
- **m3. Resampling description.** "Count-preserving sum resampling" of 100 m WorldPop counts to the model grid is ambiguous. Sum-resampling to a finer grid requires splitting, not summing. Give the target resolution and the method (areal or dasymetric). Also report the WorldPop London total against the ONS mid-2021 estimate (about 8.8 M).
- **m4. Per-million denominator.** State the per-million denominator and whether LSOA-level population is from WorldPop or the Census. Mixing sources creates ratio artefacts in small or non-residential LSOAs (e.g., the City of London; the 0.29 deaths per borough minimum).
- **m5. Dunn's tests (Fig. 2, Sup. Fig. 12).**
  - The Kruskal–Wallis statistic, quintile sizes and effect sizes (ε², or Cliff's δ for pairs) are not reported.
  - A median difference of 0.17 °C on 1-km CHELSA data across about 1 km² LSOAs (many LSOAs share pixels) is inflated by pseudo-replication and spatial autocorrelation.
  - Present effect sizes and use spatially aware tests (for example, a permutation test with spatial blocks).
- **m6. Temperature metric labels and GCM set.**
  - Metric labels are inconsistent: "mean temperature of the warmest month" (line 45), "average maximum temperature" (Fig. 1), and "July 2021 temperature from the CHELSA maximum-temperature product" (Fig. 2).
  - The 2021 baseline is a single month, whereas the projections are 20-year climatologies from a different product. The 3.9–4.2 °C "rise" mixes a weather anomaly with climate change. The UCM uses a +3 °C (22 → 25 °C) shift, which is inconsistent with this.
  - The GCM set differs between Methods (EC-Earth3-Veg) and Sup. Fig. 12 (ACCESS-CM2).
- **m7. 33.21/51.95 thresholds.** Define them explicitly (unweighted LSOA tertiles of Green30?) and report results under population-weighted tertiles.
- **m8. Cause-specific results.**
  - The cause-specific outcomes are not mutually exclusive with all-cause, and the self-harm estimates (≈2 deaths) are too small to be meaningful.
  - Mental and behavioural disorders (F00–F99) are dominated by dementia deaths (F01/F03) in older adults. The RR source should match that.
  - Consider dropping the self-harm outcome or presenting it only as rates.
- **m9. Exposure location.** Exposure is assigned at the residence pixel (outdoor daytime air temperature). Discuss indoor exposure, night-time temperature (often more relevant for UK heat mortality), and time-activity patterns. The UCM daytime-only choice (line 165) is a real limitation for mortality.
- **m10. Borough results.** Borough-level deaths averted (0.29–29.75) depend on borough crude rates. Report per-100,000 rates with age-standardised baselines.
- **m11. Monte Carlo reporting.** Report the seed and the number of draws needed for convergence. With 2,000 draws, the 2.5th and 97.5th percentiles are stable to within about 1%, which is acceptable.
- **m12. SI text error.** A duplicated phrase in SI step 4 (", using average crown-size assumptions…") needs cleaning. "Tract-level" should read "LSOA-level".
- **m13. Methods figures.** The Methods-section images (image32 to image34: bivariate maps across five indicators, and stacked bar shares) are uncaptioned and unreferenced. Integrate or remove them.

---

## 6. Statistical reporting checklist

| Item | Reported? | Where | Gap |
|---|---|---|---|
| ERF source, population, age range | Partly | Table 2 | Age range (elderly-only for CVD/respiratory), outcome type and region not tabulated |
| ERF temperature metric, lag, threshold or MMT | No | – | Essential (M2) |
| Functional form of ERF | Yes | l. 247 | Log-linear, no threshold; not justified for London |
| Seasonal scope of baseline deaths | No | – | Annual vs July unstated; totals imply annual (M1) |
| Baseline mortality year and ICD codes | Yes | l. 273, Table 2 | 2021 includes COVID (U07); crude, not age-specific |
| Population source and resampling | Partly | l. 273 | Target grid, method and total check missing |
| Allocation borough → pixel → LSOA | Partly | l. 269, Fig. 6 | Implications for equity not analysed (M5) |
| AF / averted-death formula and sign | Partly | l. 249–267 | Sign convention and the scenario of D₀ unclear |
| Uncertainty sources included | Yes | l. 271 | Only β; dominant sources omitted (M9) |
| β sampling distribution and SE derivation | Partly | l. 271 | Log scale and distribution not stated |
| Paired scenario-difference intervals | No | – | Needed for Target vs Green |
| Sensitivity to UHI_max, ERF, mortality year | No | – | Required (M3, M2, M4) |
| SVI formula, weights, standardisation | No | l. 288 | Required (M7) |
| SVI validation and sensitivity | No | – | Compare with IMD and the GLA index |
| LSOA boundary vintage | No | – | 2011 vs 2021 (4,835 count) |
| Dunn's tests: KW statistic, n, effect size | No | Fig. 2 | Only letters reported |
| GAM: k, basis, edf, deviance explained, diagnostics | No | Fig. 6 caption | Only family and REML reported |
| GAM weighting (population) | No | – | Unweighted LSOAs |
| Spatial autocorrelation (Moran's I of residuals) | No | – | Required (M8) |
| Multivariable / confounding structure | No | – | Univariable only |
| Uncertainty for 90th–10th contrasts | No | l. 83–93 | Point values only |
| Definition of tertile thresholds | Partly | Fig. 7 | Unweighted? Fixed to Green30 only |
| Multiple-comparison correction | Yes | Fig. 2 | Bonferroni within panel; none across GAM panels |
| Code and data availability | Yes | End matter | Processed outputs "upon publication" |

---

## 7. Requested additional analyses, ranked by priority

1. **Recompute the HIA with a season-appropriate D₀** (warm-season or daily deaths). Preferably run a daily HIA with a London-specific non-linear (DLNM) ERF respecting the MMT, compare against UKHSA London heat-mortality benchmarks, and relabel the outputs. (M1, M2)
2. **Use age-specific baseline mortality and LSOA age structure, with age-appropriate RRs.** Re-run all equity analyses, including the ≥75 "trade-off". (M4, M5)
3. **Replace the COVID-affected 2021 baseline** with a multi-year non-pandemic average, excluding U07. (M4)
4. **Run a joint sensitivity and uncertainty analysis** over ERF specification, UHI_max (2–5 °C), mortality baseline year and GCM/SSP, with a variance decomposition and paired Target − Green intervals. (M3, M9)
5. **Decompose equity results** into a pure exposure component (population-weighted ΔT) and a baseline-rate component, and separate within-borough from between-borough gradients (borough fixed or random effects). (M5, M8)
6. **Evaluate targeting with non-circular, scale-invariant metrics:** a concentration index on IMD or income, benefit shares by quintile, a UTCI-only targeting control, losers by quintile, and realized canopy area by scenario. (M6)
7. **Fully specify the SVI and test its sensitivity:** formula, weights, with and without age components, validation against IMD and the GLA Climate Risk index, and boundary harmonisation. (M7)
8. **Re-fit the GAMs as spatial, population-weighted models** with full reporting (k, edf, deviance explained, residual Moran's I), or present them strictly as descriptive smoothers. (M8)
9. **Validate the UCM** air-temperature patterns against London observations before using them as the ERF exposure. (M3)
10. **Clean up figures and labels** (stale ED Fig. 1 and Fig. 6 versions, Sup. Fig. 9 units, temperature metric labels, GCM list). (m1, m4, m6)

---

## 8. Recommendation

**Major revision.** This is close to "reject, resubmission encouraged" as far as the health component is concerned.

The scenario design and equity framing are valuable. The mortality quantification, however, combines a July temperature change with annual deaths and a no-threshold global slope, which inflates absolute estimates by roughly an order of magnitude. The equity analysis is substantially shaped by uniform borough crude-rate allocation and by evaluating targeting on its own targeting variable. Both can be repaired with standard HIA practice. The revised estimates and possibly some distributional conclusions (notably for older residents) will change materially, and the manuscript would need re-review.

---

## 9. Confidential comments to the editor

- The submission's central health numbers (135–496 deaths averted per year) do not survive a simple plausibility check. Back-calculation shows that they equal annual all-cause deaths × β × about 0.3 °C. That would make the benefit of a 30% canopy increase comparable to London's total heat-associated mortality in the record 2022 summer. I believe this is an error of seasonal scope, not misconduct, but it must be corrected before the paper can be assessed on its merits.
- The equity headline (45.5% → 4.5%) is largely circular. The authors acknowledge this in an SI caption but not in the Abstract. The age ≥75 "trade-off" is probably an artefact of crude, borough-level mortality allocation.
- The figure set contains stale versions of ED Fig. 1 and Fig. 6 with different numbers, which suggests the manuscript was assembled hastily. I recommend asking the authors for a reproducibility check from the stated code repository.
- I have no competing interests. My assessment covers only the health and equity chain. The energy and labour modules need separate specialist review (for example, applying monthly kWh/m²/°C coefficients and area-weighting work capacity rather than weighting it by workers).
