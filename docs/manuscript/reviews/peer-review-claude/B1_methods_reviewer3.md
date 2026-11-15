# Referee Report — Reviewer #3 (Methods: urban climate, remote sensing, ecosystem-service and impact modelling)

**Manuscript:** "Urban greening strategies reshape heat-health benefits and equity" (Nature Health)
**Remit:** The physical and environmental modelling chain: land cover and scenarios, the InVEST Urban Cooling Model (UCM), climate forcing, the exposure metric, the energy, WBGT and work-capacity models, scale handling, uncertainty and reproducibility. I comment on the mortality module only where the exposure definition feeds into it.

---

## 1. Summary of the methodological approach (as I understand it)

1. **Baseline land cover.** UKCEH Land Cover Map (10 m). Pixels classed "Suburban" that overlap the GLA 1 m tree-canopy layer are reclassified to a new "Tree Cover" class (code 100) (Methods, LULC paragraph; Sup. Fig. 1).
2. **Climate characterisation (Figs 1–2, Sup. Figs 2, 12).** The baseline is CHELSA July 2021 monthly Tmax (~1 km). Futures are WorldClim v2.1 CMIP6 30″ July Tmax for 2021–2040, 2041–2060 and 2061–2080 under SSP370/585, averaged over three GCMs. Borough "ΔT" in Fig. 1b is WorldClim future minus CHELSA 2021.
3. **Temperature model (UCM, InVEST 3.20.2).** The model is driven by *scalar* inputs, not by the climate rasters: a reference (rural) air temperature T_ref of 22/25/28 °C and a UHI_max of 2/5/5 °C for "2021/2050/2070" (Sup. Table 1, Sup. Table 3). Other inputs are a 500 m air-blending distance, a 450 m maximum cooling distance, CGIAR reference ET, and biophysical parameters from Sup. Table 2. The model is run daytime only. Its output is a 10 m air-temperature field.
4. **Land-use scenarios.**
   - AllBuilt: all vegetation reclassified to Urban.
   - TreeRisk: inventoried GiGL trees judged climate-risked using Esperon-Rodriguez et al. are converted to grassland.
   - TreeOpp: planting on GiGL open-space opportunity land.
   - Green10/20/30: InVEST Scenario Generator, proximity-based, using a "nearest-edge" rule.
   - Target10/20/30 (SI): OSM street-network candidate points ranked by normalised 14:00 UTCI × SVI percentile, rasterised to 10 m.
   - Land-use effects are computed as scenario minus baseline under the same T_ref/UHI_max. The headline values are taken at 25 °C / 5 °C (Fig. 3b).
5. **Energy.** Per-building cost = ΔT × footprint area × a type-specific coefficient (kWh m⁻² °C⁻¹ month⁻¹, said to come from a Drax Electric Insights report) × tariff. Buildings are from Colouring Britain (Sup. Table 4).
6. **Work capacity.** An Orlov et al. logistic function of WBGT for high-intensity work, run on a 30 m grid and reported as an area-weighted mean. WBGT presumably comes from the UCM's internal T_air–RH approximation. RH is a single scalar per period (55/45/42%), derived from WorldClim 1970–2000 vapour pressure and future Tx (SI).
7. **Mortality.** AF = (RR − 1)/RR with RR = exp(βΔT). Baseline deaths are 2021 borough registered deaths allocated to pixels in proportion to WorldPop population. Exposure–response uncertainty uses 2,000 Monte Carlo draws. Energy and work capacity are deterministic.
8. **Equity.** LSOA-level GAMs and bivariate tertile maps against a four-dimension SVI.

**Points I could not reconstruct from the text:**
- (a) Exactly how WBGT is computed.
- (b) Which grid is used for the health calculation, and how 100 m WorldPop counts are "sum-resampled" to it.
- (c) Which SSP's ET0 and TreeRisk layer feed the "mid-century" results in Figs 3b–7.
- (d) Whether the baseline tree-canopy share counts the woodland classes or only class 100.
- (e) The realised added-canopy area (km²) of each Green and Target scenario. The Methods claim they are matched, but no numbers are given.
- (f) The source, date and spatial extent of the 1 m UTCI field used for targeting.
- (g) How the "12–16% more cooling energy per degree" figure becomes the coefficients in Sup. Table 4.
- (h) Which annual or seasonal death count the July ΔT is applied to.

---

## 2. Overall methodological assessment

The question matters, and the design is sound in principle: hold climate fixed, vary land cover, and push the temperature change through several impact pathways. The scenario set is policy-relevant. The comparison of Target versus Green allocation is the paper's most original contribution.

As implemented, however, the physical modelling chain has three problems that bear directly on the headline numbers:

- It is **not validated against any observation** of air temperature or land-surface temperature (LST) in London.
- It is **parameterised with several values that are internally inconsistent or physically questionable**:
  - woodland is given zero shade;
  - UHI_max is taken from a satellite *surface* UHI product;
  - the UHI parameter is switched from an "average" to a "maximum" value between baseline and future.
- Its **exposure metric (a July mean-daytime-maximum air-temperature change) is transferred to impact functions built for different temporal metrics**: annual deaths, monthly energy, and hourly or daily WBGT.

The UCM's daytime air temperature is linear in UHI_max: T_air = T_ref + (1 − HM)·UHI_max, followed by spatial blending. Every land-use effect in Figs 3b–7 is therefore exactly proportional to the chosen UHI_max = 5 °C and independent of T_ref. The claim that effects are "consistent across years" follows mechanically from this structure; it is not a finding.

The climate-versus-land-use comparison is also confounded:
- Fig. 3a's "2030" step (22 °C/5 °C) is purely a change of UHI parameter, not a projection.
- The 3.9–4.2 °C mid-century "differences" in Fig. 1 mix two products (CHELSA single-year weather and a WorldClim 20-year climatology). They are much larger than the ~2.7 °C the authors themselves cite for the 2050s.

None of these problems is fatal to the paper's *relative* message, that allocation matters. Together, though, they leave the absolute magnitudes poorly supported: 1.05 °C, −0.37 °C, 135–496 deaths, £22.4 M per month, and 1.42 percentage points. These are the numbers in the abstract. Uncertainty is propagated only for the exposure–response coefficient; parameter, structural and climate uncertainty are not propagated at all.

I consider the work publishable only after substantial methodological revision: validation, a corrected parameterisation, consistent exposure metrics, and a sensitivity and uncertainty analysis.

---

## 3. Strengths

1. **Clear counterfactual logic.** The study separates the climate effect from the land-use effect and states this in the first lines of the Methods.
2. **Useful scenario span.** The scenarios run from an extreme bound (AllBuilt), through a stress (TreeRisk) and feasible options (TreeOpp), to policy-anchored targets (Green10/20/30). They are tied to the London Environment Strategy.
3. **Policy-relevant contrast.** Comparing proximity-based with vulnerability-targeted allocation at nominally equal canopy gain is novel and directly relevant to policy.
4. **Attention to local data.** The authors correct UKCEH's misclassification of suburban tree canopy using the GLA canopy layer, and use GiGL tree and open-space data and Colouring Britain building footprints.
5. **Transparent caveats in places.** Energy is labelled as warm-month and work capacity as area-weighted, not worker-weighted. The authors acknowledge the circularity of SVI in Sup. Fig. S7 and state that the 95% intervals reflect exposure–response uncertainty only.
6. **Open-source tools and code.** The chain uses an open model (InVEST), and code is said to be public.

---

## 4. Major methodological concerns

### M1. No calibration or validation of the temperature model against London observations
**Location:** Methods, Urban Cooling Model. Sup. Table 2 ("Parameters were compiled based on a UCM calibration study⁴…").

**Issue.** The only calibration cited (Hamel et al. 2024) was carried out in France and the United States. London has rich observational resources, none of which is used:
- Met Office MIDAS stations (already cited for T_ref);
- dense citizen weather-station networks (e.g., Netatmo), which have been used for London UHI studies (verify specific references);
- Landsat and ECOSTRESS LST;
- high-resolution urban climate model output (e.g., the UK Met Office or WRF-urban simulations behind Simpson et al. 2024, ref. 29).

The UCM's default formulation includes the 0.6/0.2/0.2 weights on shade, albedo and ET, the 2 ha park rule, and the cooling and blending distances. Its skill varies widely between cities, and published calibrations change the optimal blending and cooling distances by factors of several.

**Why it matters.** Every downstream number inherits the UCM ΔT. Without validation, a reader cannot judge whether −0.37 °C (or −0.34 °C; see M10) is realistic.

**Requests:**
- (i) Validate the baseline UCM T_air field against July 2021 (or 2022) station and citizen-station daytime Tmax. Report RMSE, bias, and Spearman correlations of the spatial pattern.
- (ii) Additionally compare it with Landsat or ECOSTRESS daytime LST patterns, acknowledging the difference between air and surface temperature.
- (iii) Calibrate UHI_max, the air-blending distance, the cooling distance and the CC weights, for example following the Hamel et al. procedure.
- (iv) Benchmark the canopy–temperature sensitivity against observational and model literature. Suitable comparators include street-scale and city-scale studies reporting a few tenths of a °C of air cooling per 10 percentage points of canopy, with larger values for LST. The European health-impact assessment that modelled raising tree cover to 30% across ~90 European cities (Iungman et al., *Lancet* 2023 — verify) reported a summer-mean cooling of about 0.4 °C. For context, the present study reports −0.34 °C for only ~+5.9 percentage points of canopy.

### M2. UHI_max: surface-UHI source, inconsistent "average/maximum" switch, and linear scaling of all results
**Location:** Methods: "We obtained the estimated average and maximum magnitude of the UHI effect during summer… (2 and 5 °C, respectively) from a global study conducted by Yale". Sup. Table 3 (UHI 2 °C for 2021; 5 °C for 2050/2070). Fig. 3 caption.

**Issues:**
- (a) To my knowledge the Yale YCEO UHI explorer is a **MODIS-LST-based surface UHI** product (verify). Daytime surface UHI routinely exceeds canopy-layer air UHI several-fold. The UCM requires an *air*-temperature UHI_max. Observed daytime air UHI in London is typically of order 1–2 °C; the large air UHIs occur at night.
- (b) The *average* value (2 °C) is used for 2021 and the *maximum* value (5 °C) for the future. This is a change of statistic, not a projection of UHI intensification. The text acknowledges that London-specific evidence for UHI growth "remains limited".
- (c) Because T_air − T_ref = (1 − HM)·UHI_max before blending, **all land-use ΔT, and therefore deaths and energy (which are approximately linear), scale by UHI_max/5.** Had the 2021 value of 2 °C been used, the −0.37 °C would be about −0.15 °C, and the death and energy estimates would shrink by roughly 60%.
- (d) In Fig. 3a, the "2030" (22/5 °C) bar shows ~+2.2 °C relative to 2021 with no change in reference temperature. This is purely the UHI parameter switch, yet it is presented on a "Year" axis alongside climate effects.

**Requests.**
- Justify UHI_max using **air-temperature** observations for London summer afternoons, for example urban minus rural MIDAS or citizen-station Tmax.
- Use the same UHI_max for baseline and future unless UHI growth is modelled explicitly.
- Present all land-use effects as a function of UHI_max (e.g., 1, 2, 3, 5 °C) or per °C of UHI_max.
- Remove or relabel the "2030" setting in Fig. 3a.
- State clearly in the Results and Abstract that the land-use magnitudes are conditional on UHI_max = 5 °C.

### M3. Biophysical parameter table contains errors that drive the scenario contrasts
**Location:** Sup. Table 2.

**Issues:**
- (a) **Deciduous and coniferous woodland have Shade = 0**, while class 100 "Tree Cover" has Shade = 1. With the default weights, a woodland pixel's cooling capacity (CC) is about 0.2·0.12 + 0.2·ETI ≈ 0.16, compared with about 0.78 for a Tree Cover pixel. London's densest canopy is therefore treated almost like grassland.
- (b) The Green scenarios explicitly convert "selected grassland, **woodland**, suburban and urban categories to tree cover" (Methods, Green10). Converting woodland to "Tree Cover" is a relabelling with no physical change, yet under these parameters it produces large artificial cooling. It may also inflate the "added canopy" accounting, depending on point (d) below.
- (c) Albedo is nearly uniform: 0.11 for Urban against 0.12–0.13 for vegetation. The albedo term therefore contributes almost nothing. Typical urban albedo varies widely by roof type and material, so the parameter should be documented and spatially explicit, or tested in a sensitivity analysis.
- (d) "Green area" flags are inconsistent: Acid grassland, Arable and Saltmarsh are 0, while Neutral and Calcareous grassland are 1. It is also unclear whether baseline tree-canopy cover (TCC) is computed from class 100 alone or includes woodland. The absolute TCC values in Table 1 (18–25%) cannot be reproduced from the text.
- (e) Only Suburban pixels under GLA canopy are reclassified. Canopy overhanging **Urban** pixels (street trees over roads and pavements) stays "Urban" in the baseline. The Target scenarios then *add* street trees as Tree Cover pixels in exactly these settings. This baseline/scenario asymmetry favours Target, and it under-represents existing canopy in dense, deprived neighbourhoods, which bears on the equity baseline.
- (f) Kc for Tree Cover (0.8) exceeds that of woodland (0.7). Applying Kc = 0.8 to a 10 m pixel that is physically mostly road plus one street-tree crown over-states ET.
- (g) "ECOTRESS" should read ECOSTRESS; the year and statistic of the albedo product should be stated.

**Request.** Correct the woodland shade values; Shade should reflect canopy fraction per class. Re-run all scenarios. Report a land-cover transition matrix (pixels and km², from-class × to-class) for every scenario. Decompose ΔT by source class.

### M4. Green-vs-Target "matching on realised added-canopy area" is asserted but not demonstrated, and the two scenario families differ in more than spatial allocation
**Location:** Methods, first paragraph ("Green and Target scenarios were matched on realised added-canopy area"). SI, Targeted scenarios, step 4. Table 1.

**Issues:**
- (a) No table reports the realised added canopy (km² or pixels) for Green10/20/30 versus Target10/20/30.
- (b) The SI site counts (565k, 1,129k and 1,819k) are not proportional to 10/20/30%: the ratio for Target30 is 3.2 rather than 3.0. The SI says the totals are "approximate… some projected tree crowns may overlap".
- (c) The crown-size assumption is not given. Each 5 m-spaced point is burned into a 10 m pixel, so the mapping from points to canopy area is non-linear.
- (d) Target converts mainly **Urban** (road and pavement) pixels, whereas Green converts mainly **Suburban** and existing woodland or grassland. In the UCM, ΔT per converted pixel is ΔHM × UHI_max. ΔHM is largest when the source class has the lowest CC (Urban), regardless of where vulnerable people live. Part of Target's advantage may therefore come from its source-class mix and street-tree geometry rather than from "where" in relation to vulnerability. Note that the UCM contains no dependence of cooling on local background heat other than through land cover; ranking points by UTCI therefore effectively selects low-CC pixels.
- (e) The UTCI field at 1 m, 14:00 local time, has no stated source, date, model or extent. Points "outside the valid simulation extent were excluded"; if that extent does not cover all 33 boroughs, Target is spatially restricted by construction.
- (f) "Tract-level SVI percentile" uses US terminology. Please confirm that LSOA SVI is used, and reconcile it with the z-scored SVI used in the equity analyses.

**Requests.**
- Tabulate realised added canopy and the source-class composition for each scenario.
- Add a control scenario: random or proximity-based allocation restricted to the same Urban street-pixel pool, at equal area. This would separate *source class* from *vulnerability targeting*.
- Document the UTCI data.

### M5. Climate forcing: baseline product mismatch, single-year baseline, a small and "hot" GCM subset, inconsistent model lists, and no link to the UCM inputs
**Location:** Results, first section and Fig. 1. Methods, Climate scenarios. Sup. Figs 2 and 12.

**Issues:**
- (a) **Baseline mismatch.** Fig. 1 differences are WorldClim 20-year climatologies minus CHELSA *single-month* July 2021. The 3.9–4.2 °C mid-century "differences" thus contain three things: real climate change since the WorldClim 1970–2000 base, inter-product bias, and the weather anomaly of July 2021.
  - Sup. Fig. 2 makes the problem visible: WorldClim 2021–2040 (SSP370) gives 24.6 °C, roughly 2.7 °C above CHELSA July 2021 (21.9 °C) for an overlapping period.
  - The authors' own introduction cites ≈2.7 °C by the 2050s.
  - CHELSA's 21.9 °C also appears low relative to Heathrow and Kew July mean-maximum station climatologies (verify against MIDAS).
  - The deltas should be computed against the **matching WorldClim historical** baseline, or against the same GCMs' historical runs. The Fig. 1 caption says this has been done only for SSP585 in the SI; it should be done for both SSPs in the main figure.
- (b) **GCM list is inconsistent.** The Methods list EC-Earth3-Veg, MRI-ESM2-0 and IPSL-CM6A-LR. Sup. Fig. 12 lists ACCESS-CM2, IPSL-CM6A-LR and MRI-ESM2-0. Sup. Fig. 2 additionally shows SSP245, which is not mentioned in the Methods.
- (c) **Three GCMs is too few** to characterise structural spread. At least two of the listed models are high-climate-sensitivity models (the "hot model" issue; verify ECS values). The cited EURO-CORDEX and ClimSIPS guidance recommends performance-based screening, which is not applied here. WorldClim provides many more CMIP6 models. For a UK city, **UKCP18** (12 km regional and 2.2 km local, including urban effects) is the natural, evaluated product. Alternatively, a global-warming-level framing would reduce the dependence on the SSP×GCM×period combination.
- (d) **The climate rasters do not enter the UCM.** The UCM uses scalars (T_ref 22/25/28 °C), and "28 °C for 2070" exceeds both SSP ensemble means shown (26.6 and 27.3 °C). T_ref is meant to be a *rural* reference, yet it is matched to *city-wide* WorldClim/CHELSA means, which already contain urban signal, and UHI_max is then added on top. This risks double-counting urban warming. The 22 °C baseline is derived from stations (Heathrow, Kew, Northolt, Hampton, Kenley) that are not rural.
- (e) **Scale mismatch.** At ~1 km, CHELSA and WorldClim do not explicitly resolve intra-urban heat. The SVI quintile contrast in Fig. 2a (22.00 vs 21.83 °C, a difference of 0.17 °C) is smaller than the products' plausible error, and the values are visibly quantised (0.05 °C steps). Many LSOAs share a single 1 km cell. The Dunn tests treat 4,835 LSOAs as independent despite strong spatial autocorrelation and shared pixels.

**Requests:**
- Recompute the climate deltas against a consistent baseline.
- Use an ensemble of ≥10 GCMs or UKCP18, and report the spread.
- State which models were used, consistently throughout.
- Derive T_ref from genuinely rural stations, or from the climate delta applied to a rural baseline.
- For the intra-urban exposure analysis (Fig. 2a), either use the validated UCM field or an urban-resolving product, or soften the claim. Use spatially robust tests (e.g., block bootstrap or spatial-error models).

### M6. The exposure metric (July mean daytime Tmax ΔT) is applied to impact functions defined on other time bases; mortality appears to be computed on annual deaths
**Location:** Methods, Health outcomes ("D₀(x,y) = m(x,y) × pop(x,y)"; "annual total all-cause heat-related deaths averted"). Fig. 4c.

**Issue.** The arithmetic implies that the July daytime ΔT is applied to **annual all-cause deaths**. For Green30, ΔT ≈ −0.34 °C and β = ln(1.02), giving AF ≈ 0.68%. A result of 387 deaths then implies D₀ ≈ 57,000, which is approximately London's total annual registered deaths. In other words, a July-afternoon cooling is assumed to act on every death in every month of the year.

This produces implausible totals:
- 496 deaths averted per year by +30% canopy is comparable to, or larger than, London's share of *all* heat-associated excess deaths in England in the record 2022 summer (UKHSA, ref. 28: 2,985 for England).
- It is roughly an order of magnitude larger per city than the European tree-cover health-impact assessment noted under M1 (verify).

Further problems:
- The ERF (Yang et al. 2024) is a heat-exposure effect estimate, presumably defined on daily temperature during hot periods or above a threshold. It is not a slope for a monthly mean-maximum shift applied year-round.
- 2021 deaths include COVID-19 excess mortality.
- Night-time temperature, which is the main UHI- and mortality-relevant exposure in London, is explicitly not modelled (Methods: "we focus on the daytime modeling only").

**Requests** (in coordination with the health reviewer):
- Restrict the attributable calculation to warm-season or heat-day deaths.
- Use a daily temperature distribution, e.g., HadUK-Grid or UKCP18 daily data with the UCM ΔT applied as a delta, with a London-specific threshold.
- Use a pre-COVID or multi-year mortality baseline.
- Add night-time UCM runs, or justify why daytime ΔT is an adequate proxy.

### M7. Energy model is undocumented and its implied sensitivity looks implausibly high
**Location:** Methods, Energy saving. Sup. Table 4 and footnote (Drax Q2 2018 report). Sup. Fig. 7.

**Issues:**
- (a) The derivation from "~165% spike… implies roughly 12–16% more cooling energy per degree" to coefficients of 9.89, 19.19 and 30.80 kWh m⁻² °C⁻¹ month⁻¹ is not shown. The cited Drax quarterly report is a market commentary, not a building-energy dataset.
- (b) **Order-of-magnitude check.** Summing Sup. Table 4, coefficient × footprint area gives ≈180 GWh per °C per month across London, or ≈£50 M per °C per month at ~£0.28/kWh. This is consistent with AllBuilt's £49.6 M for ~1 °C. For a city where residential air-conditioning is negligible, it implies that each 1 °C of July daytime warming adds a few percent or more of London's *total* monthly electricity consumption (verify against DESNZ sub-national electricity statistics). That seems far too high for a UK climate. A Retail coefficient of 30.8 kWh m⁻² per °C *per month* exceeds typical *annual* cooling intensities for UK commercial buildings (verify against CIBSE/BEIS benchmarks).
- (c) **Offices are absent.** The text says ~51% of UK cooling energy is used in offices, yet "Industry and Business" (presumably including offices) and "Mixed Use" are assigned 0. Community Services, which contribute the largest aggregate (≈121 GWh °C⁻¹), are assigned air-conditioning sensitivity although schools and community buildings are rarely air-conditioned.
- (d) **Coverage gaps.** Footprint area, not conditioned floor area, is used, which is a major issue for multi-storey central-London stock. Colouring Britain is crowd-sourced and has large gaps (Sup. Fig. 7 shows "Missing" for retail, leisure and utilities in many boroughs; 9.8 M m² are "NA"). The finding that "Hillingdon [has the largest gains] for energy" may simply reflect where building-use data exist (Heathrow-related retail and utilities).
- (e) The ΔT used is the afternoon peak of the mean July day. Cooling demand responds to cooling-degree-hours above a balance point, so the response is non-linear and threshold-dependent.
- (f) Results are fully deterministic.

**Requests:**
- Document the derivation of each coefficient.
- Validate total cooling-sensitive electricity against DESNZ/BEIS or National Grid demand–temperature relationships for London.
- Use floor area (e.g., from building heights or EPC/DEC records) and include offices.
- Report the coverage of building-use attribution by borough.
- Propagate coefficient uncertainty.
- Consider a degree-hour formulation.

### M8. WBGT and work-capacity chain: undocumented WBGT formula, constant-RH assumption, Jensen bias, and grid mismatch
**Location:** Methods, Work productivity. SI, Estimation of future RH. Sup. Table 3 (RH 55/45/42%).

**Issues:**
- (a) The WBGT formula is not given. If the InVEST simplified (Australian Bureau of Meteorology-type) approximation WBGT = 0.567·T + 0.393·e + 3.94 is used with a fixed RH (as InVEST does), then cooling a pixel also lowers its vapour pressure. That gives dWBGT/dT ≈ 1.0, compared with ≈0.57 at fixed vapour pressure. Physically, tree evapotranspiration *adds* moisture. Work-capacity gains from greening may therefore be overstated by up to ~75% on this account alone.
- (b) This WBGT approximation ignores solar radiation and wind. Radiation is the main pathway by which tree shade reduces outdoor WBGT for heavy work. The model therefore represents the physics of shade through the wrong channel.
- (c) The Orlov et al. function is calibrated on hourly or daily WBGT. Applying it once to the mean-July-afternoon condition (T ≈ 30 °C at T_ref + UHI_max) ignores the distribution of days and hours. Because the function is strongly non-linear (α₂ ≈ 16.6), this introduces Jensen's-inequality bias.
  - As I compute it, the baseline mid-century high-intensity capacity loss is ≈18% on an *average* July afternoon, with a marginal sensitivity of ≈8 percentage points per °C WBGT. This number should be reported and checked for plausibility.
  - Kjellstrom-type "4+4+4" or hourly approaches are standard and should be used.
- (d) The SI derivation is ambiguous ("T in future is estimated by using the average of (Tmax + Tmin)/2, or to simplify, T = Tx"). It assumes 1970–2000 vapour pressure is unchanged, and the closing sentence ("expect ~1.5 kPa…") is unsupported and unused. RH is one scalar for the whole city, so spatial humidity variation (e.g., parks versus pavement) is absent.
- (e) Work capacity is run on a **30 m** grid while the UCM runs at 10 m. The resampling is not described. The area-weighted mean includes parks, water and airports where no heavy work occurs.
- (f) The claim that this is "conservative" (Discussion) is not substantiated. Several choices above bias the estimate upwards.

**Requests:**
- Give the WBGT equation.
- Hold vapour pressure (not RH) fixed, or use a humidity field.
- Run on hourly or daily distributions.
- Weight by outdoor or heavy-work employment (e.g., workplace-zone statistics) or at least by built land.
- Propagate uncertainty in the parameters of the WBGT and work-capacity functions.

### M9. Uncertainty and sensitivity analysis are essentially absent for the physical chain
**Location:** Methods, final health paragraph ("These intervals represent exposure–response uncertainty only"). Fig. 4 caption ("energy and work-capacity estimates are deterministic").

**Issue.** The dominant uncertainties are in UHI_max, the blending and cooling distances, the CC weights, Kc and shade values, the scenario-generator settings, crown size, the GCM/SSP/period choice, ET0, RH, and the energy coefficients. None is propagated. The Monte Carlo over β covers only the smallest component. The error bars in Fig. 4c therefore give a misleading impression of precision. (The bars on an older embedded Fig. 3/work panel also carry unexplained error bars.)

**Requests:**
- (i) A global sensitivity analysis (Morris screening followed by Sobol, or at minimum one-at-a-time ranges) over the UCM parameters. Report the effect on the land-use ΔT and on the Green–Target contrast.
- (ii) Propagate GCM and SSP spread to the impacts.
- (iii) Give energy and work capacity uncertainty intervals.
- (iv) Report which conclusions are robust. I expect the *ranking* of scenarios and the Target > Green contrast to be more robust than the absolute values; show it.

### M10. Spatial-scale handling and internal consistency of reported values
**Locations:** Methods, Health ("Borough mortality is allocated to pixels in proportion to population"). Fig. 5 caption. Abstract versus Fig. 3b.

**Issues:**
- (a) Mortality rates are borough-level. Within a borough, deaths averted per million therefore vary *only* with ΔT, so LSOA-level associations with age composition (Figs 6–7) cannot reflect age structure through the baseline rate. Rates at MSOA level (ONS) or age-standardised and age-specific rates should be used.
- (b) The health grid and the method of downscaling WorldPop from 100 m are unclear. Sum-resampling 100 m counts to a 10 m grid spreads population over parks and water.
- (c) Borough results are **unweighted** borough means (Fig. 3). Population-weighted and area-weighted ΔT should be reported, since health depends on the former.
- (d) Numbers are inconsistent across the abstract, text and figures:
  - The abstract and Results give "−0.37 °C"; Fig. 3b shows −0.34 (Green30) and −0.38 (Target30).
  - Images embedded in the submission show other versions: Green10 −0.16 vs −0.12; Green30 heavy-work +2.19 vs +1.19; all-cause deaths 377/161/−1078 in Extended Data Fig. 2 vs 387/135/−1110 in Fig. 4.
  - Extended Data Fig. 2 omits the Target scenarios.
  - The Results describe T_WM as the "mean temperature of the warmest month", while the Fig. 1 caption says "average maximum temperature".
  - Fig. 1b map versions with different legend bins are both embedded.
  - The authors should confirm that every figure is regenerated from a single run.
- (e) "Vegetation loss increased borough temperatures by up to 1.05 °C" is the *mean* across boroughs for the AllBuilt bound, not a maximum. TreeRisk, the realistic vegetation-loss case, is +0.10 °C. The abstract should distinguish the two.

---

## 5. Minor methodological concerns

1. **LCM year and canopy-layer year.** State the LCM year used as baseline and the GLA canopy year (2024 per Data Availability). State the rule used to rasterise 1 m canopy to 10 m (any overlap, majority, or fraction threshold). The text says UKCEH has "the same high spatial resolution" as ESA WorldCover but also cites 25 m historical data; clarify.
2. **InVEST Scenario Generator settings.** Report the focal class, the convertible classes and their priorities, the number of pixels converted per class, the edge rule and the random seed. "Suburban areas converted… at higher rates" needs numbers.
3. **TreeRisk.** GiGL holds ~1.05 M inventoried trees, a fraction of London's total tree population (several million; verify GLA estimates), so TreeRisk is a lower bound. Also report:
   - the crown-size value used;
   - how Esperon-Rodriguez safety margins (species-level, city-level) were mapped to individual trees;
   - which SSP and period underlies the −6% figure.

   Converting a risked tree's 10 m pixel to grassland overstates loss where a crown is smaller than the pixel.
4. **TreeOpp.** Define the GiGL classes treated as opportunity land and the planting density.
5. **Reference ET.** The CGIAR dataset version and GCM are unspecified. Note that the UCM normalises ET by the raster maximum (ETI = Kc·ET0/ET_max), so climate-driven ET0 change has little effect. Say so rather than implying ET0 contributes to the climate signal. Sup. Table 3 lists "SSP370 or 585"; state which.
6. **Cooling and blending distances (450/500 m)** are held constant "to isolate land-use effects". They are, however, uncalibrated and strongly control how far park cooling propagates, and hence the equity patterns. Include them in the sensitivity analysis.
7. **Daytime only.** Add at least a night-time UCM run. Trees can reduce night-time cooling under some conditions, and London's mortality-relevant UHI is nocturnal (Simpson et al.).
8. **AF asymmetry.** AF = (RR − 1)/RR is asymmetric for negative ΔT (cooling). State that "deaths averted" equals −ED with ΔT < 0, and confirm the formula is applied consistently for gains and losses.
9. **Monte Carlo.** State the distribution used for β: normal on the log scale with SE = (ln UCL − ln LCL)/(2 × 1.96)?
10. **Energy apportionment.** For each building, state how ΔT is assigned (centroid, footprint mean) and how buildings crossing borough boundaries are apportioned (Fig. 5 caption).
11. **Tariffs.** Tariff values of £0.255–0.28/kWh need a source and year, and a statement of whether they are nominal or real for 2050.
12. **Mid-century period consistency.** 2041–2060 is labelled "2050". The InVEST runs use T_ref 25 °C, a rounding of the SSP370 value of 25.5–25.7 °C. SSP585 is therefore never represented in Figs 3b–7 except possibly through ET0 or TreeRisk; state this.
13. **SVI construction.** State the weighting and normalisation, and the percentile used for targeting versus the z-score used for analysis. Note that two indicators were dropped because of collinearity.
14. **Fig. 2a statistics.** The y-axis range (21.6–22.2 °C) visually exaggerates a 0.17 °C difference. Report effect sizes and spatially adjusted tests.
15. **Terminology.** "Bias-corrected" is inaccurate for WorldClim CMIP6, which is delta-change downscaling with calibration to the WorldClim baseline. "SSP370… business-as-usual" is debatable. "Physical/mental health outcomes" are listed among UCM co-benefits but only mortality is modelled. Change "Urban Cooling mode" to "model". Change "ECOTRESS" to "ECOSTRESS".
16. **Duplicate references.** Refs 45 and 53 are duplicates.
17. **InVEST version.** InVEST 3.20.2 is cited as 2026; ensure the version actually used is reported, together with the UCM options (e.g., "factors" versus "intensity" cooling-capacity method).
18. **Sup. Fig. 1 legend** labels are unreadable. Several SI captions refer to "the same as Sup. Fig. 2" for the LULC colours, but Sup. Fig. 2 is a temperature plot.

---

## 6. Reproducibility checklist

| Item | Reported? | Where | Gap |
|---|---|---|---|
| LULC source and resolution | Partly | Methods, LULC | Year of LCM; canopy rasterisation rule; treatment of canopy over Urban pixels |
| Canopy reclassification rule | Partly | Methods; Sup. Fig. 1 | Threshold, years, accuracy assessment |
| UCM version and options | Partly | Ref. 36 | Cooling-capacity method; weights; night-time off; exact version used |
| Biophysical table | Yes | Sup. Table 2 | Woodland shade = 0 appears erroneous; albedo source and statistic |
| T_ref derivation | Partly | Methods | Stations not rural; period and statistic not stated |
| UHI_max derivation | Partly | Methods; Sup. Table 3 | Surface vs air UHI; average vs maximum switch |
| Blending and cooling distances | Yes | Sup. Table 3 | No justification or calibration |
| Reference ET | Partly | Sup. Table 3 | Dataset version, GCM, SSP |
| RH | Partly | SI; Sup. Table 3 | Formula ambiguity; constant-RH assumption |
| Climate products | Partly | Methods; Figs 1, S2, S12 | GCM list inconsistent; CHELSA version and product unspecified; baseline mismatch |
| Scenario Generator parameters | No | — | Full settings and transition matrix |
| TreeRisk procedure | Partly | Methods | Crown size, risk threshold, SSP/period for reported results |
| TreeOpp procedure | Partly | Methods; Sup. Fig. 5 | Class list, density |
| Target procedure | Mostly | SI | UTCI source and extent; crown size; realised area |
| Green–Target area matching | No | Asserted in Methods | Table of realised added canopy and source classes |
| Validation of T_air | No | — | Entire validation step |
| Energy coefficients | Partly | Sup. Table 4 | Derivation not reproducible; office class; floor area |
| WBGT formula | No | — | Equation and inputs |
| Work-capacity function | Yes | Methods eq. | Grid resampling (10 m → 30 m); weighting |
| Mortality baseline and time base | Partly | Methods | Annual vs warm-season deaths; health grid |
| Population downscaling | Partly | Methods | Target grid, method |
| Uncertainty (physical chain) | No | — | No parameter, GCM or energy/work uncertainty |
| Sensitivity analysis | No | — | — |
| Code | Yes (link) | Code Availability | Not audited by me; should include run configurations and scenario rasters |
| Processed data | Promised | Data Availability | "Upon publication"; scenario LULC rasters and UCM outputs should be deposited (e.g., Zenodo with DOI) |
| Licensed data | Stated | Data Availability | GiGL is restricted; provide derived, non-identifying scenario rasters |

---

## 7. Requested additional analyses, ranked by priority

1. **Correct the parameterisation** (woodland shade; consistent UHI_max for baseline and future, based on air temperature) and re-run all scenarios. *(M2, M3)*
2. **Validate the UCM baseline** against London station and citizen-station air temperature, and secondarily against LST. Calibrate the key parameters. *(M1)*
3. **Fix the exposure time base for mortality**: warm-season or heat-day deaths, a daily temperature distribution, a non-COVID baseline. Consider night-time runs. *(M6)*
4. **Global sensitivity and uncertainty analysis** of the physical chain, plus propagation into energy and work capacity. Show the robustness of the Target > Green conclusion. *(M9)*
5. **Green–Target matching table**, a source-class decomposition, and an equal-area "street-pixel random" control scenario. *(M4)*
6. **Consistent climate baseline and a larger or performance-screened GCM ensemble**, or UKCP18. Consistent model lists. *(M5)*
7. **Energy model documentation and plausibility check** against DESNZ/National Grid data; floor area; offices; coverage. *(M7)*
8. **WBGT documentation**, fixed vapour pressure, hourly or daily application, and worker-relevant weighting. *(M8)*
9. **Population-weighted ΔT; MSOA- or age-specific mortality rates; reconciliation of all reported numbers** across abstract, text, figures and Extended Data. *(M10)*

---

## 8. Recommendation

**Major revision.** The concept and scenario design are valuable. The relative finding, that allocation strongly changes who benefits, may well survive. The absolute temperature, mortality, energy and work-capacity magnitudes that headline the abstract, however, currently rest on:
- an unvalidated temperature model;
- a UHI_max that appears to come from a surface-temperature product and is switched between baseline and future;
- a parameter table that gives woodland no shade;
- impact calculations that apply a July-afternoon ΔT to time bases (annual deaths, monthly energy, hourly WBGT) for which it is not defined.

If the revision cannot demonstrate model skill and robustness of the Green–Target contrast, I would not support publication.

---

## 9. Confidential comments to the editor

- The paper is ambitious and policy-relevant, and the Target-versus-Green comparison is a genuine contribution. My main worry is that the **absolute numbers are likely overstated by a large factor**, and these are the numbers that will be quoted in policy settings. Three things drive this:
  - UHI_max = 5 °C taken from what appears to be a surface-UHI source. All land-use effects scale linearly with this value.
  - A July daytime ΔT that appears to be applied to *annual* all-cause deaths. My back-calculation: 387 deaths ÷ AF ≈ 0.68% implies ≈57,000 baseline deaths, i.e., London's annual total.
  - Energy coefficients that imply a per-degree sensitivity far above what UK commercial cooling would support.
- Sup. Table 2 assigns zero shade to woodland while the Green scenarios convert woodland to "Tree Cover". This looks like a straightforward error that could materially change the scenario results.
- The submission contains multiple versions of several figures with different values (e.g., Green30 ΔT −0.37 vs −0.34; work capacity +2.19 vs +1.19; deaths 377 vs 387). The authors should confirm that every figure comes from a single, final pipeline run.
- No observational validation of the physical model is presented, although London is one of the best-instrumented cities for urban climate. I would regard validation as a condition of acceptance.
- I did not audit the GitHub repository. I recommend asking the authors to deposit run configurations and scenario rasters with a DOI.
- I have no competing interests with the authors.
