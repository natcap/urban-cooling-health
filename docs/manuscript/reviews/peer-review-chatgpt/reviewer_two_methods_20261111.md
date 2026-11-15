# Independent reviewer 2 — methods audit

## Scope and overall judgment

Independent simulated Nature Health peer review, not journal representation. The complete current manuscript and supplement text were read before checking `docs/manuscript/figure3_climate_uhi_audit.md` and `figure4_5_audit.md`. After drafting the independent assessment, I also read the coordinating reviewer's newly identified `figure3_ranking_input_audit.md`; its additional technical evidence is separately identified below. This is a text-and-design review, not independent execution or validation of the models. Embedded figure images and complete bibliographic claim support were not audited; no other reviewer report was consulted. No new model runs or source edits were performed.

**Recommendation: major revision.** Fixed-climate contrasts, matched realized canopy area, fixed demographics, paired uncertainty draws and explicit aggregation labels are sound design choices. They do not by themselves validate the absolute temperature or mortality estimates. Three issues are especially consequential: the climate/UHI linkage, July-to-annual mortality translation, and the reference-risk convention in the mortality equation. These should be resolved before relying on numerical health claims.

## Strengths

1. Holding demographic and climate conditions constant across land-use scenarios clarifies the controlled counterfactual question.
2. Count-preserving population resampling and population allocation of borough deaths are stated, with fixed 2021 denominators for equity comparisons.
3. Health intervals are explicitly limited to exposure–response uncertainty, and shared coefficient draws across scenarios preserve paired comparison structure.
4. Figure 7 uses the same neighborhoods and fixed benefit thresholds; legends distinguish neighborhood shares from resident shares and neighborhood age composition from age-specific risk.
5. Figures 4–5 distinguish area weighting, total costs and cross-borough building apportionment. The supplied text already contains substantive corrections documented by the earlier audit; these should not be requested again as though absent.

## Major comments

### M1. Climate context and urban-cooling scenarios are not yet a documented projection chain

**Anchors:** Results “Baseline T_WM was derived from observations from 2021”; Methods Climate scenarios; Sup. Tables 1 and 3; Figure 3 legend.

Figure 1 uses future WorldClim climatologies versus July 2021 CHELSA estimates, while the UCM is parameterized with 22/25/28°C reference temperatures and 2/5/5°C UHI. The baseline is both a different product and a single year. The Methods names EC-Earth3-Veg, but Sup. Fig. 12 names ACCESS-CM2. The text moves between July monthly Tmax, warmest-month temperature and peak heat exposure. No harmonization or rural-reference transfer is established.

**Required revision:** Provide a compact input/estimand table: product/version, variable, temporal averaging, baseline, GCMs, aggregation and role in each figure/model. Correct the GCM inventory. Either calculate defensible common-baseline anomalies and document the coupling to rural reference temperature, or clearly separate Figure 1 context from prescribed warm-condition experiments and stop assigning the latter calibrated SSP/year interpretations. Identify what Figure 1's spatial resolution does and does not resolve; downscaling a 1-km field does not independently recover neighborhood microclimate.

**Repository distinction:** The Figure 3 audit establishes that the old year labels came from parameter combinations and documents a replacement conditional comparison. The new Figure 3 exists, but the current manuscript still describes the old design. Correcting its integration is feasible without new runs; genuine climate-projection validation remains unresolved.

### M2. Establish air-temperature relevance of UHI and sensitivity to its prescribed magnitude

**Anchors:** UCM Methods, Yale-derived “average and maximum ... (2 and 5°C)”; Sup. Table 3, UHI rises 2→5°C; reference stations Northolt, Heathrow, Hampton, Kew and Kenley.

An average-versus-maximum contrast does not establish present-to-future intensification. The rural status, measurement period, aggregation and baseline choice for the listed stations are not supplied. The repository audit further identifies the Yale source as satellite surface-UHI statistics, whereas the model parameter is an air-temperature contrast. This is a physically material mismatch, not a citation detail. The UCM contrast depends on UHI magnitude, so common climate across scenarios does not remove its influence on effect sizes.

**Required revision:** Recover the exact extraction and statistic and justify air-temperature applicability, or label 5°C an uncalibrated prescribed value. Provide independent compatible air-temperature checks or a proportionate parameter sensitivity (including a lower UHI setting), distinguishing calibration from validation. Report sensitivity of health magnitude and strategy ranking, not only temperature. Document blend/cooling-distance choices and relevant calibration evidence. Do not call borough variation a model confidence interval.

**Repository distinction:** The Figure 3 audit supplies the model equation, notes approximately 0.4 scaling of temperature contrasts for 2 versus 5°C under fixed heat-mitigation fields, and replaces legacy CI bars with geographic distributions. This is an algebraic diagnostic, not validation or a downstream sensitivity run. The manuscript should not present it as either.

### M3. Justify the annual mortality estimand and match exposure to the epidemiological coefficient

**Anchors:** Health Methods, “summer (July) temperature increases”; annual 2021 registered mortality; Fig. 4 “annual total all-cause heat-related deaths averted”; Abstract 135–496 deaths/year.

A July mean-temperature contrast applied to a full annual mortality count does not automatically estimate annual heat-related mortality. The text does not specify heat-exposure duration, threshold/minimum-mortality temperature, seasonal deaths, lag structure, or whether the coefficient describes daily mean, maximum, apparent temperature or another exposure. Because the model uses only a temperature difference, the health benefit could be independent of absolute ambient temperature even when crossing a heat-risk threshold would matter. Relative comparisons are not automatically robust to these choices.

**Required revision:** Derive the target estimand from the original coefficient's exposure, population and time scale. Prefer an appropriately supported daily/seasonal heat-impact calculation if annual avoided deaths are to be reported. A bounded alternative is to label the output an annual-baseline mortality-equivalent scenario index and explicitly avoid forecasting annual heat deaths, provided that limitation is acceptable to the paper's aims. Simply multiplying or dividing by 12 is not a defensible repair. Give sensitivity to a London/UK-relevant function, threshold and plausible exposure duration where feasible. State how daytime-only cooling relates to daily/nighttime mortality exposure.

### M4. Resolve the attributable-fraction reference convention

**Anchors:** Health equations: `RR = exp(beta * deltaT)`, `AF = (RR − 1)/RR`, `ED = AF * D0`, with D0 defined as baseline mortality.

The formula `(RR−1)/RR` is the fraction attributable among deaths under the higher-risk/exposed condition. If `RR` is counterfactual divided by baseline risk and D0 is genuinely the baseline count, the direct contrast is `D_counterfactual − D_baseline = D_baseline * (RR−1)`, assuming the multiplicative model applies. Multiplying baseline deaths by `(RR−1)/RR` instead yields `D_baseline * (1−1/RR)`. These are different reference conventions, including for cooling where RR<1. The manuscript cannot simultaneously label D0 baseline deaths, define RR as scenario-versus-baseline, and leave this unexplained.

**Required revision:** Define exactly which temperature regime the registered deaths represent and derive the counterfactual equation from it, including the sign conversion to deaths averted. Check a simple positive and negative 1°C contrast by hand, then assess the numerical difference in the outputs. Do not change the formula solely on this review without confirming the intended estimand; if the existing expression is retained, explain why its reference population and direction are appropriate. Consistency tests in the repository are not evidence that the estimand is epidemiologically correct.

### M5. Validate coefficient transfer and distinguish modelled neighborhood burden from demographic susceptibility

**Anchors:** Table 2 cardiovascular/respiratory coefficients sourced to Bunker et al.; bibliography title “... in the Elderly”; Health Methods “all ages and sexes”; Fig. 6 borough-death allocation.

The text signals a possible age-population mismatch: elderly-derived cause-specific effects are applied to all-age borough deaths. Geographic/climatic similarity alone is not enough to justify transfer. In addition, allocating borough mortality in proportion to population produces a common within-borough baseline rate. Thus neighborhood benefit gradients are driven by cooling, borough rate and aggregation—not estimated neighborhood-specific age susceptibility. The existing captions acknowledge the age-specific boundary well, but the Discussion invokes biological vulnerability more broadly.

**Required revision:** Add a coefficient provenance table specifying exposure metric, threshold, lag, region, age range, endpoint and selected estimate. Resolve the elderly/all-age transfer through age-compatible input or explicit sensitivity/qualification. Explain that registered 2021 rates, including that year's exceptional mortality context, are held fixed rather than forecast. Test sensitivity to baseline rates or an alternative suitable year/rate where practical. Explain which aspects of the vulnerability gradients can arise mechanically from the allocation design; do not interpret the older-population contrast as differential physiological response. State that selected causes are not an additive decomposition of all-cause mortality.

### M6. Equal canopy area is necessary but insufficient to isolate equity prioritization

**Anchors:** Methods opening, “matched on realized added-canopy area to isolate ... spatial allocation”; Green conversion classes; Supplementary Methods street offsets and priority `UTCI_norm × SVI`.

The compared strategies differ in candidate land and planting geometry as well as prioritization. Green extends existing canopy across selected LULC classes; Target samples streets and footpaths. Target excludes invalid UTCI extent, which may alter geographic coverage. Candidate-point filters cannot establish soil, utilities, street-width, accessibility, ownership or survival feasibility. The million-plus proposed trees also require explicit mature-crown versus planted-tree accounting.

**Required revision:** Report baseline canopy and realized added area for every scenario, matching tolerance, land-cover transitions, coverage exclusions, crown-size and overlap rules, candidate counts and tie-breaking. Document UTCI source/model/date, inputs, extent and why 14:00 is suitable. If attributing gains specifically to heat/SVI prioritization, add an equal-budget same-candidate-set untargeted or randomized allocation comparator; otherwise narrow the inference to two complete spatial strategies. Report a limited priority-weight/crown/feasibility sensitivity, or clearly delimit these as untested design choices. Equal area must not be described as equal cost or implementability.

### M7. Specify SVI construction and separate design success from independent equity evidence

**Anchors:** Social vulnerability, “four key themes” followed by dropping housing and English proficiency; “four-dimension SVI” in legends; Supplementary S7 “SVI ... not an independent validation measure.”

The actual four-component index is not reproducible from the text: transformations, directions, weights, aggregation order, missingness and census/data vintages are absent. Retaining four indicators is not the same as representing all four CDC themes after housing is dropped. Target optimizes using SVI and is evaluated on SVI; improved alignment is useful but partly design-expected. Opposing age gradients also make the objective's value judgments consequential.

**Required revision:** Give the exact index formula and data sources, describe it as an adapted four-indicator index, and explain the weighting decision. Keep the non-independent-evaluation caveat in the main Methods/Discussion, not only a supplementary caption. Show component-level results (already largely present) and a modest alternate-weight or leave-one-component-out check if claiming robust equity improvement. Define equity as the reported distributional criterion; do not treat it as a comprehensive measure of justice.

### M8. Spatial inference needs a clear descriptive boundary or a dependence-aware robustness check

**Anchors:** Fig. 2 Dunn tests over 4,835 LSOAs; Fig. 6 robust scaled-t univariable GAMs and pointwise CIs; Results “not spatially adjusted or causal.”

The existing descriptive disclaimer is appropriate. Nevertheless, nearby LSOAs share smoothed temperature, borough mortality and spatially clustered covariates. A robust residual family does not correct spatial dependence, and Bonferroni correction does not address spatial pseudoreplication. Very narrow model-conditional bands or significance letters may overstate information content.

**Required revision:** Add an actual statistical-analysis subsection specifying smooth basis, basis dimension, diagnostics, effect contrasts and uncertainty interpretation. Check residual spatial dependence and use a proportionate spatial/borough or spatial-block robustness analysis for inferential claims. Alternatively prioritize descriptive effect sizes and distributions, explicitly retaining CIs only as conditional summaries. Compare spatially adjusted fits carefully: they answer a different question and should not silently replace the original descriptive estimand. Clarify that threshold shares are deterministic descriptive summaries, not significance tests, and optionally report population-weighted shares alongside neighborhood shares.

### M9. Co-benefit parameterization needs an auditable chain and bounded uncertainty

**Anchors:** Energy Methods, 2022 office/retail spikes and “12–16% ... per degree”; Sup. Table 4 points to a 2018 Drax report; Work equation; Sup. RH Methods, “(Tmax + Tmin)/2, or ... T = Tx.”

The derivation of building-specific monthly kWh/m²/°C coefficients is missing, and the source date does not obviously support the 2022 narrative. Footprint and conditioned floor area are different quantities. Residential AC is discussed as negligible while its coefficient is nonzero; unclassified categories have zero coefficients rather than documented missing-value treatment. Work capacity requires a specified WBGT calculation, but the supplied text gives only the response equation. RH temperature selection is explicitly ambiguous, and fixed historical vapor pressure is not future moisture projection.

**Required revision:** Provide a source-to-parameter derivation with units, price year, AC penetration, footprint/floor-area convention and zero/missing rules. Resolve the 2018/2022 citation chain. Specify the exact WBGT equation, radiation/wind/shade assumptions and chosen temperature/RH inputs. Keep the helpful area-not-worker-weighted label. Add targeted parameter sensitivities for energy coefficients and humidity/WBGT; do not assert the heavy-work choice is conservative without a defined bound. Avoid claiming net economic benefits in the absence of costs.

### M10. Complete the uncertainty and reproducibility contract

**Anchors:** Health Methods 2,000 coefficient draws; Figure 3 “95% confidence intervals for model estimates”; Data Availability “will be made available upon publication”; Code Availability mutable GitHub URL.

Exposure–response uncertainty is now labeled honestly, but uncertainty in UHI, climate inputs, canopy feasibility and mortality exposure interpretation may dominate it. The nominal three-GCM mean is not an uncertainty analysis; independence of modelling centers alone is not evidence of regional performance/spread. Shared coefficient draws should also be used to report the paired Target-minus-Green contrast if claiming a meaningful advantage, rather than comparing separate intervals.

**Required revision:** Use a compact uncertainty inventory distinguishing empirical coefficient uncertainty, input/model sensitivity, spatial variability and fixed assumptions. Report paired contrasts and identify whether signs/rankings survive the most consequential sensitivities. Avoid an indiscriminate full factorial exercise: first resolve M1–M4, then test the few assumptions with material leverage. Archive a versioned release with run manifest, input versions, parameters, grid/CRS/masks, environment, commands, scenario lineage and output checksums. Document accessible substitutes or authorized access steps for GiGL restrictions; provide nonrestricted summaries sufficient to verify major results. A public repository URL alone does not identify the exact analysis version.

### M11. Additional repository evidence: resolve material NoData sensitivity before numerical robustness claims

**Evidence boundary:** This is not a finding from the submitted Word text or my own rerun. The subsequently supplied `figure3_ranking_input_audit.md` reports an existing controlled Green30 comparison with identical valid land-cover classes, valid-cell mask, software and non-LULC inputs, changing only NoData encoding from 255 to 0. London-area-weighted cooling falls from approximately 0.3645 to 0.2875°C; the test alone is sufficient to reverse the 30% Green–Target ordering against the historical Target result. The internal numerical mechanism is not yet diagnosed. This is not equivalent to dropping a few boundary cells, and it is not established at all canopy tiers.

**Required revision:** Treat missing-data handling as a material computational-validity issue, not formatting. Establish the intended encoding semantics, identify the affected processing stage, and test invariance under equivalent missing-data encodings or justify the scientifically appropriate handling. Verify the selected production inputs and assess consequences for matched contrasts and downstream claims. The existence of a consistent revised output package does not by itself establish that its numerical behavior is correct. Report provenance and limits without attributing the reversal to canopy design, software version or area weighting alone. The source audit already contains detailed attribution; it need not be reproduced in the manuscript, but the input-validation resolution must be reproducible.

## Minor comments

1. Add the three Target rows to main Table 1 and reconcile the “six scenarios” prose with the nine-scenario analysis.
2. Specify whether 10-m LULC/targeting and 30-m work grids represent distinct stages, how they are reconciled, and how common masks and borough boundaries are handled.
3. Reconcile baseline LULC year 2021 with the UKCEH 2023 dataset citation, canopy dataset dates and 2021 census/mortality. Such a composite baseline may be acceptable but must be named accurately.
4. Define tree-risk mapping from species climatic unsuitability to assumed canopy loss and account for incomplete species-record coverage. “At risk” is not observed or probabilistically predicted death.
5. Explain the four missing LSOAs in Sup. Fig. 12 (4,831 versus 4,835) and whether their absence affects quintile comparisons; the manuscript need not imply all analyses have identical coverage.
6. Clarify mean versus area-weighted aggregation in Figure 1 and climate summaries. Distinguish London means, equal-borough means and borough maxima in every temperature claim.
7. Use percentage points for work-capacity changes; reconcile AllBuilt −6.37% in prose with −6.36 percentage points in the Figures 4–5 audit. Keep monthly energy units in Results, not just captions.
8. State random seed/distribution used for beta draws, log-scale CI conversion, and handling of asymmetric published confidence limits. Do not sum cause-specific mortality categories.

## Readiness and completed-versus-outstanding work

The supplied manuscript already incorporates the corrected 135–496 health range, 1,110 AllBuilt additional deaths, monthly energy, 1.42-point capacity gain, nine-scenario Figure 4 legend, fixed-threshold equity comparison and older-population trade-off. These are not missing revisions. The Figure 3 replacement and corrected aggregation-specific temperature values are documented as generated but not incorporated. Neither audit resolves annual mortality exposure duration, AF reference convention, UHI calibration, candidate-set attribution, or complete parameter provenance. Those are scientific readiness issues, separate from editorial integration and from any model-output consistency tests.
