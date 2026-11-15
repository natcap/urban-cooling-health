# Pre-submission action plan: low-effort fixes for the most important review issues

12 November 2026 · Based on both simulated review sets (`peer-review-claude/`, `peer-review-chatgpt/`), the project `TODO.md` and audit notes, and a check of `London urban cooling-ms-1112.docx` (accepted-change text). These are simulated reviews, not journal decisions.

## 1. Where the two review sets agree and differ

Both review sets found the same core problems. The Claude panel reached a harsher verdict (reject and resubmit) mainly because it proposed rebuilding the models. The ChatGPT editor recommended a small, evidence-led validity audit plus reporting fixes. For a deadline, the ChatGPT route is the right one, provided the single real validity problem (the mortality time base) is fixed rather than only reworded.

| Issue | Claude | ChatGPT | Status in project / ms-1112 | Plan |
|---|---|---|---|---|
| July ΔT applied to **annual** 2021 deaths | Essential (E1) | Priority 1 gate | Open; ms-1112 still says "per year" | **Fix now**: one rescaling of the denominator, no UCM rerun (§2, A1) |
| UHI 2 → 5 °C; 5 °C not a daytime air-UHI value | Essential (E2/E3) | Priority 1 gate | Station pilot and 2/3/5 °C sensitivity done; rankings identical; author kept 25/5 °C as a stated stress test | **Report, don't rerun** (A3) |
| Woodland Shade = 0 | Editor marked "confirmed" from Sup. Table 2 | Production file reportedly has Shade = 1 for woodland; supplement table is wrong | Confirm against the run manifest | **Table correction** if confirmed (A4) |
| Equity metric is circular (SVI used for targeting and evaluation) | Essential (E5) | Partly expected by design; describe as strategy package | Open | **Low-cost added metric + reframing** (A5) |
| Older-adult trade-off (28.7% → 59.0%) missing from Abstract | Essential (E8) | Headline item | Abstract in ms-1112 still omits it; conclusion still generalises | **Text fix** (A2) |
| Target scenarios missing from Table 1; canopy matching not shown | Essential (E7) | Priority 2 | Open | **Table addition** from existing outputs (A6) |
| Exposure–response function: global, no threshold, elderly-derived cause-specific RRs | Essential (E1c–e) | Priority 2 source check | Open | **Document and caveat**; optional quick heat-only check (B2) |
| Age-specific mortality / susceptibility | Essential (E4) | Not required | Open | **Defer; state limitation** (C) |
| Control allocations (heat-only, random) | Essential (E6) | Not required if claims are framed as strategy packages | Open | **Defer; narrow the claim** (C) |
| Full UCM validation / Morris sensitivity | Essential (E3) | Not mandatory | Partial (station pilot) | **Summarise existing pilot in SI** (A3); full validation deferred |
| NoData 255 → 0 sensitivity in Green30 | Not raised | Priority 1 gate | Audit shows −0.05 °C (borough mean) effect; cause undiagnosed | **Short diagnosis** (B1) |
| Legacy embedded images, number mismatches | Essential (E9) | Priority 2 reporting | Many fixed in ms-1112; some remain | **Clean-up pass** (A7) |

## 2. Why the mortality fix is lower-effort than the Claude review implied

The production code (`code/health_assessment/health_modeling_v2.py`) computes, per pixel,

`excess = (1 − exp(−β·ΔT)) × D`, where D = 2021 registered deaths allocated by population.

Excess deaths are **linear in D**. Replacing annual deaths with July deaths (the month ΔT was modelled for) is therefore a pure rescaling:

- If you apply one London-wide July (or June–August) share of annual deaths to every borough, every absolute health number scales by the same constant k. Per-million values, Fig. 6 curves and the Fig. 7 thresholds scale by k too. **All rankings, Green–Target contrasts and the 45.5% → 4.5% and 28.7% → 59.0% shares stay exactly the same.** Only the axis values and headline counts change.
- If you use borough-specific July deaths, rerun only the health script (minutes). No InVEST rerun is needed.

Rough magnitude (verify with ONS before use): London had about 57,000–60,000 registered deaths in 2021, so July is about 7–8% of annual deaths. Target30's 496 "annual" deaths would become roughly **35–40 deaths in a July**, or roughly **110–120 over June–August** if you assume the July ΔT applies all summer. That is the range UKHSA reports for London heat-associated deaths in typical summers, which answers the plausibility objection raised by A1, B1 and B2.

Related one-line issue: the code uses AF = (RR − 1)/RR × D₀, while with D₀ as baseline deaths the direct contrast is D₀(RR − 1). At βΔT ≈ 0.008 the difference is under 1%. Either switch to D₀(RR − 1) or state that the two are equivalent to first order. Do this in the same health rerun.

## 3. The plan

Effort: **S** = under half a day, **M** = 1–2 days, **L** = more than a week (defer). The schedule assumes about 2 weeks; compress A and drop B if you have less.

### A. Must do before submission (about 4–6 working days in total)

**A1. Correct the mortality time base (M).** Closes the biggest shared criticism.
1. Download ONS monthly registered deaths for London (2021 and, ideally, a 2016–2019 or 2022–2023 July average, so the result does not rest on a COVID-affected year). Choose July-only as the main analysis and June–August as a sensitivity, or the reverse. State the assumption either way.
2. Rescale D (or rerun the health script with monthly mortality rasters), and switch to D₀(RR − 1).
3. Regenerate Fig. 4c, Fig. 5c, ED Fig. 1, Fig. 6 and Fig. 7 legends and axis values. Thresholds and shares should be unchanged; check this as a test.
4. Relabel everywhere: "modelled heat-related deaths averted in July (or summer)" instead of "per year". Add one sentence benchmarking the baseline burden against UKHSA/ONS London estimates.

**A2. Rewrite the Abstract and conclusion (S).** Replace the death numbers with A1 outputs and state the period. Add the older-adult trade-off. Replace "feasible" with "spatially screened" and "conventional" with "proximity-based". Label AllBuilt as an extreme bounding case. Remove "integrated … climate projections" and describe 25/5 °C as a prescribed warm-condition setting. Fix the final Discussion sentence on older populations so it applies only to Green.

**A3. Report the climate/UHI decision (S).** Text from `ucm_climate_parameter_pilot.md` is ready:
- Methods and Sup. Tables 1 and 3: describe 22/2, 22/5, 25/5 and 28/5 °C as prescribed settings, and add 2030 to Sup. Table 1.
- Add a short SI section on the station and Crowd-Grid pilot and the 2/3/5 °C ranking sensitivity. Note that absolute ΔT, and so all downstream outcomes, scale roughly linearly with UHI_max. This answers B1-M1/M2 without a rerun.

**A4. Correct the reporting tables (S).**
- Sup. Table 2: match the shade values to the production biophysical file and cite the run manifest. If production really used Shade = 0 for woodland, escalate it, because it would change results.
- Sup. Table 4: add units.
- Write out the WBGT and RH equations as actually coded.

**A5. Reduce the circularity charge (S–M).** No new modelling is needed:
- Keep 45.5% → 4.5%, but call it a design-consistency check.
- Add one independent indicator not in the SVI or the targeting score, such as the IMD 2019 Health Deprivation and Disability domain or overall IMD rank, joined at LSOA level. Report the Green30 vs Target30 low-benefit share on it.
- Add a line on resident-weighted shares.
- Describe Target as an allocation *strategy package* (street-tree pool + heat + SVI), not an isolated effect of vulnerability weighting.

**A6. Complete Table 1 (S).** Add Target10/20/30, and add realised added canopy (km² and %) for all Green and Target rows from the existing equal-area audit. This shows the matching.

**A7. Consistency and clean-up pass (S).** Work through the ChatGPT Priority 1 list in `TODO.md` (Fig. 1 contrast wording, "six" → "nine" scenarios, figure numbering, "July mean daily maximum temperature") and the Claude Tier 3 checklist. Purge legacy images from the .docx (the uncaptioned images after Fig. 7 and superseded panels). Fix the duplicate references (45/53) and incomplete ref. 19. Add a short Statistical analysis subsection (Dunn's tests, GAM settings, intervals reflect β only).

### B. Do if time allows (about 1–3 days)

**B1. NoData diagnosis (S–M).** Document which NoData semantics production uses and why. Rerun only if it is a bug.

**B2. Exposure–response sensitivity (S).** Complete Table 2 (source population, metric, lag, threshold). As a quick bound, recompute all-cause deaths with an alternative London or UK heat slope. This is again a scalar change in β, with no rerun. Report it in the SI.

**B3. Limitations paragraph (S).** Cover:
- daytime outdoor air temperature versus night-time and indoor exposure;
- fully grown canopy assumed;
- borough mortality not age-standardised;
- green gentrification.

### C. Defer to revision (state as limitations; reviewers can request them)

These items are listed so you can say in the cover letter or limitations that you know about them:
- Age-specific (indirectly standardised) LSOA mortality.
- Control allocations (heat-only, random street-tree) to decompose why Target beats Green.
- Full UCM validation and Morris or joint uncertainty analysis.
- Night-time UCM runs.
- Energy module rebuild with floor area and offices.
- Spatially explicit GAMs.

## 4. Suggested order

1. **Day 1.** A1 data download and rescale. Confirm shares are unchanged and the new headline numbers are plausible. In parallel, check the production shade file (A4).
2. **Days 2–3.** Regenerate the health figures and do A6 and A5.
3. **Day 4.** A3 and A4 text and tables. Then A2 (Abstract and conclusion last, once the numbers are final).
4. **Day 5.** A7 clean-up and a final number-to-figure check (the TODO claim-to-CSV table).
5. **If there is time.** B1–B3.

## 5. Open decisions for the author

1. **Main health period: July only or June–August?** July only is the most defensible, because July ΔT was what was modelled. June–August gives the more policy-relevant magnitudes but rests on an added assumption.
2. **Baseline mortality year:** 2021 only, or a multi-year non-pandemic July average?
3. **Independent equity indicator for A5:** IMD Health domain, overall IMD, or the GLA Climate Risk index (only if it was not a targeting input)?
4. **Target journal:** confirm Nature Health. Deferring the C items is reasonable at submission, but expect at least age-standardisation and the control allocations to come back in review.

---

## Addendum (12 November): responses to author questions

### Q1. Is age-specific mortality modelling infeasible?

**Not fully infeasible. A partial version uses only public data you already work with.**

- **Borough deaths by age:** the Nomis dataset the pipeline already uses (`mortsa`, "Mortality statistics – underlying cause, sex and age") offers 5-year age bands by local authority and cause. The project export (`data/health-data/35915561408648160_data.tsv`) selected only "total (all ages)", so this is a re-download, not a new source.
- **LSOA population by age:** Census 2021 TS007/TS007A gives this. The project already holds TS001 from the same release.

**What's feasible (about 1–2 days):** indirect age standardisation.
- Compute borough age-specific death rates, apply them to each LSOA's age structure, then rescale so the borough totals still match.
- This replaces "one crude rate per borough, spread by population". The health script is then rerun, with no InVEST rerun.
- It answers B2-M4 and A1-M3: the over-75 and outer-borough patterns would then reflect where the older residents actually live.

**What really is limited by data:** age-specific heat sensitivity.
- The all-cause RR (Yang et al. 2024 meta-analysis) is all-age.
- The cardiovascular and respiratory RRs (Bunker et al. 2016) are elderly-specific, but the code applies them to all ages.
- Age-specific RRs for London or the UK would need a different ERF source.

**Suggested wording:** "Baseline mortality was age-standardised to LSOA age structure; heat sensitivity was not modelled as age-varying because the exposure–response estimates are not age-stratified."

**Low-effort alternative if there's no time:** keep the current approach and state clearly that LSOA death rates equal the borough crude rate. Then drop or soften the age-group equity statements, since in that case they reflect where trees were placed, not differences in heat susceptibility.

### Q2. Can the analysis be framed as summer rather than July?

**Yes, and the model structure supports it.**

- In InVEST UCM, `T = T_ref + (1 − HM) × UHI_max`. The land-use contrast is therefore `ΔT = −ΔHM × UHI_max`, which does not depend on `T_ref`. The project's own check (`matched_2050_temperature_test.md`) confirmed this to about 10⁻⁵ °C.
- HM depends on land cover (shade, albedo, Kc), the ET index and the distance parameters. The ET index is normalised by the maximum ET₀ in the raster, so a different summer month's ET₀ has a second-order effect.
- So **ΔT represents a typical summer-daytime land-use contrast for a given UHI_max**, whatever the month. Running June and August would mainly reproduce the same ΔT.

**What would change:**
- Only the absolute temperatures in Fig. 1/3a and the WBGT/RH inputs for work capacity.
- The energy module if its coefficients are July-specific; it is currently reported per month.

**Recommended:**
- Describe the experiment as "summer (June–August) daytime conditions, parameterised with July, the warmest month".
- Optionally add a JJA-mean panel or row to Fig. 1. That is a cheap WorldClim extraction.
- Do not claim that UHI_max = 5 °C is representative of all summer days. The station pilot says it is an upper-tail stress test.

**Wording correction:** WorldClim `tmax` is the *monthly mean of daily maximum temperature*, not a "monthly maximum". Use "July (or JJA) mean daily maximum temperature" throughout, as TODO item 1(e) already notes.

### Q3. A workaround for summer ΔT applied to annual deaths

Excess deaths are linear in the death denominator, so this is a denominator choice, not a remodelling. Recommended approach: a main estimate plus a bound.

1. **Main estimate: summer deaths.**
   - Use JJA registered deaths by borough from the ONS dataset "Deaths registered monthly in England and Wales", which lists London boroughs.
   - Use 2021, or better a 2018–2019 + 2022–2023 JJA average to avoid the COVID-affected year.
   - `ED_summer = AF × D_JJA`. Label it "modelled heat-related deaths averted per summer".
   - Roughly, D_JJA is about 22–24% of annual deaths, so the headline numbers fall about four- to five-fold. Rankings and the equity shares stay the same if a London-wide share is used, and change only slightly with borough-specific JJA counts.
2. **Lower bound: hot days only.**
   - The Yang (2024) and Bunker (2016) RRs are heat-exposure slopes, typically estimated above a heat threshold or during hot periods; check the exact definition in each paper.
   - Applying them to every summer day overstates the effect. As a sensitivity, use `D_hot = D_JJA × (share of JJA days with daily mean temperature above the London threshold)`.
   - The share can come from the MIDAS 2018–2023 station data already downloaded, or from HadUK-Grid. Assuming deaths are spread evenly within each month is acceptable for a bound.
3. **Report** the summer estimate as the headline and the hot-day estimate as the lower bound in the Abstract or Fig. 4. Add one sentence comparing the baseline summer heat burden with UKHSA heat-mortality monitoring for London.
4. **Avoid:** dividing by 12; "annual" wording; claiming the 95% intervals capture total uncertainty.

### Q4. What is the NoData issue?

Source: `docs/manuscript/figure3_ranking_input_audit.md` (11 November).

**What NoData is.** Every raster has a reserved "no data" value marking cells outside the study area or with no land-cover information. The old Green scenario rasters used **255**; the revised scenarios use **0**.

**What the audit found.**
- The historical Green30 land-cover raster was rerun with *only* the NoData code changed from 255 to 0, on the same InVEST version with identical valid cells and inputs.
- Green30 cooling fell from 0.374 °C to 0.324 °C (borough mean), or from 0.364 to 0.288 °C (area-weighted).
- The model outputs differ on the *same valid London cells*, so this is not just a masking or summary difference.
- The likely mechanism is how InVEST's distance-weighted (convolution) steps treat edge and NoData cells. The cooling-capacity and green-area kernels spread information up to 450–500 m, so a cell coded as "no data" versus a cell coded as class 0 ("Nothing" in Sup. Table 2, with zero shade, Kc and albedo) can change neighbouring values. This mechanism has not been confirmed.

**Why it matters.**
- This one encoding change was large enough to flip the historical Green30 vs Target30 cooling ranking (Green was ahead by 0.007 °C; after the change Target is ahead by 0.042 °C).
- The current production outputs all use the revised encoding consistently, so the manuscript's numbers are internally consistent.
- However, a reviewer could ask whether the Target > Green result depends on edge handling.

**Low-effort action (about half a day).**
1. Confirm that in all current production rasters NoData = 0 is declared *as NoData* in the GeoTIFF metadata, and is not also used as land-cover class 0 "Nothing".
2. Check which cells carry it: only outside London or the water mask, or also inside the city.
3. Run one control: current Green30 and Target30 recoded to the other convention, or with a buffered domain, to show that the Target − Green difference is stable.
4. Report it in one sentence in the SI. Rerun only if (1) or (2) reveals a coding error.
