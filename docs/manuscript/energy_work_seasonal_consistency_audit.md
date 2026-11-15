# Energy and work-capacity consistency with seasonal health

12 November 2026. Read-only methodological and input audit. No model reruns, output replacements, manuscript edits or push. Numerical examples below are illustrative calculations, not new London estimates.

## Decision summary

Follow-up: the [energy scaling and allocation correction test](energy_scaling_correction_test.md) evaluates a London-scaled pooled candidate and mapping sensitivities separately from temperature changes. This original audit remains the record of the issue; production coefficients have not been replaced.

Keep health reporting as **deaths averted per modelled June–August summer**. Do not force the other outcomes into seasonal totals simply to match its units. First resolve the energy calibration issue below; then pilot daily work capacity using the corrected temperature inputs. Existing energy and capacity figures remain comparison versions, not newly validated seasonal estimates.

| Component | Current implementation | Required distinction or update |
|---|---|---|
| Health | Corrected spatial temperatures; WorldClim-aligned daily reconstruction; threshold response; June–August mortality | Conditional summer estimate with fixed 2021 population/mortality |
| Energy | Linear building-temperature response at 25/5°C; input coefficient represents an assumed 30-day month | Monthly units are traceable, but national-to-London calibration and sector mapping need repair |
| Heavy-work capacity | Hothaps response to a single modelled temperature field and 45% relative humidity | Fixed-condition capacity index, not a summer or whole-working-day average |

## 1. Energy: an input-calibration issue, not just a time-unit issue

The preparation workbook converts a **Britain-wide 350 MW/°C** demand response to `350 × 30 × 24 × 1,000 = 252,000,000 kWh/°C` per assumed month. Sector shares of this national quantity are divided by **London building areas**. No London demand-share adjustment appears in that formula chain.

The original [Electric Insights analysis](https://reports.electricinsights.co.uk/?noamp=mobile&p=462) describes Britain-wide daily-average weekday electricity demand during 2016–2018, increasing by approximately 350 MW/°C above 20°C. It is not a London-specific, building-specific or daily-maximum-temperature response. Applying it throughout 30 days also extends a weekday association to all hours/days.

Workbook sheet `calculation` provides the following reproducible checks:

| Cells | Calculation / cached result | Interpretation |
|---|---|---|
| E32 → G32 | 350 → 252,000,000 | National MW/°C converted to a 30-day energy sensitivity |
| C2, D2, E2, G2 | 0.512; 129,024,000; 13,044,940.743; 9.890731 | Office share divided by London Community Services + Recreation/Leisure area |
| C3, D3, E3, G3 | 0.188; 47,376,000; 1,538,349.714; 30.796638 | Retail allocation |
| C5, D5, E5, G5 | 0.18; 45,360,000; 2,363,935.746; 19.188339 | Commercial denominator includes Mixed Use + Utilities |
| C6, D6, E6, G6 | 0.01; 2,520,000; 221,990,512.806; 0.011352 | Residential allocation |

Areas are m²; final coefficients are kWh m⁻² °C⁻¹ per assumed 30-day month. Formulas and cached values were inspected without recalculating or saving the workbook.

Additional mapping issues:

- The 51.2% **office** share is applied to Community Services and Recreation/Leisure. The workbook separately lists 6.2% for arts/leisure/community; this crosswalk needs justification or correction.
- The commercial coefficient uses Mixed Use plus Utilities in its denominator, but the active CSV assigns the coefficient only to Utilities; Mixed Use is zero.
- Several other building classes have zero coefficients. These represent exclusions or missing allocation, not evidence of zero cooling demand.

**Implication:** the previous absolute £ figures should not be treated as cleared for the final headline. A correction factor cannot be defensibly specified until London scaling and the sector crosswalk are resolved. This does not establish the exact size of bias or whether scenario rankings survive recalibration.

### What a new climate reconstruction does—and does not—change

Installed InVEST 3.20.2 calculates building energy value as consumption coefficient × area × tariff × `(Tref + UHImax − building temperature)`. For paired runs with identical settings/buildings, the difference therefore reduces to:

`benefit = coefficient × area × tariff × (Tbaseline − Tscenario)`.

A common additive daily climate offset cancels in this **linear** contrast. Thus WorldClim alignment alone does not make a new seasonal energy result; corrected NoData handling can change the underlying cooling contrast. A thresholded cooling-demand calculation or changing usage coefficients would no longer have this simple invariance.

### Feasible energy update

1. Prefer a London-specific demand–temperature relationship. If unavailable, test a transparent proxy scaling the Britain-wide slope by a compatible London/Britain electricity-demand share; do not mix UK and Britain denominators. This remains a transfer assumption, not local calibration.
2. Reconcile each source sector with the building classes, including zero/excluded categories. Preserve the original table and document a new coefficient table.
3. Use corrected building-temperature contrasts and the existing unique-building accounting, retaining explicit tariff assumptions. First report a conditional 30-day warm-period estimate.
4. Only label an estimate “per summer” after accounting for its exposure metric, above-threshold days, usage and weekday/weekend assumptions. The source daily-average exposure cannot silently be replaced by health-model Tmax. Do not simply multiply the existing monthly result by three.

If calibration cannot be resolved before submission, omit the absolute monetary headline or report a clearly labelled building-weighted cooling-service indicator instead. That is a change of outcome, not a substitute £ estimate.

## 2. Work capacity: daily nonlinear evaluation is necessary

The current modified work-capacity component applies the Hothaps response downstream of InVEST's unchanged cooling/WBGT functions:

`C(W) = 0.1 + 0.9 / [1 + (W / 30.94)^16.64]`.

Here W is the model's simplified WBGT estimate from air temperature and relative humidity. The production runs use RH = 45%. Scenario differences in C are multiplied by 100 and reported in **percentage points**, area-weighted within London—not worker-weighted and not measured labour output.

Because this response is nonlinear, calculating capacity at a mean temperature is not equivalent to averaging daily capacity. For illustration, with RH = 45% and 0.3°C cooling, capacity gains are approximately 0.028, 0.975 and 3.235 percentage points at 22, 28 and 34°C, respectively. Two equally weighted days at 22 and 34°C give a mean gain of 1.632 points, versus 0.975 points when evaluated at their mean temperature of 28°C. These are formula demonstrations, not London predictions.

### Feasible first pilot: baseline, Green30 and Target30

1. Reuse corrected spatial temperatures and the health reconstruction's **unlagged `offset_c`**, not its `lag01_offset_c`. Work capacity is not a lagged mortality response.
2. Calculate WBGT and Hothaps capacity for each reconstructed day, then scenario-minus-baseline changes. Average equally over the 92 summer days and the retained weather-pattern/GCM combinations; do not use mortality weights.
3. Retain the London area-weighted estimand and shared valid coverage across capacity scenarios. Report coverage rather than silently adopting the narrower health mask.
4. Retain RH = 45% as an explicit legacy assumption for the first comparison, with 35% and 55% as **assumed sensitivity settings**, not observed uncertainty bounds. Prefer time-matched temperature/humidity if available; pairing daily Tmax with daily-mean RH needs justification.
5. Compare with the corrected fixed-25/5°C calculation before extending to all nine scenarios.

**Interpretation limit:** this would be a summer average of **daily-maximum-temperature-based capacity**, not average capacity across working hours. The simplified WBGT calculation does not explicitly resolve solar radiation or wind. It must not be described as validated occupational exposure or converted into worker-hours/economic output without additional data. Retaining the fixed-condition capacity index in the main figure and using this daily pilot as sensitivity is a feasible alternative.

## 3. Spatial and version consistency

All ten inspected valuation manifests (baseline plus nine scenarios) use InVEST 3.20.2, Tref = 25°C, UHImax = 5°C, RH = 45% and July evapotranspiration inputs. They predate the corrected temperature family now used for seasonal health. Manifest paths also retain pre-rename locations; these are provenance records, not current path configuration.

The November boundary/accounting correction remains useful: capacity uses a common London cell-centre mask; energy sums unique buildings and conserves borough allocations. It does not repair temperature inputs or energy coefficients. Cross-boundary buildings retain their whole-building value allocated among London intersections—an explicit convention. New corrected building-temperature coverage must be checked before reuse.

## Recommended order

1. **Resolve energy scaling and sector mapping before approving absolute costs.**
2. Run the separate three-scenario daily-capacity pilot described above; inspect magnitude, ranking, humidity sensitivity and coverage.
3. Agree on outcome labels and temporal scope, then authorize revised energy/capacity runs and Figures 4–5. Health can remain seasonal even if the other panels retain clearly labelled conditional estimates; these must not be presented as three equivalent summer totals.

No energy/capacity rerun or update to existing figures is included in this audit. Previously reported rankings have not been revalidated under repaired coefficients or daily capacity evaluation.

## Sources and reproducibility

Data paths below are relative to the configured **Wellcome Trust Project Data** root:

- `1_preprocess/UrbanCoolingModel/OfficialWorkingInputs/energy_buildings/_UCM_Energy Consumption Table.csv` — active coefficients and tariffs.
- `1_preprocess/UrbanCoolingModel/OfficialWorkingInputs/energy_buildings/supporting_docs/_UCM_Energy Consumption Table_prep.xlsx` — formula chain and sector allocations; supporting documents in the same folder were inspected read-only.
- `2_postprocess_intermediate/UCM_official_runs/production/equal_area_2026-09/03_energy_work_capacity/` — valuation manifests and existing outputs.
- `2_postprocess_intermediate/UCM_official_runs/diagnostics/nine_scenario_health_readiness_2026-11-12_v2/shared_daily_offsets.csv` — reconstructed daily offsets; no humidity series.

Implementation inspected: [Hothaps calculation](../../code/post_processing_layers/calculate_hothaps_workability.py), [spatial accounting](../../code/post_processing_layers/valuation_accounting.py), [valuation summaries](../../code/post_processing_layers/summarize_ucm_valuations.py), and [seasonal-health calculation](../../code/health_assessment/run_seasonal_health_corrected.py). The installed InVEST 3.20.2 `urban_cooling_model.py` energy and WBGT equations were also inspected. See the [InVEST model documentation](https://storage.googleapis.com/releases.naturalcapitalproject.org/invest-userguide/latest/en/urban_cooling_model.html) and [seasonal-health results](seasonal_health_rerun_comparison.md).

No figures were generated for this audit; the numerical example evaluates the documented formulas directly.
