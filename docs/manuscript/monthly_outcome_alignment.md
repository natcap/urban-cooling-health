# July or whole-summer reporting?

13 November 2026 · [Copy-friendly HTML](monthly_outcome_alignment.html)

## Recommendation

**July is a defensible single-month illustration, and the closest of the three tested months to the summer-average health and capacity benefits. It is not an unbiased substitute for summer.** For a manuscript claiming seasonal benefits, retain June–August (JJA) health and use JJA daily-Tmax capacity at RH45 as the preferred capacity candidate; retain July and the fixed 25/5°C calculation as sensitivities or benchmarks. Energy remains a conditional 30-day proxy and must not be relabelled as a validated monthly or seasonal estimate.

No main figure, Word document or production result was replaced. This diagnostic tests baseline, Green30 and Target30, not every land-use scenario. Health includes all five outcomes; the main visual comparison uses all-cause mortality.

## What the tests show

![Monthly climate, capacity and health comparison](</Users/yingjiel/Library/CloudStorage/GoogleDrive-yingjiel@stanford.edu/Shared drives/Wellcome Trust Project Data/2_postprocess_intermediate/UCM_official_runs/diagnostics/monthly_outcome_alignment_20261113_v01/monthly_outcome_alignment_20261113_v1.png>)

**Reading the figure:** points/lines show means across three GCMs and five weather patterns. Shading and vertical ranges show member minima–maxima, not confidence intervals. JJA is a day-weighted mean, not a fourth month. Health is divided by calendar days for fair monthly comparison. Capacity is an average percentage-point gain, not an accumulating monthly total. Energy is not plotted because the current calculation has no month-specific exposure response.

| Period | Baseline mean daily Tmax (°C) | Capacity gain, Green30 / Target30 (pp) | All-cause deaths averted, Green30 / Target30 | Deaths averted/day, Green30 / Target30 |
|---|---:|---:|---:|---:|
| June | 22.53 | 0.255 / 0.297 | 6.68 / 8.76 | 0.223 / 0.292 |
| July | 25.52 | 0.582 / 0.668 | 15.65 / 20.59 | 0.505 / 0.664 |
| August | 25.46 | 0.651 / 0.733 | 16.05 / 21.05 | 0.518 / 0.679 |
| JJA | 24.53 | 0.499 / 0.569 | 38.37 / 50.40 | 0.417 / 0.548 |

- July is the warmest **mean** month in this reconstruction, only 0.064°C warmer than August. August's mean temperature is slightly closer to the JJA mean.
- July is closer than June or August to the **pooled summer-average benefit** for both scenarios: capacity is nevertheless 16.7–17.4% higher and health benefit/day 21.0–21.3% higher than JJA.
- August benefits slightly exceed July. Response nonlinearity and daily distributions matter; monthly mortality allocation also affects health. These tests do not isolate each mechanism's contribution.
- July is not the closest month in every member: it is closest for health in 8/15 GCM–weather combinations and for capacity in 12/15 (both scenarios); August is closest in the remaining members. Thus the pooled result is not universal.
- Target30 exceeds Green30 in each month's pooled estimates. This comparison alone does not establish stability of every scenario ranking or neighbourhood equity curve.

External context is consistent with July being a hot-month choice: Heathrow's 1991–2020 mean daily maxima are 21.57°C in June, 23.89°C in July and 23.40°C in August. These station historical normals are **not** the future citywide values above and do not validate the reconstruction. [Met Office station normals](https://www.metoffice.gov.uk/research/climate/maps-and-data/location-specific-long-term-averages/gcpsveqyg).

## Why the current outcomes are not already “July estimates”

| Outcome | What is currently calculated | What calendar alignment would require |
|---|---|---|
| Energy | London-scaled demand-response proxy × spatial cooling × an assumed 30-day duration | Daily-mean exposure, the source response's threshold and operating-day assumptions for the selected month/season |
| Main capacity candidate | Hothaps capacity calculated from reconstructed JJA daily Tmax at RH45 | Already calendar-aligned with health at the season level; specify that this is daily-Tmax-based, not working-hour capacity |
| Health | Daily threshold/lag response integrated over June–August with monthly mortality weights | Select July outcome days directly, retaining preceding-day lag exposure and July mortality weights; do not use JJA/3 |

The selected main capacity candidate is the JJA daily-Tmax result: Green30/Target30 gains are 0.499/0.569 pp. The fixed-condition benchmark gives 1.199/1.428 pp, whereas July daily-Tmax gains are 0.582/0.668 pp. These are distinct estimands; none should be relabelled as another period.

The JJA reconstruction already uses month-specific WorldClim temperatures and daily weather anomalies. It does not repeat July climate. The retained cross-month assumption is the **July-derived spatial temperature/cooling field**, with unchanged UHI, ET0-derived cooling pattern and assumed humidity. Separate June/August UCM or humidity validation was not performed here. InVEST requires climate inputs for the period of interest; its framework does not itself establish a calendar-month attribution. [InVEST documentation](https://storage.googleapis.com/releases.naturalcapitalproject.org/invest-userguide/latest/en/urban_cooling_model.html).

Energy remains the unresolved part of genuine common-period reporting. The previous [source/input audit](energy_work_seasonal_consistency_audit.md) documents the daily-average weekday response above 20°C. Multiplying the current value by 31/30 or by three changes an assumed duration, not the exposure or usage model. Do not present that as a validated July/summer correction.

## Reviewer debate and consensus

These are three simulated expert roles, not external consultations.

| Position | Strongest argument | Limitation and resolution |
|---|---|---|
| Energy researcher: July-first | Matches the original July land-surface/ET0 setup; narrower illustration reduces extrapolation of that cooling field across months | July is a useful case-study scope, not evidence of actual July energy savings; conditional energy still requires explicit qualification |
| Capacity researcher: JJA-first | Uses all available reconstructed summer days and evaluates the nonlinear response before averaging | JJA daily-Tmax capacity aligns averaging dates with health, not occupational working-hour exposure |
| Senior PI synthesis | Choose the reporting period from the scientific question, not whichever produces the largest or most convenient values | Prefer JJA for seasonal claims; July is the closest single-month compromise for tested benefits, but retains an upward difference and member variability |

**Consensus:** keep the completed JJA health framework. Offer July as sensitivity or a deliberately scoped illustration. If common-calendar presentation is the priority, JJA-averaged daily-Tmax capacity is a feasible alternative main estimand, with the fixed-condition result moved to sensitivity **only after author approval**. This is a choice of estimand, not an automatic reversal of the previous fixed-condition recommendation. Neither version measures average working-day capacity.

## Implemented manuscript candidate

The all-nine candidate now uses JJA health and JJA daily-Tmax capacity at RH45, while retaining conditional 30-day energy. The fixed 25/5°C capacity calculation and RH35/RH55 cases are separate sensitivities. See the [candidate results and manuscript text](energy_capacity_candidate_results.md). Different outcome units remain acceptable when explicit, but the panels must not be described as three equivalent summer totals.

If a July case study is later preferred, July health and capacity can be selected directly from this diagnostic. It should be described as a July illustration, not total summer benefit, and energy would still require the qualification above.

There is no need to discard the existing summer calculation or rerun the cooling model merely to choose between these reporting narratives.

## Reproducibility and checks

- [Monthly health script](../../code/health_assessment/check_monthly_health_alignment.py): all five central-response outcomes, exact existing common health mask (14,258,588 cells), same mortality/response inputs, 0–1-day lag retained across month boundaries. June/July/August sum to the existing JJA estimates; maximum replay difference is below 0.00001 deaths. Daily direct-arithmetic spot checks pass. London monthly cause-specific mortality is still represented using the existing England monthly fractions, not newly observed borough data.
- [Monthly capacity script](../../code/post_processing_layers/run_monthly_capacity_alignment.py): same 15,733,893-cell capacity mask, shared offsets, RH45 and equal 15-member weights. JJA equals the 30/31/31-day weighted monthly mean and reproduces the previous daily-capacity candidate within 5 × 10⁻¹⁵ pp. Lookup error bound is 3.57 × 10⁻⁷ pp.
- [Plot script](../../code/post_processing_layers/plot_monthly_outcome_alignment.py): canonical scenario colours, separate axes/units, daily-normalized health, explicit member ranges and versioned PNG. This is a diagnostic figure, not a replacement for manuscript Figures 4–5.
- [Diagnostic folder](</Users/yingjiel/Library/CloudStorage/GoogleDrive-yingjiel@stanford.edu/Shared drives/Wellcome Trust Project Data/2_postprocess_intermediate/UCM_official_runs/diagnostics/monthly_outcome_alignment_20261113_v01>): health and capacity member/summary CSVs, manifests, source hashes, joined comparison and PNG. Existing outputs remain unchanged; no push.

Run the two analysis scripts with the configured data root and fresh output directories, then run the plot script on their parent diagnostic folder. Read each script's `--help` for arguments. Source hashes are checked before/after; no month-specific recentering is introduced. Numerical reproducibility is verified; empirical seasonal cooling, work-hour exposure and energy validation remain outside this test.
