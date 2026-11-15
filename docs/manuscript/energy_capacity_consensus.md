# Energy and work-capacity review consensus

12 November 2026. Three **simulated expert reviews**: two independent senior-researcher roles (building energy; occupational heat and health) and a senior-PI synthesis. These were AI-assisted methodological reviews, not external consultations or journal peer review.

## Positions and resolution

| Reviewer | Main challenge | Agreed resolution |
|---|---|---|
| Energy researcher | London demand-share scaling and pooled allocation repair inconsistencies but do not establish a locally calibrated cooling-demand response. The source daily-mean exposure cannot silently inherit the model's full cooling contrast. | Test corrected-temperature energy as a conditional 30-day proxy; retain sector allocation and full/half cooling-transfer sensitivities. Report kWh and £, both as proxy estimates. |
| Capacity researcher | Daily Tmax with assumed humidity cannot estimate average capacity across working hours. Nonlinearity also prevents replacing daily evaluation with evaluation at a seasonal mean. | Keep corrected fixed-condition Hothaps capacity in the main candidate figures; use daily-Tmax-based summer capacity and RH35/45/55 as supplementary assumption sensitivities. |
| Senior PI | Completing a reproducible calculation is not equivalent to approving an absolute manuscript claim. Different outcome periods are acceptable only when explicit. | Proceed through all four stages with versioned candidate outputs, technical QA and precise labels; no automatic final manuscript adoption. |

All three roles explicitly agreed. The principal resolution was to separate **permission to execute candidate analyses** from **scientific adoption of final claims**, rather than postponing useful work or treating remaining assumptions as validated.

## Agreed execution order

1. **Energy pilot:** corrected baseline, Green30 and Target30 building temperatures; pooled allocation primary, sector allocation and full/half transfer sensitivities. Separate temperature effects from the earlier coefficient-only comparison.
2. **Capacity pilot:** compare corrected fixed-condition capacity with daily-Tmax-based capacity, using unlagged daily offsets and equal day/GCM/weather-pattern weights. Fixed humidity settings are assumptions, not observed uncertainty bounds.
3. **Reporting framework:** health as deaths averted per modelled summer; energy as conditional 30-day savings; main capacity as fixed-condition percentage-point changes. These are complementary outcomes, not three equivalent summer totals.
4. **All nine scenarios:** after technical checks, generate new candidate summaries and versioned Figures 4–5, with daily-capacity sensitivities separately. Preserve prior outputs; do not edit Word or push.

## Technical gates

- Match building identifiers and common valid coverage across scenarios; report excluded and partially covered building counts and footprint areas.
- Match energy-coefficient denominators to recipient buildings; conserve the allocated response; reconcile borough energy sums with the citywide total.
- Reproduce the legacy calculation independently. Baseline changes must be zero; retain negative savings and capacity losses for warming scenarios.
- Keep Hothaps capacity fractions within 0.1–1. Check any lookup/interpolation approximation against the direct formula at sampled extremes and random temperatures.
- Reconcile borough area-weighted capacity with the citywide result; document the London mask and its difference from health support.
- Preserve the **same climate-offset centring across scenarios**. Do not recenter individual scenarios or silently change the reference when extending to a broader capacity mask.
- Do not use an unchanged or favorable Green–Target ranking as a pass criterion. Report reversals and sensitivity explicitly.

## Adoption limits and source documents

Full/half cooling transfer, humidity settings and alternative allocation mappings are **scenario sensitivities, not confidence intervals**. Agreement across them supports robustness within those assumptions; it does not validate absolute costs or occupational outcomes. Without working-hour exposure and workforce data, do not convert the capacity proxy into worker-hours or economic productivity.

Detailed evidence, formulas, worked examples and source provenance remain in the [energy correction test](energy_scaling_correction_test.md) and [energy/work consistency audit](energy_work_seasonal_consistency_audit.md). The [seasonal-health report](seasonal_health_rerun_comparison.md) defines the retained health framework. This note records consensus and execution gates only, not new results or completed QA.
