# Energy scaling and building-allocation correction test

12 November 2026. Isolated coefficient test; production inputs, figures and model outputs are unchanged. No push.

## Recommendation

Use a **London-scaled, pooled building-allocation proxy** as the feasible candidate, with sector mapping as sensitivity—not as a validated London energy model. The existing broad land-use classes do not identify offices, schools, hospitals and hospitality reliably enough to assign precise national sector shares. Pooling removes that false precision, but assumes equal demand sensitivity per footprint area within the pool. It does not resolve building height, occupancy or cooling-system uptake.

Both supplied compact building GeoPackages (`bld_with_attr_compact_ucm.gpkg` and `bld_with_attr_compact_ucm2.gpkg`) were checked: each has 2,223,481 features and only `fid_attr` and `type` attributes. Thus the detailed sector crosswalk cannot be recovered from these inputs alone. The pooled candidate is a deliberate change of allocation method, not a claim that the missing building uses have been identified.

Keep the monetary result conditional on a 30-day warm-period scenario for now. Correcting scaling and allocation does not turn the source weekday, daily-average-temperature association into a seasonal Tmax-based energy estimate. Final adoption still requires the corrected temperature contrasts and an explicit exposure-transfer assumption.

## What was wrong, what changed, and why

The problem is primarily the **interpretation and allocation of the energy inputs**, not an error in InVEST's multiplication or the subtraction of baseline from scenario. It is useful to separate identifiable inconsistencies from assumptions that require evidence. The [preceding audit](energy_work_seasonal_consistency_audit.md#1-energy-an-input-calibration-issue-not-just-a-time-unit-issue) records the workbook cells and source files; the examples here explain their consequences.

### 1. A national response was distributed over London alone

**Original calculation:** the workbook converted 350 MW/°C into 252 million kWh/°C for an assumed 30-day month, split this into sectors, and divided by London building areas. The 350 MW/°C refers to the response of **Britain's electricity system**, not London's. Using London area as the denominator does not itself make the numerator London-specific.

**Intuitive example:** suppose a national electricity system needs an additional 100 units of electricity when it gets warmer, and London uses about 13 units out of every 100 nationally. Assigning all 100 additional units to London treats the rest of Britain as if it had no share of the national response. A first approximation is to assign London 13 additional units—not all 100.

**Change tested:** multiply the national response by London's same-year share of GB electricity consumption, 13.361%. The response becomes 46.764 rather than 350 MW/°C. This alone changes Green30 from £16.294m to £2.177m under otherwise identical inputs.

**Why this is better justified:** numerator and denominator now refer to the intended geography under an explicit allocation rule. However, the rule assumes that temperature-sensitive demand is distributed like total electricity consumption. London's building mix and air-conditioning uptake may differ from Britain's. Therefore, this fixes the unscaled national allocation but does **not prove that 46.764 MW/°C is London's true response**. The unscaled value is about 7.48 times this proxy; that ratio is not an independently measured overestimate of real savings.

### 2. National sector labels did not match the building categories

**Original allocation:** the workbook's office share was applied to Community Services and Recreation/Leisure, while Industry And Business had a zero coefficient. The building inputs do not identify which individual buildings are offices, schools or hospitals. A sector percentage cannot be transferred merely because another category also contains non-residential buildings.

**Intuitive example:** knowing the national share of cooling electricity used by offices does not tell us how much should be assigned to a London leisure centre. A leisure centre and an office can have different equipment, operating hours and cooling demand. Giving one the other's coefficient can change both the total and where the benefits appear.

**Change tested:** the pooled candidate distributes a fixed non-residential response by footprint area across the specified eligible classes. It no longer claims to distinguish office cooling from hospital or leisure cooling. The alternative sector mapping is retained only as sensitivity, with every assumed correspondence listed below.

**Why this is preferable with the available data:** it avoids unsupported fine-sector assignments and makes the allocation reproducible. It does **not** establish that all non-residential buildings have the same true demand per square metre. Pooling trades uncertain sector detail for a simpler, visible assumption; it can still misallocate demand where building height, use or equipment differ. Thus it is the preferred feasible proxy, not an empirically proven superior predictor.

### 3. One denominator included buildings that received no allocation

The original commercial coefficient divided its assigned energy quantity by approximately **2.364 million m² of Mixed Use plus Utilities**. But the active table gave Mixed Use a zero coefficient and applied the calculated coefficient only to approximately **0.184 million m² of Utilities**.

**Simple example:** divide a 100-unit allocation by the combined area of ten equal-sized buildings. The coefficient assigns ten units to each. If it is then applied to only one building, nine-tenths of the intended allocation disappears. In the actual workbook areas, Utilities represent about 7.78% of that combined denominator: only that proportion of this particular commercial allocation would be assigned under uniform 1°C cooling. This is an accounting diagnostic, not a claim that every scenario loses 92.22% of its total energy benefit.

**Change tested:** calculate each pool's denominator using exactly the same eligible London buildings that receive its coefficient. Mixed Use is included explicitly in the pooled candidate rather than accidentally receiving zero. Because its residential fraction is unknown, treating the whole Mixed Use footprint as non-residential remains an assumption.

**Why this correction is mathematically sound:** for uniform 1°C cooling, summing `coefficient × building area` over the pool recovers exactly its assigned energy sensitivity. This conservation check fixes the mismatch between the allocation denominator and recipient buildings. It verifies accounting, not the empirical validity of the total sensitivity.

### 4. Worked example for a single building

Consider a **hypothetical** 1,000 m² Community Services building, cooled by 0.3°C, with the unchanged tariff of £0.28/kWh. Benefit is `coefficient × 1,000 × 0.3 × 0.28`. These inputs illustrate the arithmetic; this is not an observed building.

| Calculation | Coefficient (kWh m⁻² °C⁻¹ per assumed 30 days) | Conditional saving |
|---|---:|---:|
| Original | 9.8907 | £830.82 |
| London scaling only | 1.3215 | £111.01 |
| London scaling + pooled allocation | 1.4880 | £124.99 |

Scaling reduces the national quantity assigned to London. Pooling then redistributes that smaller quantity among building classes, so some coefficients increase relative to the scaling-only version and others decrease. **The pooled correction is not simply the old result multiplied by one factor.** That is why allocation can change relative scenario benefits when scenarios cool different kinds of neighbourhoods.

### 5. What this test does not correct

The 30-day conversion remains a conditional extrapolation of a weekday demand relationship, not an observed monthly or summer saving. The source temperature metric is daily-average temperature; the modelled cooling contrast cannot automatically be assumed to transfer in full to that metric. The 1% residential share, footprint-based allocation, fixed tariffs and transfer to a future climate remain assumptions. The current comparison also holds the previous temperature inputs fixed rather than incorporating the corrected temperature family.

It would therefore be too strong to say **“the recommended method is now correct”** without qualification. The defensible statement is: **the test removes the unscaled national allocation, makes category allocation explicit, and fixes the denominator/recipient mismatch; the resulting energy estimate remains a transparent proxy requiring exposure consistency checks before manuscript adoption.**

### What was actually changed in the project

- Added a separate diagnostic calculation and results, rather than replacing the production consumption table.
- Tested scaling alone, the pooled candidate, a sector-mapping sensitivity and removal of the assumed residential share against the original replay.
- Kept temperatures, cooling-model settings, building geometry and tariffs fixed so coefficient effects can be identified separately.
- Verified reproduction of old totals, conservation of the candidate allocation and unchanged input hashes. Agreement among sensitivity results supports the tested **matched Green–Target direction**, not validation of absolute costs.
- Updated this explanation and links in the related documentation. Existing energy figures and production outputs have **not** been changed.

## Verified scaling

The [DESNZ regional electricity dataset](https://www.gov.uk/government/statistics/regional-and-local-authority-electricity-consumption-statistics), December 2025 release, gives 2021 consumption of 34,555.363 GWh for London and 258,627.498 GWh for Great Britain including unallocated consumption. Their ratio is **0.13361055**. These are same-release, same-year totals; do not substitute an older GB total from a differently revised release.

Multiplying the original [350 MW/°C Britain-wide response](https://reports.electricinsights.co.uk/?noamp=mobile&p=462) by that ratio gives **46.764 MW/°C** for London. This is a demand-share transfer assumption, not an observed London slope. Converting with the original 30 × 24 hours retains a controlled comparison: 33.670 million kWh/°C per assumed month. It assumes the marginal response is sustained through that period; it is not an estimate of observed monthly consumption.

Using the mean London share from the source-response years 2016–2018 instead gives 0.13877019 and 48.570 MW/°C, increasing every scaled estimate by approximately 3.86%. This is a useful temporal-scaling sensitivity. Total electricity share is preferred to non-domestic share for scaling a total-system slope; neither directly measures London's share of cooling sensitivity.

## Completed comparison

**£ million per conditional 30-day warm period**, relative to baseline. Negative values mean higher electricity costs. These are deterministic coefficient sensitivities on unchanged legacy temperature contrasts, not new seasonal estimates or confidence intervals.

| Scenario | Existing replay | London scaling only | Pooled candidate | Sector sensitivity |
|---|---:|---:|---:|---:|
| AllBuilt | −49.584 | −6.625 | −8.705 | −8.172 |
| TreeRisk | −4.635 | −0.619 | −0.835 | −0.837 |
| TreeOpp | 2.795 | 0.374 | 0.504 | 0.560 |
| Green10 | 5.772 | 0.771 | 1.045 | 1.016 |
| Green20 | 10.938 | 1.461 | 1.970 | 1.910 |
| Green30 | 16.294 | 2.177 | **2.924** | 2.824 |
| Target10 | 8.762 | 1.171 | 1.427 | 1.526 |
| Target20 | 15.590 | 2.083 | 2.623 | 2.827 |
| Target30 | 22.374 | 2.989 | **3.908** | 4.114 |

Baseline changes are zero in every variant. Main findings:

- Scaling alone reduces all values by 86.64%, without altering their ranking. Allocation partly offsets that reduction: the pooled Green30 and Target30 values are **82.06% and 82.53% below** their legacy counterparts.
- **Target exceeds Green at each matched 10/20/30 level in all five tested variants.** Target30 exceeds Green30 by 33.66% in the pooled candidate, versus 37.31% previously and 45.67% in the sector sensitivity. The advantage's magnitude depends on allocation.
- The complete nine-scenario ordering is not invariant: sector sensitivity places Target20 (£2.827m) just above Green30 (£2.824m). Do not claim every ranking is robust.
- Removing the assumed 1% residential pool and reallocating it to non-residential buildings gives £2.924m for Green30 and £3.912m for Target30. This particular allocation assumption has little effect on these citywide comparisons; it does not validate residential energy estimates or borough patterns.
- The model replay matches all saved citywide changes within £0.01; maximum difference between recalculated and saved building-level changes is below £0.000001. All inspected input hashes are unchanged. **2,223,411 unique buildings** have positive London overlap; 70 outside buildings are excluded.

Full coefficients, class areas, exact results and checks: [diagnostic comparison](../../data/diagnostics/energy_scaling_20261112_v01/comparison.json). The official [regional input CSV](../../data/diagnostics/energy_scaling_20261112_v01/elec_region_stacked_2005-2024.csv) is archived beside it. These files are local diagnostic artifacts and are not automatically included in Git.

## Allocation versions tested

All versions keep exactly the same building-temperature differences and class-specific tariffs. This isolates coefficient changes from the separate NoData/temperature correction.

| Version | Allocation | Purpose |
|---|---|---|
| Legacy replay | Original consumption table | Reproduce saved unique-building London totals |
| Scale only | Every original coefficient × 0.13361055 | Isolate national-to-London scaling; preserve known mapping problems |
| Pooled candidate | 99% of scaled sensitivity across known occupied non-residential classes by footprint area; 1% residential | Avoid unsupported fine-sector mapping |
| Sector sensitivity | Explicit broad-class mapping below, normalized to the same total | Test whether allocation changes conclusions |
| Pooled, no residential | Entire sensitivity assigned to the same non-residential pool | Test removal of the unsupported legacy 1% residential share |

The **1% residential share is an assumption retained for continuity**, not a measured share of cooling electricity. The government review does not establish that share. The no-residential variant reallocates the total; it is not a lower confidence bound.

### Complete class treatment

| Code | Building class | Pooled candidate | Sector sensitivity |
|---|---|---|---|
| −1 | Missing/NA | Excluded, not assumed physically zero | Excluded |
| 0 | Agriculture | Excluded | Excluded |
| 1 | Community Services | Non-residential pool | Shared civic/leisure/hospitality pool with 6 |
| 2 | Defence | Non-residential pool | Military share |
| 3 | Industry And Business | Non-residential pool | Office/storage proxy with 5 |
| 4 | Minerals | Excluded | Excluded |
| 5 | Mixed Use | Non-residential pool (uncertain residential fraction) | Office/storage proxy with 3 |
| 6 | Recreation And Leisure | Non-residential pool | Shared civic/leisure/hospitality pool with 1 |
| 7 | Residential | Assumed 1% pool | Assumed 1% pool |
| 8 | Retail | Non-residential pool | Retail share |
| 9 | Transport | Non-residential pool | Unmapped/excluded |
| 10 | Unclassified presumed non-residential | Excluded pending classification | Excluded |
| 11 | Utilities And Infrastructure | Non-residential pool | Unmapped/excluded |
| 12 | Vacant | Excluded | Excluded |

Exclusions define this proxy's allocation domain, not evidence of zero consumption. No unknown class is silently assigned an observed cooling coefficient. The sector variant redistributes a fixed total over its narrower mapped pool; it does not estimate missing sectors independently.

For sensitivity only, the [government cooling report, Table 2](https://assets.publishing.service.gov.uk/government/uploads/system/uploads/attachment_data/file/1019896/cooling-in-uk.pdf) provides non-domestic cooling/humidification shares. The assumed mapping combines civic/leisure/education/emergency/health/hospitality (25.5%) into codes 1/6; military (1.6%) into 2; offices/storage (54.2%) into 3/5; retail (18.8%) into 8. Rounded shares sum to 100.1%, so they are normalized before assigning 99% of the total. These are shares of energy consumption, **not measured shares of marginal temperature sensitivity**. Mapping offices to Industry And Business is an explicit sensitivity proxy, not a verified office inventory.

## Reproducibility and safeguards

Script: [test_energy_scaling.py](../../code/post_processing_layers/test_energy_scaling.py). It reads existing baseline plus all nine scenario building layers and the active consumption table; reuses the validated London allocation cache; matches building identifiers and class codes; independently reproduces stored baseline-relative energy totals; and checks input hashes before/after. It refuses to overwrite an existing diagnostic JSON.

The coefficient denominator is the exact summed footprint area of eligible unique buildings with positive London overlap, matching the numerator's domain. Whole boundary-crossing building values remain included, consistent with existing accounting. The comparison is citywide; borough estimates have not been regenerated.

For pool g, the coefficient is `252,000,000 × London_share × pool_share / summed_building_area_g` (kWh m⁻² °C⁻¹ per assumed 30-day period). A building's benefit is that coefficient × footprint area × its original tariff × `(baseline_temperature − scenario_temperature)`. Warming produces negative savings. Pool coefficients conserve the allocated London response exactly; the replay checks are independent comparisons with saved building values and city totals.

Reproduce with the project's InVEST environment:

```sh
python code/post_processing_layers/test_energy_scaling.py \
  --data-root "$URBAN_COOLING_DATA_ROOT" \
  --regional-csv data/diagnostics/energy_scaling_20261112_v01/elec_region_stacked_2005-2024.csv \
  --output /tmp/energy_scaling_new_comparison.json
```

Use a new output filename on each run. The archived source is the official CSV, not a manually transcribed table. The diagnostic JSON retains selected source rows, coefficients, category areas, input hashes and scenario results for independent checking.

No InVEST rerun is necessary for this controlled linear coefficient test. Its diagnostic coefficients are not installed as a production input table. Existing temperatures deliberately remain fixed, so these are **not yet corrected-temperature energy results**.

## Remaining adoption steps

1. Review the pooled-allocation assumption and its mapping sensitivity. Obtain a finer use/floor-area inventory only if precise sector-specific claims are essential.
2. Apply the chosen candidate to corrected baseline and scenario building temperatures in a new output family, checking building coverage.
3. Retain explicit fixed tariffs and conditional reporting; do not annualize or multiply by three. A seasonal energy calculation requires daily-mean exposure, threshold/usage assumptions and an explicit transfer of modelled cooling to that metric.

A bounded literature check found a London study using 1997–2001 residential/commercial load, but no directly usable updated London cooling slope was verified from the accessible preview ([Psiloglou et al., 2009](https://doi.org/10.1016/j.energy.2009.07.033)). A newer London bottom-up cooling paper was identified, but full text was inaccessible; no coefficient was extracted from it. Thus the practical candidate is a transparent proxy, not a claim that local empirical calibration has been completed.
