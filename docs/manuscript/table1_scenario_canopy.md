# Table 1: greening scenarios and realized canopy cover

Verified directly from the final corrected-run land-cover inputs on 16 November 2026. This is the canonical source for the six Green/Target rows; [HTML](table1_scenario_canopy.html) provides a copy-ready version. Other Table 1 scenarios are outside this update. No Word files, rasters or model outputs were changed.

## Proposed Table 1 rows

Values use valid 10 m pixel centres within the London borough boundary. Baseline canopy occupies **320.3361 km²**, or **20.36%** of the **1,573.3893 km²** valid mapped area.

| Scenario | Code | Description | Realized added canopy (km²) | Realized relative increase (%) | Final canopy-cover share (%) |
|---|---|---|---:|---:|---:|
| Policy-inspired greening (nominal +10%) | Green10 | Proximity-based expansion inspired by the London Environment Strategy's 10% relative canopy-increase target for 2050. Eligible land near existing canopy was converted using the InVEST Scenario Generator. | 30.7746 | 9.61 | 22.32 |
| Ambitious greening (nominal +20%) | Green20 | Higher-intensity expansion using the same proximity-based allocation approach as Green10. | 59.5040 | 18.58 | 24.14 |
| Transformative greening (nominal +30%) | Green30 | Further canopy expansion using the same proximity-based allocation approach as Green10. | 89.4176 | 27.91 | 26.04 |
| Targeted greening (nominal +10%) | Target10 | Screened street-tree planting locations prioritized by heat exposure and social vulnerability, with the added-canopy budget matched to Green10 across the full raster footprint. | 30.7702 | 9.61 | 22.32 |
| Targeted greening (nominal +20%) | Target20 | Expansion to additional ranked planting locations, with the full-raster added-canopy budget matched to Green20. | 59.4965 | 18.57 | 24.14 |
| Targeted greening (nominal +30%) | Target30 | Further expansion across ranked planting locations, with the full-raster added-canopy budget matched to Green30. | 89.4058 | 27.91 | 26.04 |

**Table note.** Scenario names retain nominal targets; realized changes were measured from the final rasters. Canopy comprises broadleaved woodland (code 1), coniferous woodland (2) and the project canopy class (100). Relative increase is net added canopy divided by baseline canopy; final share is canopy area divided by the fixed valid mapped London area. Added canopy is conversion of eligible non-canopy classes 4, 20 or 21 to 100; no canopy removals occurred. Same-tier Green and Target budgets match exactly over the full raster footprint. Within the London boundary, Target additions are 0.013–0.014% smaller because of edge exclusions. Equal canopy area does not imply equal planting costs or implementation feasibility.

Policy reference: [London City Hall's clarification of the 10% relative target](https://www.london.gov.uk/who-we-are/what-london-assembly-does/questions-mayor/find-an-answer/tree-coverage-0). The policy's historical baseline is not substituted for this project's measured raster baseline.

## What changes from the earlier draft?

- Replace unqualified “+10%, +20%, +30%” with **nominal** targets and report measured increases separately. Do not describe Green20/30 as exactly doubling/tripling the realized Green10 increase.
- Replace the draft final shares **22%, 23%, 25%** with **22.32%, 24.14%, 26.04%** for the London-boundary definition. The Target shares round to the same values, but their exact areas differ slightly.
- Avoid “maximum plausible co-benefits”: these scenarios explore higher canopy amounts, not an established feasibility ceiling.
- Main-figure legends currently describing Green30 as an exact 30% increase should say **“a nominal 30% canopy-increase scenario (27.91% realized within London)”**, or simply **“the higher-intensity Green30 scenario (Table 1)”**. This audit changes reporting, not model outputs.

The nominal-versus-realized distinction is consistent with the [scenario construction record](../../code/lc_scenarios/README.md#observed-green-transitions): historical conversion totals included relabelling existing woodland, whereas revised budgets retained eligible new-canopy additions. The audit confirms current values; it does not retrospectively reinterpret every historical naming decision.

## Denominator sensitivity and pair matching

The full valid raster footprint covers **1,594.6858 km²**, with baseline canopy **320.4866 km² (20.0972%)**. It extends beyond the borough boundary. These values describe scenario construction and should not be mixed with London-boundary percentages.

| Pair | Full-raster added canopy, each scenario (km²) | Full-raster relative increase, each (%) | Full-raster final share, each (%) | London-boundary Target minus Green addition (km²) |
|---|---:|---:|---:|---:|
| Green10 / Target10 | 30.7768 | 9.6031 | 22.0271 | −0.0044 |
| Green20 / Target20 | 59.5084 | 18.5681 | 23.8288 | −0.0075 |
| Green30 / Target30 | 89.4249 | 27.9029 | 25.7048 | −0.0118 |

The boundary differences are 44, 75 and 118 cells, respectively. They do not constitute exact London-only matching, even though they are small. No rerun is proposed solely for these differences; retain the transparent qualification. The area denominator includes all valid mapped classes, including water, and does not use the smaller population/temperature intersection employed in some downstream health calculations. Pixel-centre boundary assignment approximates vector area; it is not fractional-cell intersection weighting.

## Verification and reproduction

**Source trace:** corrected temperature-run `arguments.json` files identify the actual LULC inputs for baseline, Green30 and Target30 in the two NoData diagnostic folders, and Green10/20 and Target10/20 in the remaining-corrected-temperatures folder. Exact paths, argument files, boundary components and MD5 hashes are recorded in [provenance](data/table1_canopy_provenance.json).

**Checks passed:** all seven rasters have identical grids and NoData footprints; EPSG:27700, 10 m cells (100 m²); zero canopy removals; zero ineligible additions; fixed denominator within each audit domain; exact Green–Target matching over the full footprint; all source hashes unchanged after counting. The complete [counts](data/table1_canopy_audit.csv) and [pair differences](data/table1_canopy_budget_checks.csv) retain unrounded values. Confidence: suitable for manuscript reporting with the stated nominal-target and boundary qualifications.

Generator: [audit_table1_canopy.R](../../code/lc_scenarios/audit_table1_canopy.R). Requires R, terra and jsonlite. It reads rasters in blocks and does not run InVEST or health models.

```sh
Rscript code/lc_scenarios/audit_table1_canopy.R "$URBAN_COOLING_DATA_ROOT" docs/manuscript/data
pandoc docs/manuscript/table1_scenario_canopy.md --standalone --css report-light.css --metadata title="Table 1: scenarios and realized canopy" -o docs/manuscript/table1_scenario_canopy.html
```
