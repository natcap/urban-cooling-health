# Main and Supplementary Methods: targeted greening scenarios

Reconciled on 16 November 2026 against the final Target-generation manifest, generator code and [final-raster Table 1 audit](table1_scenario_canopy.md). [HTML copy-ready version](targeted_scenarios_methods.html). Word files and model outputs remain unchanged.

## Proposed main-text description

Place after “(6) Transformative greening (Green30)” and before “Modeling co-benefits”. Construction details remain in Supplementary Methods; realized canopy amounts remain in Table 1.

**(7–9) Vulnerability-targeted greening (Target10, Target20 and Target30).** We prioritized screened street-tree planting locations using the product of normalized afternoon Universal Thermal Climate Index (UTCI) and neighbourhood SVI percentile. Eligible non-canopy cells were converted in priority order until added-canopy area matched Green10, Green20 or Green30, respectively, across the full raster footprint. Scenario names denote nominal targets; realized canopy increases and small London-boundary matching differences are reported in Table 1. These scenarios compare the distribution of benefits under proximity-based and vulnerability-targeted allocation at approximately equal canopy amounts. Candidate screening and rasterization procedures are detailed in Supplementary Methods.

## Proposed Supplementary Methods opening

We developed three vulnerability-targeted greening scenarios (Target10, Target20 and Target30) that prioritize locations with jointly elevated heat exposure and social vulnerability. The scenario labels denote nominal canopy-increase targets; realized additions were matched to the corresponding Green scenarios across the full raster footprint, rather than set by a prescribed number of trees. Realized relative increases and final London canopy-cover shares are reported in Table 1.

## Proposed Supplementary Methods step 4

**4) Generation of targeted greening scenarios.** The screened dataset contained 3,378,103 candidate planting points. We processed points in priority order, resolving tied ranks by ascending feature identifier (FID). A candidate contributed a new canopy cell only if the centre of its containing 10 m × 10 m cell lay strictly within 5 m of the point and the baseline cell represented improved grassland, urban or suburban land (codes 4, 20 and 21). Cells already accepted were not counted again, and existing canopy (codes 1, 2 and 100) was retained. Accepted cells were converted to canopy code 100 until the realized added-canopy budget of the corresponding Green scenario was reached. This produced nested Target10, Target20 and Target30 scenarios with 307,768, 595,084 and 894,249 unique new canopy cells, equivalent to 30.7768, 59.5084 and 89.4249 km² across the full raster footprint. These are modelled canopy-cell counts, not estimates of the number of physical trees required. Within the London borough boundary, Target canopy increases were 9.61%, 18.57% and 27.91%, respectively; added areas differed from their Green counterparts by less than 0.015% because of boundary exclusions (Table 1).

## Reconciliation of counts

| Quantity | Target10 | Target20 | Target30 | Interpretation |
|---|---:|---:|---:|---|
| Historical trial feature counts quoted in the Supplement | 565,000 | 1,129,000 | 1,819,000 | Superseded trial selections, not counts used to stop final generation |
| Ranked candidate prefix through final cutoff | 636,851 | 1,252,107 | 1,890,953 | Zero-based sorted-record cutoff plus one; includes rejected and duplicate-cell candidates, not accepted sites |
| Final accepted unique canopy cells | 307,768 | 595,084 | 894,249 | One retained contributing candidate per new cell in this implementation; the reporting unit is canopy cells |

**Why the figures differ.** Final generation skips candidates without an eligible new cell and counts duplicate cells only once. It stops at the matched raster-area budget, not at the historical trial point count. Processing the first 636,851 ranked records, for example, yields 307,768 accepted cells for Target10. Prefix sizes are determined by the cutoff's position in the sorted list, not its `rank` value or original source-row number; ties make these different. Batch processing can read beyond a scenario cutoff, so these are prefixes through the last accepted candidate, not total computational records read.

The old figures came from `tree_equity_scenario710v2.gpkg`, `tree_equity_scenario730v2.gpkg` and `tree_equity_scenario730v3.gpkg`, as recorded in the [historical candidate audit](../../code/lc_scenarios/TARGET_OPPORTUNITY_METHOD_AUDIT.md). Keep them as historical provenance only. The final manifest does not justify replacing them with new physical-tree totals. A 5 m candidate radius is a raster acceptance rule: the counted area is 100 m² per accepted grid cell, not an exact circular crown area. Any area-to-tree-equivalent calculation would require a separately stated crown-size assumption and should not be called a site count.

## Scope and consistency notes

- Replace the Supplement's opening claim of exact 10%, 20% and 30% realized increases with the proposed opening above.
- Replace step 4 in full, including the duplicated “using average crown-size assumptions” wording and the historical tree-site counts. Steps 1–3 are not rewritten here; their upstream point screening and UTCI–SVI scoring are collaborator-described methods, not independently recreated by this reconciliation.
- Do not use the source-code transition counts (e.g., 21,412 grassland cells in Target10) as planting-site counts.
- Exact budget matching refers to the full raster footprint. Use the linked Table 1 audit for London-boundary areas, final cover shares and the common denominator; do not duplicate that table here.
- The nominal labels can remain for continuity, but Green/Target descriptions and figure legends must not imply exact realized 10/20/30% increases or exact matching after London-boundary clipping.

## Evidence and reproducibility

Final construction source, relative to `URBAN_COOLING_DATA_ROOT`:

```text
1_preprocess/UrbanCoolingModel/OfficialWorkingInputs/LULC/lc_tree_equity_scenarios_output/revised_v2_rank_fid_equal_area_2026-09-10/target_scenario_manifest_revised.json
```

The manifest records 3,378,103 candidates, `(rank, FID)` ordering, 5 m radius, eligible and canopy codes, per-scenario budgets, cutoffs, raster hashes and nested-scenario QA. The candidate-prefix totals above are `sorted_record_index_zero_based + 1` for each scenario. Generator: [generate_target_scenarios.py](../../code/lc_scenarios/generate_target_scenarios.py), especially candidate filtering, unique-cell acceptance and cutoff recording. Final-raster verification: [audit_table1_canopy.R](../../code/lc_scenarios/audit_table1_canopy.R); its measured full-raster additions agree exactly with the construction manifest. No scenario regeneration was needed for this correction.
