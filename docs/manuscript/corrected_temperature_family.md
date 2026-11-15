# Corrected temperature family: all nine scenarios

12 November 2026. **Seven temperature-only reruns completed and verified against the corrected baseline. Existing outputs unchanged; not pushed. No seasonal health rerun performed.**

## Outcome

The remaining temperature-input gate is closed. Corrected AllBuilt, TreeRisk, TreeOpp, Green10/20 and Target10/20 now join the existing corrected baseline, Green30 and Target30. All ten configurations share the required footprint and are compatible with the previously recomputed full-coverage climate offsets.

**Target scenarios retain greater citywide cooling than the corresponding Green scenarios at all three canopy levels**, with either area or population weighting. AllBuilt and TreeRisk retain warming. This establishes temperature consistency, not unchanged health totals or neighbourhood equity findings.

## What changed—and what did not

Each new run used InVEST **3.20.2**, private copies of its original LULC and biophysical inputs, and a new workspace. The sole lookup change was removal of the row for declared LULC NoData code **0**. All retained biophysical rows match the corrected baseline table exactly; all shared physical arguments match the corrected baseline. Reference temperature 25°C, maximum UHI 5°C, cooling parameters and scenario land cover were not changed. Energy and productivity valuations were disabled.

Original input and production-temperature hashes, and hashes of the three previously corrected temperature rasters, were unchanged after execution and verification. Neither old production outputs nor earlier pilots were overwritten. The seven new runs use sentinel 0; the earlier baseline/Green30/Target30 sentinel-invariance test was not repeated for each new scenario.

## Temperature comparison

Mean **baseline minus scenario** temperature in °C on the shared London footprint. Positive values mean cooling; negative values mean warming. The previous column compares each existing production scenario with its own production baseline. The corrected columns consistently use the corrected baseline; no corrected/uncorrected mixing is used.

| Scenario | Previous area-weighted cooling | Corrected area-weighted cooling | Corrected population-weighted cooling |
|---|---:|---:|---:|
| AllBuilt | −1.0859 | **−1.1303** | −1.0049 |
| TreeRisk | −0.0789 | **−0.0801** | −0.1014 |
| TreeOpp | 0.0605 | **0.0630** | 0.0484 |
| Green10 | 0.1070 | **0.1095** | 0.1257 |
| Green20 | 0.2042 | **0.2089** | 0.2383 |
| Green30 | 0.3044 | **0.3112** | 0.3570 |
| Target10 | 0.1221 | **0.1230** | 0.1797 |
| Target20 | 0.2308 | **0.2334** | 0.3232 |
| Target30 | 0.3410 | **0.3457** | 0.4602 |

These figures use the full tested footprint, so Green30/Target30 values differ slightly from earlier climate-intersection summaries. Do not attribute that separate footprint difference to a new model change. The comparison CSV also retains maximum pixel-level absolute temperature changes from production; these can exceed 1°C even though paired citywide cooling changes are much smaller. The latter does not justify assuming negligible neighbourhood-level effects.

## Verification

- **Grid and footprint:** baseline plus all nine corrected scenarios match in dimensions, geotransform and CRS. Their common footprint equals the corrected-baseline/population mask exactly: **14,258,588 cells; 8,834,005 residents**. Adding all five mortality rasters causes no further exclusion.
- **Offset compatibility:** corrected baseline area mean is **28.52703536382223°C**, exactly matching the mean used for the saved 5,520 shared offsets. The common footprint is unchanged. The buffered climate coverage established in the [coverage audit](climate_coverage_consistency_audit.md) therefore remains applicable; a duplicate offset recalculation is unnecessary.
- **Signs:** AllBuilt warms all valid cells; TreeRisk warms some and leaves others unchanged. TreeOpp and Green scenarios cool all valid cells; Target scenarios cool some and leave others unchanged. No unexpected opposite-sign cells were found.
- **Signed health implementation on corrected data:** 1,500 reproducibly sampled pixels across ten configurations, five outcomes and three coefficient values yield **150 tests**. Each compares the accelerated helper against direct arithmetic over 113 exposure offsets, including cold and hot extremes. Maximum error is **1.44 × 10⁻¹³** in sample weighted counts. Warming-related negative effects are retained; baseline effects are exactly zero. These are numerical sample checks, not seasonal health totals.
- **Execution warnings:** InVEST/GDAL emitted polygon/multipolygon-type warnings during borough-summary operations and a deprecated-memory-driver warning. All runs completed. Verification uses the temperature rasters and independently reconstructed common mask, not an assumption that warnings imply valid results.

The canonical scientific assumptions remain in [the reconstruction comparison](spatial_reconstructed_health_comparison.md) and [UHI/spatial sensitivity report](uhi_threshold_scale_health_sensitivity.md). This correction does not calibrate UHI, validate full Tmax transfer or remove epidemiological transfer limitations.

## Follow-up status

The user subsequently approved the versioned all-nine-scenario seasonal health rerun. It is complete for all five outcomes, with signed warming harms, reconciled citywide/borough/LSOA totals and matched equity comparisons. See the [seasonal-health results and equity report](seasonal_health_rerun_comparison.md). Existing production and manuscript figures remain unchanged; adoption of the new health estimates is a separate decision.

## Reproducibility and output location

Scripts:

- [`run_remaining_corrected_temperatures.py`](../../code/Urban_Cooling_Modeling_Runs/run_remaining_corrected_temperatures.py): seven isolated UCM runs, shared-parameter checks, preserved-source hashes and temperature comparison.
- [`verify_corrected_temperature_family.py`](../../code/health_assessment/verify_corrected_temperature_family.py): complete-family/mortality coverage, saved-offset compatibility and signed-response checks using [`signed_health_response.py`](../../code/health_assessment/signed_health_response.py).

Run both with the existing InVEST 3.20.2 Python environment. The first refuses an existing output folder; the second refuses to overwrite its verification record. No figures were generated.

New folder relative to the configured project data root:

```text
2_postprocess_intermediate/UCM_official_runs/diagnostics/remaining_corrected_temperatures_2026-11-12_v1/
```

Each new scenario has a `<scenario>_nodata0/` folder containing private inputs, `arguments.json`, `T_air_test.tif` and model intermediates. At the root, `manifest.json` records **all ten corrected temperature paths**, source/output hashes and run provenance; `temperature_comparison.csv` contains the table's unrounded values; `health_readiness_verification.json` records the final checks and shared-offset path/hash. `progress.json` and `run.log` retain execution history. Earlier corrected baseline/Green30/Target30 files are referenced, not copied or replaced.
