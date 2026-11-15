# Nine-scenario coverage, offsets and signed health implementation

12 November 2026. **Checks completed; existing outputs unchanged. No full health rerun, manuscript edits or push.**

## Decision summary

**Subsequent update:** the [seven corrected temperature runs](corrected_temperature_family.md) are now complete and verified. The input gate described below is closed; this page retains the original implementation-test record. Consult the linked report for current corrected paths and readiness status.

Coverage and signed-response implementation pass. Shared daily offsets have been recomputed using buffered climate and the corrected baseline on the full tested footprint. **All-nine-scenario adoption is still gated by corrected temperature inputs for seven scenarios.** Do not mix their existing production temperatures with the corrected baseline.

This follows the [coverage and mortality audit](climate_coverage_consistency_audit.md); its scientific assumptions remain unchanged. This is implementation validation, not a new health-results comparison.

## Coverage and reconstruction

- Checked baseline plus **all nine scenarios**: AllBuilt, TreeRisk, TreeOpp, Green10/20/30 and Target10/20/30, using the existing 25°C/5°C production family for input inspection.
- All temperature grids, the corrected baseline, population and five mortality rasters share an identical compatible grid. Their common London footprint is **14,258,588 cells and 8,834,005 residents**; none of the additional scenario rasters reduces the previously recovered footprint.
- All 24 buffered climate combinations (three GCMs × two SSPs × May–August) are valid throughout that footprint. No residual climate gap was found.
- Corrected baseline area mean on this footprint is **28.527035°C**. New offsets use this mean, not the earlier restricted-footprint mean.
- Saved **5,520 unique daily offsets**: two reconstructions × two SSPs × three GCMs × five weather patterns × 92 June–August days. Each uses previous-day history for lag 0–1; the WorldClim-aligned monthly baseline mean reproduces its climate anchor to numerical precision.

**Why one offset file, not nine copies?** The daily climate offset is shared by every land-use scenario. A pixel's reconstructed temperature is its scenario UCM temperature plus that shared offset (or corrected baseline minus its scaled cooling contrast plus the offset). Giving each scenario its own centring mean would remove the very citywide cooling/warming difference being evaluated. `scenario_input_registry.csv` records all nine scenarios plus baseline; `shared_daily_offsets.csv` supplies their common time series. Offsets are not threshold-subtracted; the health caller subtracts 24.7°C exactly once.

These offsets are candidates for the corrected-baseline family. They are **not authorization to combine corrected baseline and uncorrected scenario temperatures**. Recheck their footprint and hashes when the remaining corrected scenario inputs exist; regenerate offsets if the shared footprint changes.

## Signed health calculation

New reusable helper: [`signed_health_response.py`](../../code/health_assessment/signed_health_response.py). Existing cooling-only scripts remain unchanged.

```text
avoided = D × [1 − exp(log(RR) × (max(Tscenario − threshold, 0)
                               − max(Tbaseline − threshold, 0)))]
```

The helper computes exact summed pixel effects using sorted cumulative sums, splitting cooling and warming pixels. It retains negative values for warming-related harm; it does not clamp results to zero. It also supports mixed cooling/warming within one scenario. The caller still applies cause-specific monthly mortality fractions and day counts; the helper does not silently annualise or seasonalise results.

Validation completed:

| Test | Result |
|---|---|
| Actual raster samples | 1,500 reproducibly sampled pixels, ten configurations including baseline, five outcomes, three coefficient values: **150 comparisons**, each over 113 cold/hot/reconstruction offsets |
| Agreement with direct pixel arithmetic | Maximum absolute difference **6.22 × 10⁻¹⁴** in sample weighted counts |
| Warming scenarios | AllBuilt and TreeRisk produce negative effects when heat exposure increases; below-threshold effects remain zero |
| Cooling scenarios | TreeOpp and all Green/Target scenarios produce non-negative effects in these tests |
| Baseline identity | Exactly zero |
| Synthetic mixed-sign/threshold tests | Random mixed warming/cooling plus exact-threshold, crossing and unchanged cases; maximum error **1.38 × 10⁻¹¹** |
| Cooling-only regression | Agrees with the earlier integration helper within the 10⁻⁸ absolute tolerance |
| Preservation | Source hashes checked before and after; no existing raster or model output changed |

Real-data numerical tests use the **existing production baseline together with its production scenarios**, consistently, without substituting the corrected baseline into that family. Test offsets span cold, hot and reconstructed exposure ranges. Their purpose is numerical verification, not estimation of corrected summer impacts. Sample results are not London totals, seasonal estimates, confidence intervals or validation of the epidemiological transfer.

## Remaining input gate: seven corrected temperature runs

The manifests for all nine production scenarios and baseline point to the biophysical table containing the declared LULC NoData code (0). This is the same lookup collision addressed by the earlier isolated correction tests. Raster coverage can pass despite this upstream issue; complete coverage does not prove consistent temperature generation.

Corrected candidates already exist for **baseline, Green30 and Target30**. The remaining scenarios are:

**AllBuilt, TreeRisk, TreeOpp, Green10, Green20, Target10 and Target20.**

Recommended next step is to approve isolated corrected **temperature-only** runs for those seven scenarios using the established NoData-table correction, preserving their other parameters and inputs. Then verify common coverage and scenario hashes, connect the shared offsets and signed helper to a new health run configuration, and execute the approved all-scenario seasonal health calculation. No additional response reconstruction is required merely to fix this input gate.

The current outputs do not yet establish revised citywide mortality totals or unchanged Figures 6–7 equity conclusions. Both remain for the full, versioned health rerun.

## Reproduction and records

Audit/offset generator: [`prepare_nine_scenario_health_checks.py`](../../code/preprocessing_layers/prepare_nine_scenario_health_checks.py). Run with the existing InVEST 3.20.2 Python environment:

```text
python code/preprocessing_layers/prepare_nine_scenario_health_checks.py DATA_ROOT NEW_OUTPUT_FOLDER
```

Validated diagnostic folder, relative to the project data root:

```text
2_postprocess_intermediate/UCM_official_runs/diagnostics/nine_scenario_health_readiness_2026-11-12_v2/
```

Files: `shared_daily_offsets.csv`, `worldclim_monthly_means.csv`, `scenario_input_registry.csv`, `production_family_temperature_checks.csv`, `signed_sample_checks.csv` and `manifest.json`. The manifest records exact input paths/hashes and distinguishes the corrected offset baseline from the production-family numerical tests. No figures generated.

The first attempt (`..._v1`) stopped at its source-hash guard because the audit script was edited during execution. It is incomplete and superseded by the fully completed v2 run; it has no success manifest and must not be used. Original production and earlier pilot outputs were never overwritten.
