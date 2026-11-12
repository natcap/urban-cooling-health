# Matched 2050 temperature sensitivity

**Follow-up:** the observation-anchored [UCM climate-parameter pilot](ucm_climate_parameter_pilot.md)
is the current decision document. This mean-matching test remains a sensitivity,
not the recommended method for physically defining `T_ref` and `UHI_max`.

Temperature-only diagnostic, 11 November 2026. Existing production outputs and
manuscript figures are unchanged. Health, work capacity and energy are on hold.

## Decision-relevant result

Matching Figure 1's 2041–2060 London mean lowers absolute temperatures relative
to the existing 25/5°C configuration, but does not change land-use cooling
contrasts or their ranking when UHI and all other cooling inputs remain fixed.
This is mean matching, not independent validation or a newly executed InVEST run.

| Configuration | Reference temperature (°C) | Maximum UHI (°C) | Baseline-land-use equal-borough mean (°C) | Shift from existing run (°C) |
|---|---:|---:|---:|---:|
| Existing | 25.0000 | 5 | 28.6133 | 0 |
| SSP370-matched 2050 | 22.1052 | 5 | 25.7185 | −2.8948 |
| SSP585-matched 2050 | 22.4516 | 5 | 26.0649 | −2.5484 |

The fitted reference is an effective model parameter, not an observed rural
temperature. Do not interpret its numerical proximity to 22°C as evidence that
regional climate does not warm: the starting 25/5°C configuration was already
warmer than the target London mean.

## Assumptions and checks

- Targets are the equal-weight means of 33 boroughs and three GCMs in Figure 1,
  separately for SSP370 and SSP585, for July mean daily maximum temperature in
  2041–2060. The SSPs are not pooled.
- Match the baseline land-use configuration only. Apply the same shift to all
  nine alternative landscapes; independently fitting each landscape would erase
  the effects being evaluated.
- Retain UHI = 5°C, current land cover, ET, cooling method, mixing parameters,
  grid and London mask. The suitability of 5°C as a July air-UHI parameter is
  not established by this test.
- InVEST 3.20.2 uses `T_ref + (1 − HM) × UHI_max` before normalized mixing.
  A constant reference-temperature change therefore translates the field.
  This was checked against executed 25/5 and 28/5 rasters for all ten landscapes:
  the largest deviation from a uniform +3°C shift was **0.000017524°C** over
  **15,733,956 common London cells**. Relevant land-cover and cooling arguments
  agree between each paired run.
- Calculation: `shift = Figure 1 target − existing baseline borough mean`;
  `matched T_ref = 25 + shift`; `matched temperature = existing temperature + shift`.
  No temperature rasters were overwritten or new full model runs undertaken.
- The diagnostic exports both equal-borough and area-weighted means. Only the
  first is calibrated, because it matches Figure 1's aggregation. The matched
  area-weighted baseline means are 25.6657°C and 26.0122°C, respectively.
- Every borough's scenario-minus-baseline difference is unchanged to numerical
  precision. Spatial gradients and within-landscape temperature spread are also
  unchanged by the translation; the Figure 1 spatial field is not reproduced.

## Land-use effects under all three configurations

Signed changes relative to baseline land use; negative values mean cooling.

| Landscape | Equal-borough mean change (°C) | London area-weighted change (°C) |
|---|---:|---:|
| AllBuilt | +1.0521 | +1.1088 |
| TreeRisk | +0.0970 | +0.0764 |
| TreeOpp | −0.0588 | −0.0624 |
| Green10 | −0.1202 | −0.1021 |
| Target10 | −0.1410 | −0.1134 |
| Green20 | −0.2264 | −0.1954 |
| Target20 | −0.2583 | −0.2153 |
| Green30 | −0.3364 | −0.2910 |
| Target30 | −0.3780 | −0.3192 |

These unchanged contrasts are conditional on fixed UHI and cooling inputs, not
proof that cooling benefits are insensitive to all climate assumptions. No
conclusion about unchanged health or work-capacity benefits follows: those
components can respond nonlinearly to absolute temperatures and were not tested.

## Interpretation and next decision

The existing 25/5°C setting is a warmer thermal scenario than Figure 1's 2050
ensemble means. Mean matching offers a transparent sensitivity alternative,
but neither validates the UHI magnitude nor solves the rural-reference,
temperature-variable and spatial-resolution questions. Decide whether to retain
25/5°C as a warm scenario or adopt a climate-matched configuration only after
reviewing those assumptions. Downstream tests remain explicitly on hold.

## Reproduce and locate evidence

Generating script: [test_matched_2050_temperature.py](../../code/post_processing_layers/test_matched_2050_temperature.py).
No figures were generated or modified by this test.

Outputs under the external data root:
`2_postprocess_intermediate/UCM_official_runs/diagnostics/matched_2050_temperature_20261111_v1/`:

- `summary.csv`: 30 landscape/configuration rows, both aggregation methods.
- `borough.csv`: 990 borough/landscape/configuration rows.
- `manifest.json`: source hashes, calibration targets and translation errors.

```sh
conda run -n urban-cooling-invest-3.20.2 python \
  code/post_processing_layers/test_matched_2050_temperature.py \
  --data-root "$URBAN_COOLING_DATA_ROOT" \
  --output-dir "$URBAN_COOLING_DATA_ROOT/2_postprocess_intermediate/UCM_official_runs/diagnostics/matched_2050_temperature_20261111_v1"
```

The script refuses to overwrite an existing diagnostic folder. Use a new suffix
for a repeat. Original climate and run history remain in the
[Figure 1 reference comparison](figure1_historical_reference_comparison.md) and
[Figure 3 audit](figure3_climate_uhi_audit.md).
