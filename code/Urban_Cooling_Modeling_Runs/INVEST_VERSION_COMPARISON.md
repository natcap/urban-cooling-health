# Why InVEST 3.20.2 produces different Urban Cooling temperatures

## Technical summary

The approximately **+0.438 C** mean difference between the InVEST 3.20.2 and
3.14.1 Target30 temperature rasters is real and is primarily caused by a
documented Urban Cooling bug fix released in InVEST 3.15.0. InVEST 3.14.1
calculated park cooling by convolving a binary green-area raster. The intended
equation requires each green pixel to be weighted by its Cooling Capacity (CC).
InVEST 3.15.0 and later correctly convolve `green_area * CC`.

A second 3.15.0 change aligns outputs to the LULC grid instead of the ET0 grid.
For this project, the park-cooling correction raises mean air temperature by
about **+0.501 C** on a common grid, while the alignment change and its
associated resampling effects offset about **-0.063 C**. Their net effect is
approximately **+0.438 C**. Changes in GDAL and pygeoprocessing do not explain
the discrepancy: rerunning the old formula with the new computational engine
reproduced the 3.14.1 output within about `0.0000001 C` mean absolute error.

**Decision:** use InVEST 3.20.2 for all new production results, but regenerate
the baseline and every compared intervention with 3.20.2. Do not mix a 3.20.2
Target30 raster with 3.14.1 baseline or Green30 rasters.

## The two numerical changes entered in InVEST 3.15.0

The official [InVEST 3.15.0 release notes](https://github.com/natcap/invest/releases/tag/3.15.0)
identify both changes under Urban Cooling.

### 1. Park cooling now includes the Cooling Capacity Index

Issue [#1726](https://github.com/natcap/invest/issues/1726) documented that the
old source code used only `green_area` in the exponential convolution even
though the documented equation is:

`CC_park(i) = sum(g(j) * CC(j) * exp(-d(i,j) / d_cool))`.

Commit [`2ec399307`](https://github.com/natcap/invest/commit/2ec3993074db5e5e672c0cc89857103a512aba1d)
fixed the implementation. The relevant execution path changed from:

```text
3.14.1: cc_park = exponential_convolution(green_area)
3.15+:  cc_park = exponential_convolution(green_area * CC)
```

Because `CC` is generally below 1, the 3.14.1 calculation overestimated the
cooling influence of parks. Correcting it reduces `cc_park` and heat mitigation,
which makes the corrected air-temperature raster warmer. This is not a change
to project parameters; it is a correction to the model implementation.

### 2. Outputs now align to the LULC grid

Commit [`84fe059a0`](https://github.com/natcap/invest/commit/84fe059a0e17ae9efbc1bd88ace16eb543bcb271)
changed `raster_align_index` from `1` (ET0) to `0` (LULC). The reason was to
preserve the grid coordinates of the primary categorical input and avoid
outputs whose origins are offset by a non-integer number of pixels from other
prepared rasters. The general motivation is described in
[issue #1488](https://github.com/natcap/invest/issues/1488).

This explains the project grid difference:

| Output | Size | Upper-left origin | Pixel size |
|---|---:|---:|---:|
| InVEST 3.14.1 Target30 | 5,887 x 4,552 | 503089.4839, 200935.6039 | 10 m |
| InVEST 3.20.2 Target30 | 5,840 x 4,509 | 503560, 200940 | 10 m |

The 3.20.2 grid exactly matches the current 2021 population raster and the
existing baseline and Green30 grid. Grid equality alone does not make mixed
versions scientifically comparable because the park-cooling algorithm also
changed.

## Function-level source review

An abstract-syntax-tree comparison of the installed 3.14.1 and 3.20.2 Urban
Cooling modules found the following:

| Function | Status | Relevance to temperature results |
|---|---|---|
| `execute` | Changed | Contains both the LULC-alignment change and corrected `cc_park` execution path; this is the material change. |
| `mask_cc_green_areas_op` | Added | Implements `green_area * CC` with correct nodata handling before park convolution. |
| `calc_cc_op_factors` | Unchanged | Project uses this factors method; its CC equation did not change. |
| `calc_eti_op` | Unchanged | ETI equation did not change. |
| `hm_op` | Unchanged | Heat-mitigation selection equation did not change; its `cc_park` input changed. |
| `calc_t_air_nomix_op` | Unchanged | Unmixed air-temperature equation did not change. |
| `convolve_2d_by_exponential` | Numerically unchanged | Only a documentation typo changed in this wrapper. |
| `calculate_wbgt`, `map_work_loss` | Unchanged | No effect on `T_air`; valuations were disabled in the production run. |
| `calculate_uhi_result_vector` | Output-type update | Float casting affects vector output compatibility, not raster values. |
| `calculate_energy_savings` | Validation refactor | No effect because energy valuation was disabled. |
| `validate` and model specification | Refactored | Changes validation and file registration, not the factors-method temperature calculation. |

InVEST 3.17.2 also fixed nodata handling for the **intensity** CC method. This
project uses `cc_method = factors`, so that fix is not relevant to the observed
difference.

## Quantitative decomposition

The comparison used the already-created 25 C Target30 v4 outputs. No additional
baseline or intervention scenario was run for this diagnosis.

| Component | Mean effect on `T_air` (C) | Interpretation |
|---|---:|---|
| Corrected park-cooling calculation, holding the 3.20.2 grid and all other calculations fixed | +0.5012 | Dominant driver; removes cooling that 3.14.1 overstated. |
| Alignment/resampling contribution and negligible residual numerical effects | -0.0633 | Partially offsets the park correction. |
| Observed net 3.20.2 minus aligned 3.14.1 | +0.4378 | Matches the combined effect. |

The observed comparison included 23,088,902 common valid pixels after
bilinearly aligning the continuous 3.14.1 temperature raster to the 3.20.2
grid. Its median difference was +0.4016 C, mean absolute difference was
0.4379 C, 95th-percentile absolute difference was 1.1458 C, and maximum
absolute difference was 1.9979 C.

The intermediate rasters support the same mechanism:

| Intermediate output | Mean 3.20.2 minus aligned 3.14.1 |
|---|---:|
| Reference ET0 | -0.00008 mm |
| Cooling Capacity | +0.00722 |
| Park Cooling Capacity | -0.13067 |
| Heat Mitigation | -0.08795 |
| Unmixed air temperature | +0.43975 C |
| Final mixed air temperature | +0.43782 C |

The near-zero ET0 difference rules out the climate raster values as the main
cause. The large reduction in `cc_park`, followed by reduced heat mitigation
and warmer temperature, follows the corrected formula directly.

## Robustness checks and limitations

- Applying the current pygeoprocessing convolution to the old 3.14.1
  intermediates reproduced the old final `T_air` with mean absolute difference
  `0.000000086 C` and maximum absolute difference `0.00000763 C`. Dependency
  upgrades therefore have negligible explanatory power here.
- The old 3.14.1 run was interrupted during optional building-energy processing,
  after `T_air` and its required upstream intermediates had completed. It is
  suitable for this algorithm diagnostic, but it should not be treated as a
  complete production run.
- The alignment contribution is estimated as the residual after holding the
  3.20.2 grid fixed for the park-formula counterfactual. It includes small
  raster-resampling effects and cannot be interpreted as a universal effect of
  LULC alignment for other datasets.
- The findings apply to this project's factors-method inputs and parameters.

## Production follow-up status

All required follow-up steps are complete:

1. Baseline and Green30 were rerun with InVEST 3.20.2 at 25 C and 28 C in
   versioned workspaces.
2. Target30 v4 was run with InVEST 3.20.2 at both temperature settings after
   its realized canopy budget was matched exactly to Green30.
3. Every production `T_air` raster matches the 2021 population grid exactly.
4. The version effect, Green30 NoData sensitivity and scenario effects were
   recorded separately in machine-readable comparison tables.
5. The population-weighted health assessment and Figure 7 were regenerated
   from the consistent 3.20.2 outputs.
6. Checksums, parameters, environment versions and validation results are
   retained beside the model workspaces and in the tracked audit tables.
