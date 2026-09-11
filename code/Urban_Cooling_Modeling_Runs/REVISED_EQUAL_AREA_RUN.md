# Revised equal-area Urban Cooling run

## Scope

The final revised scenario set was run with InVEST 3.20.2 at the approved
25°C primary and 28°C sensitivity settings. The batch includes the baseline
and Green10/20/30 and Target10/20/30, uses identical model parameters, and
writes temperature-only results to a new versioned directory:

```text
2_postprocess_intermediate/UCM_official_runs/
└── revised_equal_area_invest3202_2026-09-10/
```

Use
[`execute_invest_urban_cooling_model_revised_scenarios.py`](execute_invest_urban_cooling_model_revised_scenarios.py)
to validate or reproduce the run. Each of the fourteen temperature rasters has
a colocated JSON manifest containing input and output checksums, model
arguments, grid metadata, software versions and Git commit `f54b2cd`.

## Quality checks

- All fourteen configurations passed InVEST validation before execution.
- All fourteen runs completed and produced a temperature raster and manifest.
- Every temperature raster has the same 23,089,980-cell valid grid.
- The 28°C field equals its corresponding 25°C field plus 3°C within
  float32 precision (maximum cooling-difference discrepancy <0.000021°C).
- Historical UCM workspaces were not overwritten.

## Citywide temperature comparison

The table reports the unweighted mean change across the common valid raster
domain. Population-weighted health effects must be calculated by the health
model rather than inferred from these values.

| Scenario | Mean cooling vs baseline (°C) | 95th percentile (°C) | Maximum (°C) |
|---|---:|---:|---:|
| Green10 | 0.07199 | 0.22134 | 0.36002 |
| Target10 | 0.07883 | 0.38567 | 0.85062 |
| Green20 | 0.13782 | 0.38618 | 0.59080 |
| Target20 | 0.15041 | 0.61165 | 1.00392 |
| Green30 | 0.20521 | 0.55782 | 0.81511 |
| Target30 | 0.22315 | 0.75233 | 1.06957 |

Target's mean cooling exceeds equal-area Green by approximately 9.5%, 9.1%
and 8.7% at levels 10, 20 and 30. This is a model result for the revised
spatial allocations, not evidence that every Target candidate is practically
plantable. The Target input is dominated by screened locations whose
underlying LCM class is Urban; that composition difference is documented in
the land-cover scenario audit.

## Next dependent step

Run the population-weighted health analysis using
`health-analysis-revised-equal-area.example.json`. It fixes 2021 population,
2021 registered mortality, 2,000 cause-stable draws and seed `20260908` across
all twelve Green/Target scenario-temperature combinations.

## Energy and productivity valuation addendum

The same seven final scenarios were rerun at the primary 25°C setting with
InVEST 3.20.2 energy valuation and WBGT enabled. Results are isolated from the
temperature/health workspaces under:

```text
2_postprocess_intermediate/UCM_official_runs/
└── revised_equal_area_invest3202_2026-09-11_valuations/
```

Each scenario contains a unique-building energy layer, WBGT raster and run
manifest. Project-specific heavy-work productivity was then calculated with
the continuous Hothaps equation, not InVEST's built-in threshold classes. The
versioned `summary/` directory contains citywide, borough and
revised-versus-original tables plus their manifests. These aggregate results
remain on the restricted shared drive until publication in Git is explicitly
approved. See
[`../post_processing_layers/README.md`](../post_processing_layers/README.md)
for the complete commands, validation rules and historical comparison method.
