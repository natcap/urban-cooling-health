# Revised equal-area health results

## Analysis definition

These results use the revised equal-realized-canopy Green10/20/30 and
Target10/20/30 temperature fields from InVEST 3.20.2. The health model holds
2021 population and 2021 registered mortality fixed, allocates mortality by
pixel population, and uses 2,000 cause-stable exposure-response draws with
seed `20260908`.

The 25°C setting is the primary analysis. Results must be described as 2050
temperature scenarios evaluated under fixed 2021 population and mortality,
not as a demographic projection for 2050.

## Citywide all-cause results

| Level | Green deaths averted, mean (95% interval) | Target deaths averted, mean (95% interval) | Paired Target minus Green (95% interval) | Target advantage |
|---|---:|---:|---:|---:|
| 10 | 135.5 (94.9–174.4) | 188.4 (131.8–242.6) | 52.9 (37.0–68.2) | 39.0% |
| 20 | 258.7 (181.1–333.1) | 346.1 (242.1–445.8) | 87.4 (61.0–112.7) | 33.8% |
| 30 | 388.4 (271.8–500.3) | 496.9 (347.5–640.3) | 108.5 (75.7–140.0) | 27.9% |

The paired difference is positive in all 2,000 draws at every intervention
level. These intervals propagate exposure-response uncertainty only; modeled
temperature, population and baseline mortality are held fixed.

The 28°C results are numerically equivalent to the 25°C results because both
the matched baseline and intervention fields shift upward by 3°C and the
current health model responds to their difference. The 28°C run therefore
checks implementation consistency but does **not** test how absolute heat risk
changes at a hotter baseline. A true absolute-temperature sensitivity would
require a temperature-dependent baseline-risk or threshold formulation.

## Figure 7 equity result

For the main Green30–Target30 comparison, Target30 reduces the share of
high-vulnerability LSOAs receiving relatively low mortality benefit for:

- income deprivation: 41.6% to 6.1% (-35.5 percentage points);
- ethnic-minority composition: 51.4% to 3.5% (-47.9 points).

The result reverses for the age-75+ measure: the low-benefit share increases
from 28.7% to 59.0% (+30.3 points), and high-age-75 LSOAs receive 26.8 fewer
deaths averted in aggregate under Target30. This is not a plotting artifact;
age-75+ vulnerability is negatively correlated with income deprivation,
ethnic-minority composition and the composite SVI used in targeting. The
manuscript should present this as a distributional trade-off, not claim that
Target30 improves equity for every vulnerability dimension.

## Reproducible outputs

Health rasters and run manifests are stored externally under:

```text
2_postprocess_intermediate/UCM_official_runs/
└── health_v3_revised_equal_area_population_weighted_2021_2026-09-10/
```

Small LSOA inputs, paired draws, Figure 7 panels and quantitative tables are
versioned in `data/derived/` and
`figures/equity_map_biscale_revised_equal_area/`.
