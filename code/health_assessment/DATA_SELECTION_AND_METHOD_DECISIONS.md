# Health data-selection and method decisions

Last reviewed: 9 September 2026

This record explains choices that affect the revised health assessment. It is
intended to accompany the scripts, QA tables and run manifests so future users
can distinguish source observations from modeled spatial allocation.

## 1. Mortality source and selection

**Decision:** use `data/health-data/387056911188607_data.tsv` for the revised
2021 mortality preparation.

**Selection:** calendar year 2021; 33 London boroughs (`E09...`); total gender;
all ages; registered deaths; and the following underlying-cause groups:

| Model name | Nomis cause group |
|---|---|
| `all_cause` | `A00-R99,U00-Y89 All causes, all ages` |
| `mental_disorder` | `F00-F99 V Mental and behavioural disorders` |
| `cardiovascular` | `I00-I99 IX Diseases of the circulatory system` |
| `respiratory` | `J00-J99 X Diseases of the respiratory system` |
| `self_harm` | `X60-X84 Intentional self-harm` |

**Justification:** this export contains the five definitions already used by
the health model. The newer `842881776194293_data.tsv` lacks the broad
`I00-I99` circulatory category and instead contains `I20-I25` and `I60-I69`.
Those subgroups are not a complete replacement for all circulatory deaths and
must not be summed as though they were equivalent.

The source exports contain exact duplicate records. The converter removes only
fully identical rows and rejects conflicting duplicate borough/cause records.
Nomis values are numbers of deaths registered in the calendar year, not
occurrence-date counts or mortality rates.

## 2. Population year

**Decision:** use the 2021 population raster for every health comparison,
including both the 25°C and 28°C temperature settings.

**Justification:** a robust, spatially compatible future population projection
is not available for this project. Holding population and baseline mortality
at 2021 values isolates the modeled temperature and land-cover difference.
Results must therefore be described as temperature scenarios evaluated under
2021 population and mortality, not demographic projections for a future year.

**Source and preprocessing decision:** create the production population surface
directly from the raw WorldPop count raster
`gbr_pop_2021_CN_100m_R2025A_v1.tif`. Reproject population counts to the 10 m
UCM grid with sum resampling, then retain pixels whose centres fall within a
London borough. Preserve the raw raster and all older processed rasters.

The legacy preprocessing notebook used bilinear resampling on population
counts. Its full-GB result totals 35.218 million, compared with 67.403 million
in the raw raster, and its London product totals only 4.743 million. Bilinear
interpolation is appropriate for continuous surfaces but does not conserve an
extensive quantity such as people. The old
`gbr_pop_2021_10m_areal.tif`, the later CRS-only copy, and mortality rasters
allocated from either one are retained for audit only and must not be used for
production results.

The supplied download page is WorldPop listing 135. That page describes the
100 m constrained R2025A v1 series; the specific UK 2021 record is item 75899,
produced 1 September 2025 under DOI `10.5258/SOTON/WP00839`. The corrected
London raster totals 8,834,023.32 people, 0.387% above the
[published 2021 Census benchmark of 8.8 million](https://data.london.gov.uk/download/24fd8ec4-c65a-4c17-947b-e6f8c40d2c11/18e4a444-11a7-4777-b383-8cf199176157/2021%20census%20first%20release.pdf).
The existing file's name identifies release `R2025A`, but its embedded tags say
`WorldPop R2024B v1`. A fresh copy was retained separately from the
[official R2025A download](https://data.worldpop.org/GIS/Population/Global_2015_2030/R2025A/2021/GBR/v1/100m/constrained/gbr_pop_2021_CN_100m_R2025A_v1.tif)
on 9 September 2026. The fresh official file contains the same stale R2024B
tags. Its CRS, affine transform, dimensions, NoData definition and all pixel
values are identical to the existing project raster; both total
67,403,041.1927 people. The files have different binary SHA-256 hashes, so they
are not byte-for-byte copies, but there are zero differing raster cells.

**Conclusion:** treat the TIFF tags as an upstream WorldPop packaging-metadata
error. Identify the analytical input as R2025A v1 using listing 135, item
75899, DOI and the direct download URL, and retain its checksum. The existing
project raster remains analytically valid and does not need replacement.

## 3. Figure 7 population geography

**Finding:** the current `health_sf.rds` contains 4,835 unique neighbourhood
IDs, corresponding to London LSOA 2011 geography. The official Census 2021
TS001 bulk file contains 4,994 London LSOA 2021 records totaling 8,799,776
usual residents. ONS confirms that Census 2021 outputs use LSOA 2021 geography,
including LSOAs that were split or merged after 2011.

The official TS001 archive was downloaded from Nomis to
`data/health-data/ons-census-2021-ts001/`. It is a valid reference source but
must not be joined directly to the current 4,835-row Figure 7 geography.

**Approved production choice:** retain the established LSOA 2011 Figure 7
geography and calculate its 2021 denominators by zonally summing the same
count-preserved WorldPop surface used by the health model. This avoids an
unverifiable row-order join and keeps pixel- and LSOA-level population
internally consistent. Label these as modeled WorldPop 2021 estimates, and use
official TS001 totals as an external validation check. The alternative is to
rebuild vulnerability inputs, zonal statistics and Figure 7 on LSOA 2021
boundaries.

**Implemented result (10 September 2026):**
`prepare_lsoa_population_2021.R` generated 4,835 positive, official-code
denominators totaling 8,832,324.67. This is 0.370% above the TS001 London total
and 0.019% below the masked raster total. Eight ring self-intersections were
repaired deterministically with `sf::st_make_valid`; their `LSOA11CD` values
and all checksums are recorded in the generation manifest. The new values are
exactly identical to the earlier numeric-ID lookup, but population,
vulnerability and health results now join exclusively by official code. The
legacy `health_sf.rds` and numeric lookup remain historical audit inputs only.

## 4. Population-weighted mortality allocation

**Decision:** distribute each borough's observed cause-specific deaths among
10 m pixels in proportion to the 2021 population:

```text
pixel deaths = borough registered deaths
             × pixel population_2021
             / borough population_2021
```

**Justification:** mortality events arise from a population at risk. The legacy
area-weighted approach assigns equal deaths per square metre throughout a
borough, including sparsely populated parks, industrial land and other areas.
Population weighting locates the modeled baseline mortality burden where
residents are represented while preserving every observed borough/cause total.
Pixels with no population receive no allocated deaths.

**Boundary rule:** assign a pixel only when its centre falls inside a borough
(`all_touched=False`). Do not use `all_touched=True`, because adjacent boroughs
can touch the same pixel and assignment would then depend on feature order.
Population outside the official borough polygons is excluded because the
mortality totals cover London boroughs only. The population-preparation script
applies this mask before allocation, so the revised mortality run assigns no
deaths outside the boroughs. The manifest records both the count before masking
and the London total after masking.

**Interpretation and limitations:** the pixel values are modeled allocations,
not observed pixel-level deaths. Within each borough, the method assumes equal
cause-specific mortality per resident because compatible small-area
cause-specific mortality and demographic risk strata are unavailable. It does
not capture within-borough differences in age, health status or baseline risk.
The legacy area-weighted rasters are retained for the documented regression
comparison, and borough-total conservation must pass before derived rasters are
used.

The deterministic 25°C comparison is recorded in
[`HEALTH_VERSION_COMPARISON.md`](HEALTH_VERSION_COMPARISON.md). It is diagnostic,
not a replacement for the final paired Monte Carlo analysis.

## 5. Temperature setting

**Decision:** use the 25°C setting as the primary manuscript analysis and the
28°C setting as a sensitivity analysis. Use identical population, mortality,
relative-risk parameters, scenario definitions, random seed and Monte Carlo
draw count in both settings.

## 6. Target30 scenario identity

**Decision:** existing `scenario530` UCM outputs identify the legacy Target30
v3 calculation, but may no longer be used for the final equal-budget analysis.

**Justification:** `LULC_Scenario530.tif` and the renamed
`LULC_Scenario730v3.tif` are byte-for-byte identical. The final Target30 input
is the separately generated `LULC_Scenario730v4_equal_budget.tif`; its UCM and
health outputs must be regenerated. The v3 equivalence does not apply to the
Target10 or Target20 old/new raster pairs, which remain under review.

## 7. Realized canopy budget

The LULC files referenced by the current UCM scripts do not provide equal
realized additions. Relative to
`LCM2023_London_10m_clip2aoi_tcc24.tif`, Green30 adds 930,000 tree pixels
(93.0000 km²; 29.890% of baseline canopy), while `LULC_Scenario530.tif` adds
869,444 pixels (86.9444 km²; 27.944%). Target30 therefore adds 60,556 fewer
pixels, a 6.511% deficit relative to Green30, and fails the prespecified 0.5%
tolerance.

**Approved and implemented resolution (9 September 2026):** extend the v3
scenario using the next features in the original ascending vulnerability rank,
then stop at exactly 930,000 realized added pixels. The new
`LULC_Scenario730v4_equal_budget.tif` adds 60,556 pixels to v3 and matches
Green30 at 930,000 pixels (93 km²), removes no baseline canopy and passes with
a 0.000% difference. Its manifest verifies five sampled prefix positions,
records rank cutoff 1,948,394 and hashes all inputs. Existing UCM, health and
Figure 7 scenario results remain provisional until rerun with this raster.

## Reproducibility evidence

Retain together for each production run:

- the original Nomis data and geography TSV files;
- `mortality_2021_long.csv`;
- the raw and count-preserved 2021 population rasters and the preparation
  manifest;
- population-weighted mortality rasters, `allocation_qa.csv` and
  `manifest.json`;
- the health JSON configuration and `run_manifest.json`; and
- the Git commit and software environment used for the run.

## Preparation run recorded on 9 September 2026

The verified preparation run produced 165 standardized records for 33 boroughs
and five causes. Registered-death totals were: all cause 56,945;
cardiovascular 12,937; mental and behavioural disorders 3,544; respiratory
4,810; and intentional self-harm 438.

The corrected population raster is EPSG:27700, 10 m resolution and totals
8,834,023.32 people inside London boroughs. The five population-weighted
mortality rasters use the exact UCM grid. The maximum absolute borough
allocation difference was `1.60e-09` deaths, which is numerical rounding rather
than loss of cases. All four configured input sets (`green30_25c`,
`target30_25c`, `green30_28c` and `target30_28c`) passed the revised model's
exact common-grid validation.

A deterministic 25°C regression run found that revised all-cause deaths
averted changed from 379.37 to 391.82 for Green30 and from 386.36 to 480.91 for
Target30. The Target30-minus-Green30 difference therefore changed from 6.99 to
89.09 deaths averted. These values are provisional until the equal-canopy,
paired uncertainty and LSOA output checks are complete.

This was a development run from a dirty Git worktree. The manifests record that
status and the preparation-script checksums. Before publication, commit the
reviewed code and rerun so the archived manifests point to a clean commit.
