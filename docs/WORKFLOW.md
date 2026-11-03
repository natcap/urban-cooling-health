# Reproducing the full analysis

This is the ordered map from source data to manuscript figures. Each stage
links to its authoritative procedure; commands and parameters live in those
stage guides so they are not copied into several documents. The current
Figure 7 can be reproduced from tracked derived inputs without completing
the external-data stages; use the separate [Figure 7 guide](../REPRODUCIBILITY.md).

## Before starting

1. Read the [data directory guide](../data/README.md) and
   [release review](../DATA_LICENCE_AND_REDISTRIBUTION.md). Raw LCM, scenario
   and UCM rasters are not distributed through this repository. Obtain
   authorized access to the versioned Wellcome Trust Project Data store for a
   full rebuild.
2. Set `URBAN_COOLING_DATA_ROOT` to that store and install the environments
   described in the [UCM workflow guide](../code/workflows/README.md) and
   [health guide](../code/health_assessment/README.md). The historical
   `HEALTH_DATA_ROOT` name is a compatibility alias, not a second production
   data root.
3. Choose an immutable input snapshot and a new, dated output directory.
   Record source versions, licences, checksums, CRS, units, geography and
   configuration as specified in the [data guide](../data/README.md). Do not
   overwrite manuscript-era outputs to make a new run appear historical.

## Run order and hand-offs

| Stage | Authoritative guide and entry point | Check before moving on |
|---|---|---|
| 1. Prepare baseline and planting scenarios | [Land-cover scenarios](../code/lc_scenarios/README.md), including the revised Green/Target generators | Preserve the baseline and point/rank provenance; run the full Green–Target grid, eligibility, nesting and equal-canopy QA. Historical `510/520/530` aliases are not production inputs. |
| 2. Run cooling and co-benefit post-processing | [UCM workflow](../code/workflows/README.md) and [post-processing](../code/post_processing_layers/README.md); start with `code/workflows/run_ucm_pipeline.py` | Validate configuration first. Keep the primary 25°C valuation and 28°C temperature sensitivity separate; retain run manifests and checksums. Use project-specific Hothaps workability, not InVEST's built-in threshold work-loss layer, for the manuscript. |
| 3. Run health assessment | [Health assessment](../code/health_assessment/README.md), using `run_health_scenario_set.py` with the revised equal-area configuration | Use 2021 population and registered mortality for both climate settings; validate all selected rasters on one grid and retain paired exposure-response draws. |
| 4. Aggregate to official geography | [Health LSOA aggregation](../code/health_assessment/README.md#aggregate-the-revised-outputs-to-lsoa11) and the post-processing guide | Join by official `LSOA11CD`; verify one-to-one coverage of the 4,835 production LSOA11 areas and reconcile borough/city totals. |
| 5. Make figures and review claims | [Code/figure index](../code/README.md), [Figure 4 methods](../code/post_processing_layers/FIGURE4_METHODS.md), and [Figure 7 guide](../REPRODUCIBILITY.md) | Check scenario order, estimands, units, source CSVs, uncertainty type and figure appearance. Figure 6 remains exploratory pending spatial inference; see [Figures 6–7 review](../code/health_assessment/FIGURES6_7_REVIEW.md). |

The [project TODO](../TODO.md) records unresolved scientific, provenance and
release gates. A successful run is not evidence that a manuscript claim or a
public data release is approved. The [manuscript revision brief](manuscript/MANUSCRIPT_REVISION_BRIEF.md)
collects the current interpretation and figure-order decisions.
