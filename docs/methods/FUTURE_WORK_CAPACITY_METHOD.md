# Future worker-exposure-weighted work-capacity model

## Purpose and status

This document specifies a feasible upgrade from the current area-weighted
heavy-work-capacity indicator to a worker-exposure-weighted model. It is a
future method-development plan, not part of the present manuscript analysis.

For the current manuscript, continue to use **modelled heavy-work capacity**.
The existing workflow converts InVEST WBGT rasters to a continuous Hothaps
workability fraction and averages scenario changes across valid 10 m land
pixels. It does not represent the locations, occupations, schedules or actual
output of London workers. The upgrade described here would change the
estimand to capacity-weighted worker-hours for a defined exposed workforce.

## Recommended first implementation

Start with a construction-sector pilot. Construction provides a comparatively
clear test population with substantial outdoor and heavy-work exposure. A
London-wide model covering every occupation should follow only after the pilot
has defensible exposure assignments and validation evidence.

The pilot should answer:

> How do land-use scenarios change potential heavy-work capacity during
> working hours for construction workers at their estimated work locations?

Report physical work-capacity outcomes first. Monetization should remain a
secondary extension because it introduces further assumptions about wages,
hours, substitution, adaptation and the relationship between physiological
capacity and economic output.

## Primary estimand

For scenario `s`, spatial unit `i`, worker group `g` and time step `t`, define:

```text
capacity_weighted_worker_hours_s =
    sum(workers_i,g × outdoor_share_g × hours_g,t × workability_s,i,g,t)
```

The scenario effect is:

```text
change_s = capacity_weighted_worker_hours_s
           - capacity_weighted_worker_hours_baseline
```

Calculate scenario and baseline under the same climate period, workforce,
working schedule and valid spatial support. Report both:

- change in capacity-weighted worker-hours; and
- percentage change relative to the matched baseline.

This quantity is still modelled potential capacity. It should not be described
as observed labour productivity, output, earnings or gross value added.

## Minimum viable data

Create a versioned data inventory before implementation. Every input record
should include source, release/date, geography, units, licence, checksum and
preprocessing history.

| Input | Minimum requirement | Preferred source or resolution |
|---|---|---|
| Workplace employment | Worker counts for construction or selected occupations at work location | Workplace Zone preferred; MSOA or borough acceptable for a pilot |
| Occupation or industry | A documented mapping to metabolic work intensity | SOC or SIC categories with an explicit crosswalk |
| Outdoor exposure | Proportion of working time outdoors for each worker group | Empirical UK evidence; otherwise a documented low/central/high range |
| Working hours | Representative hourly schedule and workdays | Sector-specific schedule; retain hours explicitly rather than assuming all-day exposure |
| Heat exposure | Hourly WBGT or inputs needed to calculate it | Air temperature, humidity, solar radiation and wind on a common grid |
| Scenario temperature | Matched baseline and land-use-scenario fields | Same grid, climate period and mask for every scenario |
| Workability relationship | Function matched to work intensity and acclimatization assumptions | Retain the existing Hothaps heavy-work function for the first comparison, then test alternatives |
| Validation evidence | Observed disruption, work-rest behaviour, symptoms, absence or survey data | London or comparable UK construction evidence where available |

Do not use residential population as a substitute for workplace employment.
If only borough workplace totals are available, describe the result as a
borough-level exposure allocation and do not imply 10 m knowledge of worker
locations.

## Implementation stages

### Stage 1: Define the worker population

1. Select one construction definition using documented SIC and/or SOC codes.
2. Retain workplace-based counts, not residence-based counts.
3. Record the reference year and decide whether the workforce remains fixed
   across climate and land-use scenarios.
4. Check London totals against the published source total.
5. Report missing, suppressed or geographically unmatched records.

Recommended pilot decision: hold the workforce fixed at the best available
baseline year so the comparison isolates land-use and heat-exposure effects.

### Stage 2: Assign work intensity and outdoor exposure

Create a crosswalk with one row per worker group and these fields:

```text
worker_group
source_classification
metabolic_intensity
outdoor_share_low
outdoor_share_central
outdoor_share_high
working_hours_profile
evidence_source
assumption_note
```

Use ranges for outdoor-work proportions when direct estimates are uncertain.
Do not select assumptions by fitting them to the desired scenario result.

### Stage 3: Construct hourly heat exposure

The preferred exposure metric is hourly WBGT calculated from air temperature,
humidity, solar radiation and wind. A daily or monthly air-temperature proxy
cannot represent within-day work exposure with the same interpretation.

For each scenario:

1. align all weather and land-use outputs to one CRS, grid and time zone;
2. calculate or obtain hourly WBGT for representative hot days or a defined
   warm-season period;
3. use identical time steps and masks for baseline and scenario;
4. retain NoData rather than treating it as zero exposure; and
5. record whether conditions represent shade, sun or a mixture relevant to the
   selected occupations.

If hourly WBGT cannot be produced, implement a clearly labelled interim
sensitivity using the current WBGT surface and representative working hours.
Do not present that interim calculation as a full occupational-exposure model.

### Stage 4: Calculate workability

For the first pilot, retain the existing continuous heavy-work relationship:

```text
workability = 0.1 + 0.9 / (1 + (WBGT / 30.94)^16.64)
```

The output is a fraction between 0.1 and 1.0. A scenario-minus-baseline
difference of `0.01` equals a 1 percentage-point change in modelled capacity.

Before using this function for occupational inference:

- confirm the original evidence base and intended acclimatization/work-intensity
  conditions;
- verify that WBGT units and range match the function;
- compare at least one alternative workability or work-rest formulation; and
- document whether clothing, direct sun and adaptation are represented.

### Stage 5: Allocate workers spatially

Use the finest defensible workplace geography. Allocate workers to heat-grid
cells only when an employment or workplace-area layer supports that allocation.

Preferred allocation order:

1. observed workplace points or establishment employment, where permitted;
2. workplace-zone employment distributed over workplace land/building area;
3. MSOA employment distributed over documented employment land; or
4. borough totals retained at borough resolution.

Avoid uniform redistribution across all land pixels. Preserve source totals
after allocation and export a reconciliation table by geography and group.

### Stage 6: Aggregate worker exposure

For every scenario and time step:

1. multiply workability by workers, outdoor share and scheduled hours;
2. sum capacity-weighted worker-hours across the defined workforce;
3. subtract the matched baseline result;
4. calculate uncertainty or sensitivity across outdoor-share, schedule and
   workability assumptions; and
5. aggregate to London and borough while preserving the same underlying worker
   weights.

Keep unweighted land-area results as a bridge to the current manuscript model,
but label them as a separate estimand.

### Stage 7: Validate before expanding scope

Validation should include:

- workforce totals reproduced exactly or within documented rounding;
- spatial allocations summed back to their source geography;
- no scenario-dependent changes in worker weights or masks;
- WBGT and workability ranges checked for every scenario and hour;
- comparison with observed hot-day work interruption, work-rest practices,
  absence, symptoms or survey evidence where available;
- sensitivity to outdoor-work share, work schedule, acclimatization and access
  to shade or indoor cooling; and
- comparison of area-weighted and worker-weighted scenario rankings.

Agreement with the present area-weighted result is not validation. The purpose
of the upgrade is to test whether worker location and timing change the
estimated magnitude or ordering of scenario benefits.

## Required outputs

Write outputs to a new versioned folder; do not overwrite the existing Hothaps
rasters or Figure 4 summaries.

Minimum outputs:

```text
worker_exposure_config.yml
worker_group_crosswalk.csv
workforce_allocation_reconciliation.csv
hourly_wbgt_manifest.json
capacity_weighted_worker_hours_by_scenario.csv
borough_capacity_weighted_worker_hours.csv
assumption_sensitivity.csv
validation_summary.md
run_manifest.json
session_or_environment_record.txt
```

The run manifest should record input checksums, code commit, parameter values,
time zone, spatial/temporal support, software versions, random seeds and output
checksums. Export exact figure-source tables beside any plots.

## Suggested script organization

Add future implementation scripts only after the inputs and estimand have been
reviewed. Suggested names are:

```text
prepare_workplace_employment.py
build_worker_exposure_crosswalk.py
calculate_hourly_wbgt.py
allocate_workers_to_heat_grid.py
calculate_worker_weighted_capacity.py
validate_worker_capacity_model.py
```

Each script should accept explicit configuration or command-line paths, support
a validation-only mode, refuse accidental overwrite, and write a manifest.
Integrate the workflow into `code/workflows/run_ucm_pipeline.py` only after the
standalone pilot is validated.

## Decision gates

Do not advance to the next stage until the preceding decision is documented:

1. **Population:** exact construction workforce definition and reference year.
2. **Exposure:** defensible workplace geography and outdoor-work proportions.
3. **Heat metric:** hourly WBGT method, period, sun/shade interpretation and
   valid mask.
4. **Response:** workability/work-rest relationship and applicability.
5. **Validation:** acceptable external evidence and sensitivity bounds.
6. **Reporting:** physical capacity outcome approved before monetization.

## Future monetization

Only consider monetization after the physical exposure model is validated.
Possible inputs include hourly compensation or GVA per worker-hour, but the
calculation must address whether lost physiological capacity translates into
lost output, schedule shifting, work-rest rules, substitution, paid absence and
adaptation. Report monetary results as scenario-dependent estimates under
explicit assumptions, not avoided economic loss observed in London.

## Manuscript language for a future upgraded model

If this upgrade is later implemented and validated, a cautious description is:

> We estimated changes in potential heavy-work capacity by weighting hourly
> WBGT-derived workability by workplace-based worker counts, occupation-specific
> outdoor-exposure proportions and scheduled working hours. Results represent
> modelled capacity-weighted worker-hours rather than observed productivity or
> economic output.

Until then, the current manuscript should retain **modelled heavy-work
capacity** and its area-weighted limitation.
