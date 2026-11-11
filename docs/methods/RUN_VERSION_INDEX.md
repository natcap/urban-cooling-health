# Run versions and current manuscript source package

Reviewed 10 November 2026. Folder dates identify processing stages, not competing
complete analyses. V0/V1 below are analytical labels; `health_v2/v3` are health
pipeline generations; InVEST 3.14.1/3.20.2 are software versions.

External root: `Wellcome Trust Project Data/2_postprocess_intermediate/UCM_official_runs/`.
Each run directory has a concise `README_RUN.md` identifying its role. Physical
folders were reorganized on 10 November; old paths remain compatibility links.
Model outputs and original provenance manifests are preserved unchanged.

Subsequent Figure 4–5 layout update: existing PNG/CSV files were moved unchanged
into `summary/`. Ten PDF/SVG exports and the original figure manifests are retained
under `archive/retired_figure_exports_20261110/`. Current figure manifests were
updated only for paths and the PNG-only policy; original manifests remain intact.
The exact moves and checksums are recorded in
`summary/figure_output_relocation_20261110.json`. The old figure4/figure5
subdirectories were removed after becoming empty; use the pooled paths below.

## Current folder layout

```text
UCM_official_runs/
  production/equal_area_2026-09/
    01_temperature_invest3202/
    02_health_population2021/
    03_energy_work_capacity/
      derived_heavy_work_capacity/
      summary/                         Figure 4–5 PNG/CSV/JSON together
  archive/
    legacy_runs/
    diagnostic_runs/
    reference_outputs/
```

Browse `production/` or `archive/`, not the old-name links at the root. Old
manifests and historical scripts retain their recorded paths and use those
links; active production scripts/configurations/docs use the canonical paths.
The nested `hothaps_work_intensity` name is likewise a compatibility link.
Root-level loose diagnostic files were left untouched.

The exact [old-to-new map](../../code/workflows/configs/run-layout-20261110.json)
and [migration/rollback utility](../../code/workflows/migrate_run_layout.py)
record all 49 moves (48 top-level folders plus Hothaps). The utility defaults
to a read-only preflight; `--apply --rollback` restores mapped folders and removes
only their verified compatibility links. It does not roll back code/doc changes.
Use `--root /absolute/path/to/UCM_official_runs`.

**Other computers:** symlink synchronization through Google Drive is not
guaranteed. Active canonical paths do not require those links. Historical
absolute-path manifests remain machine-specific; restore/check old-path aliases
locally before using legacy scripts or resuming historical tasks. Do not claim
cross-machine validation based on this local migration.

## Versions and major decisions

All production rows below belong to **one revised analytical version**:
`equal_area_2026-09`. Numbers 01–03 denote workflow stages, not successive versions.
Dates distinguish folder labels from recorded execution dates; historical dates
not verified from manifests are explicitly left unconfirmed.

| Stage in analytical version | Date / evidence | Folder or group | Major decision and reason |
|---|---|---|---|
| Historical V0 | Before revised September package; exact run dates vary/unconfirmed | `archive/legacy_runs/` | Earlier canopy trials, area mortality allocation and borough summaries; retain as comparison evidence, not current inputs. |
| Intermediate revisions and diagnostics | Before complete September package; exact dates not re-audited | `archive/diagnostic_runs/` (`health_v2*`, InVEST/NoData/Target30 checks) | Separate software correction, canopy mismatch and grid-mask effects rather than attributing every difference to targeting. |
| **Same revised version — 01 temperature** | Folder originally dated 10 Sep 2026; run manifests dated 11–12 Sep UTC | `production/equal_area_2026-09/01_temperature_invest3202/` | Equal realized added-canopy area within each Green/Target tier; common InVEST 3.20.2/grid. Compare planting location at matched canopy amount. |
| **Same revised version — 02 health** | Package label 10 Sep 2026; exact execution date not established here | `production/equal_area_2026-09/02_health_population2021/` | Downstream of matched temperature contrasts; allocate fixed 2021 borough deaths by population so health burden follows residents rather than land area. |
| **Same revised version — 03 energy/WBGT** | Folder label 11 Sep 2026; UCM run manifests dated 11–12 Sep UTC | `production/equal_area_2026-09/03_energy_work_capacity/` | Companion UCM executions with the same scenario design, energy valuation and WBGT enabled; not a newer scientific version. |
| **Same revised version — derived Hothaps capacity** | Original Green30 manifest: 11 Sep 2026 UTC; additional scenarios added later | `03_energy_work_capacity/derived_heavy_work_capacity/` | Continuous WBGT-response post-processing for capacity instead of built-in threshold work-loss classes; distinguishes small scenario changes. |
| **Same UCM version — corrected reporting** | Initial Figure 4 manifest: 13 Sep 2026 UTC; boundary/accounting and artwork corrected 10 Nov 2026 | `03_energy_work_capacity/summary/` | London-only common capacity mask; conserved borough energy allocation; citywide Fig. 4 and borough Fig. 5. PNG/CSV/JSON pooled with descriptive prefixes; previous summaries archived. No new UCM run. |

The historical folder names in the table are retained basenames under `archive/`;
the mapping records their exact group. Historical classification follows existing project audits, not a new reconstruction
of every archived run. Other exploratory and aggregate folders are marked
reference/unclassified in their local notes rather than inferred from their names.

## Plain-language definitions and dependencies

**Equal-area temperature production** means running the temperature model on
Green and Target land-cover scenarios that add the same *realized canopy area*
within each 10/20/30 tier. It does not mean equal temperatures, uniform planting,
equal numbers of trees or equal costs. Earlier trial scenarios did not consistently
match canopy amounts; matching them helps distinguish the effect of *where*
canopy is added from *how much* is added. AllBuilt/TreeRisk/TreeOpp are separate
counterfactuals, not equal-area planting pairs.

**Population-based health** means allocating each borough's observed 2021 deaths
to pixels in proportion to their 2021 resident population, preserving the borough
total. This replaces uniform allocation by land area, which assigns deaths even
to unpopulated land. It assumes equal per-resident baseline risk within a borough;
it is not observed pixel mortality or a future population projection. Reporting
deaths averted per million is a separate normalization step.

```text
One scenario/input version: equal_area_2026-09
  01 temperature contrasts ───────────────> 02 health outcomes
  03 companion UCM run -> building energy ───────────────┐
                       -> WBGT -> Hothaps capacity ──────┤
  02 health outcomes ──────────────────────────────────┤
                                             summary/ -> Figures 4–5
```

Stage 03 reruns the same scenario design with optional valuation outputs enabled;
it is not derived by passing stage 01 temperature files into a new UCM model.
The health configuration uses stage 01 baseline/Green/Target temperatures and,
for the added AllBuilt/TreeRisk/TreeOpp 25°C cases, stage 03 scenario temperatures.
Thus the exact configuration, not numbering alone, defines input lineage.

## Why Hothaps stays inside the valuation folder

```text
production/equal_area_2026-09/03_energy_work_capacity/
  baseline/, allbuilt/, ..., target30/  UCM temperature, building energy, WBGT
  derived_heavy_work_capacity/               Derived continuous capacity by scenario
  summary/                             Citywide/borough CSVs and manifests
    figure4_*.png / *_data.csv          Citywide results
    figure5_*.png / *_data.csv          Borough results
```

`calculate_hothaps_workability.py` reads each UCM WBGT raster and writes
`0.1 + 0.9 / (1 + (WBGT / 30.94)^16.64)`. Output is a fraction; scenario minus
baseline multiplied by 100 gives percentage points, not relative percent change.
`summarize_ucm_valuations.py` and the Hothaps writer now use this renamed nested
folder. A compatibility link preserves old manifest paths; provenance is not rewritten.

### Is Hothaps an InVEST replacement or improvement?

It is a **modified work-capacity component of this study's modelling framework**,
implemented through separate post-processing rather than edits to InVEST source
code. `calculate_hothaps_workability.py` replaces the built-in threshold-based
work-loss indicator for this study with a continuous Hothaps function applied to
InVEST-derived WBGT. InVEST's cooling, temperature, WBGT and energy calculations
remain unchanged. Built-in work-loss outputs are retained but not used as the
manuscript measure. This component-level model modification represents small WBGT
changes continuously; improved predictive accuracy for London workers has not
been demonstrated, and the results are not worker-weighted. The
future exposure-weighted improvement is documented separately in
[future work-capacity methods](FUTURE_WORK_CAPACITY_METHOD.md).

## Decisions already made versus next decisions

10 November reporting correction: capacity now uses 15,733,956 common valid
London cells, not the wider 23,089,980-cell domain. Energy excludes 70 wholly
outside buildings and splits cross-borough totals by normalized intersection
area. The primary London estimate and equal-borough sensitivity are exported
together in `summary/figure45_aggregation_comparison.{csv,md}`. Historical full-domain
values are labelled as superseded. See the [updated audit and legends](../manuscript/FIGURE4_5_AUDIT.md).

Already implemented: common InVEST 3.20.2, equal realized canopy budgets,
fixed 2021 demographics, population mortality allocation, continuous Hothaps,
and separate citywide/borough estimands. Modelled capacity is area-weighted,
not worker-weighted; the worker-exposure upgrade remains future work.

Next author review: approve the consistent nine-scenario Figure 4/5 package;
inspect citywide versus equal-borough sensitivity; then update both figures,
Results and legends together. Do not select a run merely by the newest folder
date. Pending draft prose is in [the audit](../manuscript/FIGURE4_5_AUDIT.md).

Detailed evidence (not duplicated here):

- [InVEST algorithm/version diagnosis](../../code/Urban_Cooling_Modeling_Runs/INVEST_VERSION_COMPARISON.md)
- [Health data and allocation decisions](../../code/health_assessment/DATA_SELECTION_AND_METHOD_DECISIONS.md)
- [V0–V1 controlled health comparisons](../manuscript/V0_V1_HEALTH_AND_FIGURE6_CHANGES.md)
- [Production workflow and Hothaps commands](../../code/post_processing_layers/README.md)
- [Figure 4 estimands and uncertainty](../../code/post_processing_layers/FIGURE4_METHODS.md)

## Maintenance rule

Migration validation on 10 November 2026: all 49 compatibility links resolved
to their mapped directories; all 21 checked Figure 4 input/table checksums
still matched; all 20 Hothaps input/output manifest paths resolved; all 37
run-root paths in the revised health configuration existed. Seven workflow tests
(including nested migration, repeat execution, collision safety and rollback)
and five post-processing tests passed. Figure builders were exercised on test
fixtures, not production outputs. No production model or figure was regenerated.
Cloud synchronization and a second-computer rebuild were not tested.

For a new run, create a new versioned directory, retain its manifests and add a
local README stating inputs, parent run, differences and review status. Update
this index and downstream figure manifests only after validation. Never silently
replace the selected source package or erase historical outputs.
