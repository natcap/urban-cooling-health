# Urban cooling, health and equity in London

This repository contains the research code for comparing how alternative
land-cover and tree-planting strategies affect urban cooling, co-benefits and
the distribution of heat-related health benefits across London. It is under
active manuscript revision; the [project TODO](TODO.md) distinguishes completed
analyses from remaining scientific and release decisions.

The analysis holds **2021 population and registered mortality fixed** while
evaluating mid-century temperature scenarios. The primary manuscript setting
uses assumed 25/5°C reference-temperature/UHI conditions for cooling contrasts;
seasonal health and work capacity use reconstructed daily summer exposures.
The 28/5°C setting is an additional assumed climate comparison. These are not
projections of London's 2050 population or mortality. Display labels are defined in
[`code/func_colors.R`](code/func_colors.R); the current comparison includes
baseline, AllBuilt, TreeRisk, TreeOpp, Green10/20/30 and Target10/20/30.

## Repository map

```text
urban-cooling-health/
├── README.md                 Project overview and navigation
├── REPRODUCIBILITY.md        Current Figure 7 procedure
├── TODO.md                   Open scientific and release gates
├── docs/
│   ├── README.md             Documentation index
│   ├── manuscript/index.md   Current figure text and decision index
│   └── WORKFLOW.md           Ordered source-to-figure workflow
├── code/
│   └── README.md             Production scripts and historical analyses
├── data/
│   ├── README.md             Local data roles and handling
│   └── derived/              Selected tracked Figure 7 inputs
└── figures/
    ├── manuscript_current/   Selected Figures 1–7 and supplements in one folder
    └── ...                   Generation outputs, diagnostics and historical evidence
```

Start with the links below rather than inferring which script to run from a
filename. Large local data are ignored by Git; additional source/model data
live in the external project store. Neither is included in a Git push.

## Key input data

| Input | What it contributes | Detailed documentation |
|---|---|---|
| Land cover and tree canopy | UKCEH LCM2023 land cover plus the project's 2024 canopy overlay define the baseline and planting eligibility. Codes 1–21 are land-cover classes; 100 is the project canopy class. | [Complete code–name table and scenario roles](code/lc_scenarios/README.md#complete-land-cover-code-list) |
| Climate and weather | CHELSA July 2021 provides the recent historical reference; WorldClim projections and MIDAS station weather support reconstructed daily summer exposures. | [Temperature reconstruction and assumptions](docs/manuscript/temperature_reconstruction_comparison.md) |
| Population and mortality | Fixed WorldPop 2021 population and ONS/Nomis borough deaths support population-based mortality allocation; seasonal estimates additionally use cause-specific monthly mortality fractions. | [Population and annual mortality selection](code/health_assessment/DATA_SELECTION_AND_METHOD_DECISIONS.md) and [seasonal-health analysis](docs/manuscript/seasonal_health_rerun_comparison.md) |
| Neighbourhood vulnerability | The four-dimension Social Vulnerability Index (SVI) and its component indicators support targeting and neighbourhood equity comparisons. | [SVI construction and comparison](docs/manuscript/SVI_SPATIAL_RESULTS.md) |
| Buildings and electricity demand | Colouring Britain building footprints, demand-response assumptions and tariffs support conditional electricity-saving estimates. | [Energy inputs and allocation](docs/manuscript/energy_capacity_candidate_results.md) |

For storage locations and raw-versus-derived data handling, see the
[data directory guide](data/README.md). Use the linked stage documentation and
run manifests for exact input versions; this overview is not a download inventory.

## Where to start

### Run-version clarification — 10 November 2026

The [run-version and decision index](docs/methods/RUN_VERSION_INDEX.md) distinguishes
historical trials from the current temperature, health and valuation packages.
`derived_heavy_work_capacity` is derived WBGT post-processing within the valuation
package, not another UCM run. Each external run folder now has a concise
`README_RUN.md`. Real folders are now grouped under `production/equal_area_2026-09/`
and `archive/`; old names remain compatibility links. Scientific outputs are
unchanged, and active script/configuration paths have been updated. Figures 4–5 will be
reviewed together after source-package confirmation. The saved
[audit and draft Abstract/Figure 4 text](docs/manuscript/figure4_5_audit.md)
records the numerical corrections and current author decisions.

Stages 01–03 are **one revised version**, not competing versions: matched-canopy
temperature scenarios feed population-allocated health estimates, alongside
energy/WBGT and a modified work-capacity component using Hothaps. This separate
post-processing leaves InVEST's cooling, WBGT and energy calculations unchanged.
Figure 4–5 PNGs and their CSV/JSON
sources now share the valuation `summary/` folder; PDF/SVG exports are retired
to a recoverable archive. The run index defines the terms, dates and rationale.

Figures 4–5 now use the corrected London-only capacity boundary and conserved
borough energy accounting. The pooled summary includes the equal-borough
sensitivity and a comparison table; see the audit above before transferring
numbers into Word. UCM temperature and health-model outputs are unchanged.

| You want to… | Open this guide | External project data? |
|---|---|---:|
| Reproduce the current Figure 7 from tracked derived inputs | [Figure 7 reproduction](REPRODUCIBILITY.md) | No |
| Rebuild scenarios, model results and figures in order | [Full workflow map](docs/WORKFLOW.md) | Yes |
| Find the current production scripts | [Code/figure index](code/README.md) | Depends on stage |
| Find the latest manuscript figures and copy-ready text | [Current figure folder](figures/manuscript_current/) and [manuscript index](docs/manuscript/index.md) | No |
| Understand local raw, processed and derived data | [Data directory guide](data/README.md) | No |
| Review scientific decisions, open tasks and manuscript status | [Documentation index](docs/README.md) and [TODO](TODO.md) | No |
| Verify selected figures or prepare a full rebuild | [Reproducibility guide](docs/reproducibility.md) | Figure verification: yes; full rebuild: yes |
| Check whether data or figures can be shared | [Licence and redistribution review](DATA_LICENCE_AND_REDISTRIBUTION.md) | No |

The repository tracks the small derived inputs required for the current
Figure 7 run. A full rebuild requires authorized access to the separately
stored, versioned geospatial inputs; raw UKCEH land cover, scenario and UCM
rasters are not included here. The [full workflow map](docs/WORKFLOW.md)
identifies each stage's authoritative instructions and validation gate.

## Quick check: Figure 7

After installing the R packages listed in the [Figure 7 guide](REPRODUCIBILITY.md),
make a first run in a separate, ignored output folder from the repository root:

```bash
URBAN_COOLING_FIG7_OUTPUT_DIR=./figures/fig7_local_check Rscript code/run-fig7.R
```

The reviewed production panels and supporting tables are under
[`figures/figure7/`](figures/figure7/);
the command above leaves them untouched.
See the [Figure 7 guide](REPRODUCIBILITY.md) for installation, exact inputs,
output files, a non-destructive test run and the final review checklist.

## Current boundaries

- The revised Green/Target comparison uses matched **realized added-canopy
  area**, not equal financial cost or equal numbers of physical trees. The
  [scenario guide](code/lc_scenarios/README.md) defines eligible classes,
  provenance and the equal-budget validation.
- Figure 4 combines citywide energy, area-weighted modelled heavy-work
  capacity and population-allocated mortality estimates. The
  [Figure 4 methods](docs/manuscript/figure4_methods.md) define
  these estimands and the borough sensitivity.
- Figures 6–7 use the current seasonal-health candidates. Figure 6 remains a
  descriptive ecological analysis with a separate spatial sensitivity; Figure
  7 uses paired LSOAs and fixed Green30 benefit thresholds. See the
  [seasonal-health report](docs/manuscript/seasonal_health_rerun_comparison.md)
  and [current manuscript index](docs/manuscript/index.md).
- Some historical notebooks and Windows launchers retain machine-specific
  paths. Start new production runs from the [workflow guide](code/workflows/README.md)
  and its versioned configuration, not from an archived launcher.

## Reuse and contributions

Keep run configurations, input checksums, source versions and figure-source
tables with any revised analysis. Do not commit credentials or restricted
source rasters. Report the scenario, data snapshot and output affected when
opening a GitHub issue or proposing a change.

The repository [`LICENSE`](LICENSE) does not, by itself, grant reuse rights to
third-party datasets or derived products with separate source terms. Review
the [data licence and redistribution record](DATA_LICENCE_AND_REDISTRIBUTION.md)
before releasing or reusing data and figures.
