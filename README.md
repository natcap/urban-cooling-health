# Urban cooling, health and equity in London

This repository contains the research code for comparing how alternative
land-cover and tree-planting strategies affect urban cooling, co-benefits and
the distribution of heat-related health benefits across London. It is under
active manuscript revision; the [project TODO](TODO.md) distinguishes completed
analyses from remaining scientific and release decisions.

The analysis holds **2021 population and registered mortality fixed** while
evaluating mid-century temperature scenarios. The primary manuscript setting
is 25°C; 28°C is a sensitivity. These are not projections of London's 2050
population or mortality. Scenario names and display labels are defined in
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
│   └── WORKFLOW.md           Ordered source-to-figure workflow
├── code/
│   └── README.md             Production scripts and historical analyses
├── data/
│   ├── README.md             Local data roles and handling
│   └── derived/              Selected tracked Figure 7 inputs
└── figures/                  Current Figure 7 outputs and historical evidence
```

Start with the links below rather than inferring which script to run from a
filename. Large source/model data are kept outside this repository.

## Where to start

| You want to… | Open this guide | External project data? |
|---|---|---:|
| Reproduce the current Figure 7 from tracked derived inputs | [Figure 7 reproduction](REPRODUCIBILITY.md) | No |
| Rebuild scenarios, model results and figures in order | [Full workflow map](docs/WORKFLOW.md) | Yes |
| Find the current production scripts | [Code/figure index](code/README.md) | Depends on stage |
| Understand local raw, processed and derived data | [Data directory guide](data/README.md) | No |
| Review scientific decisions, open tasks and manuscript status | [Documentation index](docs/README.md) and [TODO](TODO.md) | No |
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
  [Figure 4 methods](code/post_processing_layers/FIGURE4_METHODS.md) define
  these estimands and the borough sensitivity.
- Figure 6 is exploratory until its spatial-inference review is complete.
  Figure 7 is the current paired equity workflow. See the
  [Figures 6–7 review](code/health_assessment/FIGURES6_7_REVIEW.md) and
  [manuscript brief](docs/manuscript/MANUSCRIPT_REVISION_BRIEF.md).
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
